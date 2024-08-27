# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.transformer_config import TransformerConfig


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        if config.moe_extended_tp:
            tp_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        else:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()

        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        raise NotImplementedError(
            'Currently distributed checkpointing is not supported for GroupedMLP'
        )


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.
    
    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)                                        # trace_info : t_10340, t_11480
        self.add_bias = config.add_bias_linear                                 # trace_info : t_10343, t_11483
        self.moe_extended_tp = config.moe_extended_tp                          # trace_info : t_10344, t_11484
        self.num_local_experts = num_local_experts                             # trace_info : t_10345, t_11485
        self.local_experts = torch.nn.ModuleList()                             # trace_info : t_10346, t_11486
        for _ in range(self.num_local_experts):                                # trace_info : t_10347, t_10663, t_11487, t_11803
            expert = MLP(self.config, submodules, is_expert=True)              # trace_info : t_10348, t_11488
            self.local_experts.append(expert)                                  # trace_info : t_10662, t_11802

    def forward(self, permuted_local_hidden_states, tokens_per_expert):

        output_local = torch.zeros_like(permuted_local_hidden_states)          # trace_info : t_18849, t_19604, t_23194, t_23949, t_27539, ...
        output_bias_local = None                                               # trace_info : t_18850, t_19605, t_23195, t_23950, t_27540, ...
        if self.add_bias:                                                      # trace_info : t_18851, t_19606, t_23196, t_23951, t_27541, ...
            output_bias_local = torch.zeros_like(permuted_local_hidden_states) # trace_info : t_18852, t_19607, t_23197, t_23952, t_27542, ...

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)             # trace_info : t_18853, t_19608, t_23198, t_23953, t_27543, ...
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)# trace_info : t_18854, t_19609, t_23199, t_23954, t_27544, ...
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))        # trace_info : t_18855, t_19610, t_23200, t_23955, t_27545, ...
        for expert_num, expert in enumerate(self.local_experts):               # trace_info : t_18856, t_18975, t_19611, t_19730, t_23201, ...
            start = cumsum_num_tokens[expert_num]                              # trace_info : t_18857, t_19612, t_23202, t_23957, t_27547, ...
            end = cumsum_num_tokens[expert_num + 1]                            # trace_info : t_18858, t_19613, t_23203, t_23958, t_27548, ...
            hidden = permuted_local_hidden_states[start:end]                   # trace_info : t_18859, t_19614, t_23204, t_23959, t_27549, ...
            output, output_bias = expert(hidden)                               # trace_info : t_18860, t_19615, t_23205, t_23960, t_27550, ...

            output_local[start:end] = output                                   # trace_info : t_18971, t_19726, t_23316, t_24071, t_27661, ...
            if self.add_bias:                                                  # trace_info : t_18972, t_19727, t_23317, t_24072, t_27662, ...
                output_bias = output_bias.expand_as(output)                    # trace_info : t_18973, t_19728, t_23318, t_24073, t_27663, ...
                output_bias_local[start:end, :] = output_bias                  # trace_info : t_18974, t_19729, t_23319, t_24074, t_27664, ...

        return output_local, output_bias_local                                 # trace_info : t_18976, t_19731, t_23321, t_24076, t_27666, ...

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Maps local expert to global experts. """
        if self.moe_extended_tp:
            raise NotImplementedError(
                'Currently distributed checkpointing is not supported for moe_extended_tp'
            )

        sharded_state_dict = {}
        num_global_experts = (
            parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        expert_sharded_prefix = f'{prefix}experts.'
        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(),
                )

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
