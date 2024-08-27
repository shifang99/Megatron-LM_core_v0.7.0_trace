# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)                             # trace_info : t_10239, t_11379
        self.config = config                                                   # trace_info : t_10242, t_11382
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()# trace_info : t_10243, t_11383
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"# trace_info : t_10257, t_11397

        if self.config.moe_extended_tp:                                        # trace_info : t_10258, t_11398
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0# trace_info : t_10259, t_11399
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size# trace_info : t_10260, t_11400
            local_expert_indices_offset = (                                    # trace_info : t_10275, t_11415
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts# trace_info : t_10261, t_11401
            )

        self.local_expert_indices = [                                          # trace_info : t_10276, t_10278, t_11416, t_11418
            local_expert_indices_offset + i for i in range(self.num_local_experts)# trace_info : t_10277, t_11417
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))# trace_info : t_10279, t_10280, t_11419, t_11420
        self.router = None                                                     # trace_info : t_10281, t_11421
        self.experts = None                                                    # trace_info : t_10282, t_11422
        self.token_dispatcher = None                                           # trace_info : t_10283, t_11423
        self.layer_number = layer_number                                       # trace_info : t_10284, t_11424

    @abstractmethod
    def forward(self, hidden_states):
        pass

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number                                       # trace_info : t_10682, t_11822
        self.router.set_layer_number(layer_number)                             # trace_info : t_10683, t_11823


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules                                           # trace_info : t_10237, t_11377
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)# trace_info : t_10238, t_11378
        self.router = TopKRouter(config=self.config)                           # trace_info : t_10285, t_11425
        if self.config.moe_grouped_gemm:                                       # trace_info : t_10337, t_11477
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)                  # trace_info : t_10338, t_11478
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)# trace_info : t_10339, t_11479
        if config.moe_token_dispatcher_type == "allgather":                    # trace_info : t_10664, t_11804
            self.token_dispatcher = MoEAllGatherTokenDispatcher(               # trace_info : t_10665, t_10667, t_11805, t_11807
                self.num_local_experts, self.local_expert_indices, config=self.config# trace_info : t_10666, t_11806
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute                  # trace_info : t_10679, t_11819

    def forward(self, hidden_states: torch.Tensor):
        if (                                                                   # trace_info : t_18673, t_19431, t_23021, t_23776, t_27366, ...
            self.training                                                      # trace_info : t_18672, t_19430, t_23020, t_23775, t_27365, ...
            and self.config.tensor_model_parallel_size > 1                     # trace_info : t_18674, t_18676, t_19432, t_19434, t_23022, ...
            and not self.config.sequence_parallel                              # trace_info : t_18675, t_19433, t_23023, t_23778, t_27368, ...
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):                                     # trace_info : t_18677, t_19435, t_23025, t_23780, t_27370, ...
            probs, indices = self.router(hidden_states)                        # trace_info : t_18680, t_19438, t_23028, t_23783, t_27373, ...
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(# trace_info : t_18757, t_18759, t_19512, t_19514, t_23102, ...
                hidden_states, probs, indices                                  # trace_info : t_18758, t_19513, t_23103, t_23858, t_27448, ...
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)# trace_info : t_18848, t_19603, t_23193, t_23948, t_27538, ...
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)# trace_info : t_18977, t_19732, t_23322, t_24077, t_27667, ...
            return output, mlp_bias                                            # trace_info : t_19071, t_19826, t_23416, t_24171, t_27761, ...

        if self.moe_layer_recompute:                                           # trace_info : t_18678, t_19436, t_23026, t_23781, t_27371, ...
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)                   # trace_info : t_18679, t_19437, t_23027, t_23782, t_27372, ...

        return output, mlp_bias                                                # trace_info : t_19072, t_19827, t_23417, t_24172, t_27762, ...
