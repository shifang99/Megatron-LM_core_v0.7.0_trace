# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass                                              # trace_info : t_6338
from typing import Optional, Tuple, Union                                      # trace_info : t_6339

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__(config=config)                                        # trace_info : t_7285, t_8287

        self.config: TransformerConfig = config                                # trace_info : t_7288, t_8290

        self.input_size = input_size if input_size != None else self.config.hidden_size# trace_info : t_7289, t_8291

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size                          # trace_info : t_7290, t_8292
        if self.config.gated_linear_unit:                                      # trace_info : t_7291, t_8293
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(                                        # trace_info : t_7292, t_7303, t_8294, t_8305
            submodules.linear_fc1,                                             # trace_info : t_7293, t_8295
            self.input_size,                                                   # trace_info : t_7294, t_8296
            ffn_hidden_size,                                                   # trace_info : t_7295, t_8297
            config=self.config,                                                # trace_info : t_7296, t_8298
            init_method=self.config.init_method,                               # trace_info : t_7297, t_8299
            gather_output=False,                                               # trace_info : t_7298, t_8300
            bias=self.config.add_bias_linear,                                  # trace_info : t_7299, t_8301
            skip_bias_add=True,                                                # trace_info : t_7300, t_8302
            is_expert=is_expert,                                               # trace_info : t_7301, t_8303
            tp_comm_buffer_name='fc1',                                         # trace_info : t_7302, t_8304
        )

        self.activation_func = self.config.activation_func                     # trace_info : t_7450, t_8452

        self.linear_fc2 = build_module(                                        # trace_info : t_7451, t_7462, t_8453, t_8464
            submodules.linear_fc2,                                             # trace_info : t_7452, t_8454
            self.config.ffn_hidden_size,                                       # trace_info : t_7453, t_8455
            self.config.hidden_size,                                           # trace_info : t_7454, t_8456
            config=self.config,                                                # trace_info : t_7455, t_8457
            init_method=self.config.output_layer_init_method,                  # trace_info : t_7456, t_8458
            bias=self.config.add_bias_linear,                                  # trace_info : t_7457, t_8459
            input_is_parallel=True,                                            # trace_info : t_7458, t_8460
            skip_bias_add=True,                                                # trace_info : t_7459, t_8461
            is_expert=is_expert,                                               # trace_info : t_7460, t_8462
            tp_comm_buffer_name='fc2',                                         # trace_info : t_7461, t_8463
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)  # trace_info : t_15616, t_16117, t_19255, t_19756, t_22894, ...

        if self.config.bias_activation_fusion:                                 # trace_info : t_15676, t_16177, t_19315, t_19816, t_22954, ...
            if self.activation_func == F.gelu:                                 # trace_info : t_15677, t_16178, t_19316, t_19817, t_22955, ...
                if self.config.gated_linear_unit:                              # trace_info : t_15678, t_16179, t_19317, t_19818, t_22956, ...
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True                 # trace_info : t_15679, t_16180, t_19318, t_19819, t_22957, ...
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)# trace_info : t_15680, t_16181, t_19319, t_19820, t_22958, ...
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)           # trace_info : t_15684, t_16185, t_19323, t_19824, t_22962, ...

        return output, output_bias                                             # trace_info : t_15740, t_16241, t_19379, t_19880, t_23018, ...

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                sub_sd = self._sharded_state_dict_for_glu(
                    name, module, prefix, sharded_offsets, metadata
                )
            else:
                sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def _sharded_state_dict_for_glu(
        self,
        module_name: str,
        module: torch.nn.Module,
        prefix: str,
        sharded_offsets: Tuple[Tuple[int, int, int]],
        metadata: Optional[dict] = None,
    ):
        assert module_name == 'linear_fc1', module_name
        sharded_state_dict = module.sharded_state_dict(
            f'{prefix}{module_name}.', sharded_offsets, metadata
        )
        weight_key = f'{prefix}{module_name}.weight'
        prev_sh_ten = sharded_state_dict[weight_key]

        # We must split the tensor into 2 parts, each sharded separately.
        # This requires a ShardedTensorFactory which `chunk`s during saving
        # and `cat`s during loading
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        tp_shard_axis = 0
        prepend_axis_num = len(sharded_offsets)

        def sh_ten_build_fn(key: str, t: torch.Tensor, replica_id: ReplicaId):
            offset_w = (tp_shard_axis + prepend_axis_num, tp_rank, tp_size * 2)
            offset_v = (tp_shard_axis + prepend_axis_num, tp_size + tp_rank, tp_size * 2)
            with torch.no_grad():
                tensor_w, tensor_v = torch.chunk(t, 2, dim=tp_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]

        def sh_ten_merge_fn(sub_state_dict):
            with torch.no_grad():
                return torch.cat(sub_state_dict)

        sharded_state_dict[weight_key] = ShardedTensorFactory(
            prev_sh_ten.key,
            prev_sh_ten.data,
            sh_ten_build_fn,
            sh_ten_merge_fn,
            prev_sh_ten.replica_id,
        )
        return sharded_state_dict
