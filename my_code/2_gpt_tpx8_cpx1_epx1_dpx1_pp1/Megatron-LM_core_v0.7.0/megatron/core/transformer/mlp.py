# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass                                              # trace_info : t_11205
from typing import Optional, Tuple, Union                                      # trace_info : t_11206

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
        super().__init__(config=config)                                        # trace_info : t_12152, t_13154

        self.config: TransformerConfig = config                                # trace_info : t_12155, t_13157

        self.input_size = input_size if input_size != None else self.config.hidden_size# trace_info : t_12156, t_13158

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size                          # trace_info : t_12157, t_13159
        if self.config.gated_linear_unit:                                      # trace_info : t_12158, t_13160
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(                                        # trace_info : t_12159, t_12170, t_13161, t_13172
            submodules.linear_fc1,                                             # trace_info : t_12160, t_13162
            self.input_size,                                                   # trace_info : t_12161, t_13163
            ffn_hidden_size,                                                   # trace_info : t_12162, t_13164
            config=self.config,                                                # trace_info : t_12163, t_13165
            init_method=self.config.init_method,                               # trace_info : t_12164, t_13166
            gather_output=False,                                               # trace_info : t_12165, t_13167
            bias=self.config.add_bias_linear,                                  # trace_info : t_12166, t_13168
            skip_bias_add=True,                                                # trace_info : t_12167, t_13169
            is_expert=is_expert,                                               # trace_info : t_12168, t_13170
            tp_comm_buffer_name='fc1',                                         # trace_info : t_12169, t_13171
        )

        self.activation_func = self.config.activation_func                     # trace_info : t_12317, t_13319

        self.linear_fc2 = build_module(                                        # trace_info : t_12318, t_12329, t_13320, t_13331
            submodules.linear_fc2,                                             # trace_info : t_12319, t_13321
            self.config.ffn_hidden_size,                                       # trace_info : t_12320, t_13322
            self.config.hidden_size,                                           # trace_info : t_12321, t_13323
            config=self.config,                                                # trace_info : t_12322, t_13324
            init_method=self.config.output_layer_init_method,                  # trace_info : t_12323, t_13325
            bias=self.config.add_bias_linear,                                  # trace_info : t_12324, t_13326
            input_is_parallel=True,                                            # trace_info : t_12325, t_13327
            skip_bias_add=True,                                                # trace_info : t_12326, t_13328
            is_expert=is_expert,                                               # trace_info : t_12327, t_13329
            tp_comm_buffer_name='fc2',                                         # trace_info : t_12328, t_13330
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)  # trace_info : t_20477, t_20962, t_24087, t_24572, t_27697, ...

        if self.config.bias_activation_fusion:                                 # trace_info : t_20529, t_21014, t_24139, t_24624, t_27749, ...
            if self.activation_func == F.gelu:                                 # trace_info : t_20530, t_21015, t_24140, t_24625, t_27750, ...
                if self.config.gated_linear_unit:                              # trace_info : t_20531, t_21016, t_24141, t_24626, t_27751, ...
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True                 # trace_info : t_20532, t_21017, t_24142, t_24627, t_27752, ...
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)# trace_info : t_20533, t_21018, t_24143, t_24628, t_27753, ...
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
        output, output_bias = self.linear_fc2(intermediate_parallel)           # trace_info : t_20537, t_21022, t_24147, t_24632, t_27757, ...

        return output, output_bias                                             # trace_info : t_20597, t_21082, t_24207, t_24692, t_27817, ...

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
