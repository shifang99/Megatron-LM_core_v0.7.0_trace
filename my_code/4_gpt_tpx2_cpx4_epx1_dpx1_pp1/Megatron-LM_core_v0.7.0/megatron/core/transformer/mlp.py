# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass                                              # trace_info : t_9648
from typing import Optional, Tuple, Union                                      # trace_info : t_9649

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
        super().__init__(config=config)                                        # trace_info : t_10454, t_11232

        self.config: TransformerConfig = config                                # trace_info : t_10457, t_11235

        self.input_size = input_size if input_size != None else self.config.hidden_size# trace_info : t_10458, t_11236

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size                          # trace_info : t_10459, t_11237
        if self.config.gated_linear_unit:                                      # trace_info : t_10460, t_11238
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(                                        # trace_info : t_10461, t_10472, t_11239, t_11250
            submodules.linear_fc1,                                             # trace_info : t_10462, t_11240
            self.input_size,                                                   # trace_info : t_10463, t_11241
            ffn_hidden_size,                                                   # trace_info : t_10464, t_11242
            config=self.config,                                                # trace_info : t_10465, t_11243
            init_method=self.config.init_method,                               # trace_info : t_10466, t_11244
            gather_output=False,                                               # trace_info : t_10467, t_11245
            bias=self.config.add_bias_linear,                                  # trace_info : t_10468, t_11246
            skip_bias_add=True,                                                # trace_info : t_10469, t_11247
            is_expert=is_expert,                                               # trace_info : t_10470, t_11248
            tp_comm_buffer_name='fc1',                                         # trace_info : t_10471, t_11249
        )

        self.activation_func = self.config.activation_func                     # trace_info : t_10564, t_11342

        self.linear_fc2 = build_module(                                        # trace_info : t_10565, t_10576, t_11343, t_11354
            submodules.linear_fc2,                                             # trace_info : t_10566, t_11344
            self.config.ffn_hidden_size,                                       # trace_info : t_10567, t_11345
            self.config.hidden_size,                                           # trace_info : t_10568, t_11346
            config=self.config,                                                # trace_info : t_10569, t_11347
            init_method=self.config.output_layer_init_method,                  # trace_info : t_10570, t_11348
            bias=self.config.add_bias_linear,                                  # trace_info : t_10571, t_11349
            input_is_parallel=True,                                            # trace_info : t_10572, t_11350
            skip_bias_add=True,                                                # trace_info : t_10573, t_11351
            is_expert=is_expert,                                               # trace_info : t_10574, t_11352
            tp_comm_buffer_name='fc2',                                         # trace_info : t_10575, t_11353
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)  # trace_info : t_18414, t_18621, t_21600, t_21807, t_24786, ...

        if self.config.bias_activation_fusion:                                 # trace_info : t_18421, t_18628, t_21607, t_21814, t_24793, ...
            if self.activation_func == F.gelu:                                 # trace_info : t_18422, t_18629, t_21608, t_21815, t_24794, ...
                if self.config.gated_linear_unit:                              # trace_info : t_18423, t_18630, t_21609, t_21816, t_24795, ...
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True                 # trace_info : t_18424, t_18631, t_21610, t_21817, t_24796, ...
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)# trace_info : t_18425, t_18632, t_21611, t_21818, t_24797, ...
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
        output, output_bias = self.linear_fc2(intermediate_parallel)           # trace_info : t_18429, t_18636, t_21615, t_21822, t_24801, ...

        return output, output_bias                                             # trace_info : t_18436, t_18643, t_21622, t_21829, t_24808, ...

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
