# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass                                              # trace_info : t_9367
from typing import Optional, Tuple, Union                                      # trace_info : t_9368

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
        super().__init__(config=config)                                        # trace_info : t_10314, t_11316

        self.config: TransformerConfig = config                                # trace_info : t_10317, t_11319

        self.input_size = input_size if input_size != None else self.config.hidden_size# trace_info : t_10318, t_11320

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size                          # trace_info : t_10319, t_11321
        if self.config.gated_linear_unit:                                      # trace_info : t_10320, t_11322
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(                                        # trace_info : t_10321, t_10332, t_11323, t_11334
            submodules.linear_fc1,                                             # trace_info : t_10322, t_11324
            self.input_size,                                                   # trace_info : t_10323, t_11325
            ffn_hidden_size,                                                   # trace_info : t_10324, t_11326
            config=self.config,                                                # trace_info : t_10325, t_11327
            init_method=self.config.init_method,                               # trace_info : t_10326, t_11328
            gather_output=False,                                               # trace_info : t_10327, t_11329
            bias=self.config.add_bias_linear,                                  # trace_info : t_10328, t_11330
            skip_bias_add=True,                                                # trace_info : t_10329, t_11331
            is_expert=is_expert,                                               # trace_info : t_10330, t_11332
            tp_comm_buffer_name='fc1',                                         # trace_info : t_10331, t_11333
        )

        self.activation_func = self.config.activation_func                     # trace_info : t_10479, t_11481

        self.linear_fc2 = build_module(                                        # trace_info : t_10480, t_10491, t_11482, t_11493
            submodules.linear_fc2,                                             # trace_info : t_10481, t_11483
            self.config.ffn_hidden_size,                                       # trace_info : t_10482, t_11484
            self.config.hidden_size,                                           # trace_info : t_10483, t_11485
            config=self.config,                                                # trace_info : t_10484, t_11486
            init_method=self.config.output_layer_init_method,                  # trace_info : t_10485, t_11487
            bias=self.config.add_bias_linear,                                  # trace_info : t_10486, t_11488
            input_is_parallel=True,                                            # trace_info : t_10487, t_11489
            skip_bias_add=True,                                                # trace_info : t_10488, t_11490
            is_expert=is_expert,                                               # trace_info : t_10489, t_11491
            tp_comm_buffer_name='fc2',                                         # trace_info : t_10490, t_11492
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)  # trace_info : t_18764, t_19257, t_22401, t_22894, t_90008, ...

        if self.config.bias_activation_fusion:                                 # trace_info : t_18816, t_19309, t_22453, t_22946, t_90060, ...
            if self.activation_func == F.gelu:                                 # trace_info : t_18817, t_19310, t_22454, t_22947, t_90061, ...
                if self.config.gated_linear_unit:                              # trace_info : t_18818, t_19311, t_22455, t_22948, t_90062, ...
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True                 # trace_info : t_18819, t_19312, t_22456, t_22949, t_90063, ...
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)# trace_info : t_18820, t_19313, t_22457, t_22950, t_90064, ...
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
        output, output_bias = self.linear_fc2(intermediate_parallel)           # trace_info : t_18824, t_19317, t_22461, t_22954, t_90068, ...

        return output, output_bias                                             # trace_info : t_18884, t_19377, t_22521, t_23014, t_90128, ...

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}                                                # trace_info : t_26263, t_28057, t_93856, t_95650
        for name, module in self._modules.items():                             # trace_info : t_26264, t_26498, t_26715, t_28058, t_28292, ...
            if name == 'linear_fc1' and self.config.gated_linear_unit:         # trace_info : t_26265, t_26499, t_28059, t_28293, t_93858, ...
                sub_sd = self._sharded_state_dict_for_glu(
                    name, module, prefix, sharded_offsets, metadata
                )
            else:
                sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)# trace_info : t_26266, t_26500, t_28060, t_28294, t_93859, ...
            sharded_state_dict.update(sub_sd)                                  # trace_info : t_26497, t_26714, t_28291, t_28508, t_94090, ...
        return sharded_state_dict                                              # trace_info : t_26716, t_28510, t_94309, t_96103

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
