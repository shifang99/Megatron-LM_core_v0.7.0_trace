# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC                                                            # trace_info : t_9742
from dataclasses import dataclass, field                                       # trace_info : t_9743
from typing import Dict, Optional, Union                                       # trace_info : t_9744
                                                                               # trace_info : t_9745
import torch                                                                   # trace_info : t_9746
                                                                               # trace_info : t_9747
from megatron.core import parallel_state                                       # trace_info : t_9748
from megatron.core.dist_checkpointing.mapping import ShardedStateDict          # trace_info : t_9749
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping        # trace_info : t_9750
from megatron.core.transformer.enums import AttnMaskType                       # trace_info : t_9751
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor


@dataclass
class TransformerLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class BaseTransformerLayer(ABC):
    """ A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)                                        # trace_info : t_9980, t_10982
        self.submodules_config = submodules                                    # trace_info : t_9983, t_10985

        self.layer_number = layer_number + self._get_layer_offset()            # trace_info : t_9984, t_10986
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout# trace_info : t_10005, t_11007

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(                                   # trace_info : t_10006, t_10011, t_11008, t_11013
            submodules.input_layernorm,                                        # trace_info : t_10007, t_11009
            config=self.config,                                                # trace_info : t_10008, t_11010
            hidden_size=self.config.hidden_size,                               # trace_info : t_10009, t_11011
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10010, t_11012
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(                                    # trace_info : t_10048, t_10050, t_11050, t_11052
            submodules.self_attention, config=self.config, layer_number=layer_number,# trace_info : t_10049, t_11051
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)            # trace_info : t_10531, t_11533

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(                          # trace_info : t_10534, t_10539, t_11536, t_11541
            submodules.pre_cross_attn_layernorm,                               # trace_info : t_10535, t_11537
            config=self.config,                                                # trace_info : t_10536, t_11538
            hidden_size=self.config.hidden_size,                               # trace_info : t_10537, t_11539
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10538, t_11540
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(                                   # trace_info : t_10555, t_10557, t_11557, t_11559
            submodules.cross_attention, config=self.config, layer_number=layer_number,# trace_info : t_10556, t_11558
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config,)# trace_info : t_10573, t_11575

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(                                 # trace_info : t_10590, t_10595, t_11592, t_11597
            submodules.pre_mlp_layernorm,                                      # trace_info : t_10591, t_11593
            config=self.config,                                                # trace_info : t_10592, t_11594
            hidden_size=self.config.hidden_size,                               # trace_info : t_10593, t_11595
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10594, t_11596
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)            # trace_info : t_10632, t_11634
        if hasattr(self.mlp, 'set_layer_number'):                              # trace_info : t_10958, t_11960
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)                        # trace_info : t_10959, t_11961

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad                 # trace_info : t_10962, t_11964

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()      # trace_info : t_9985, t_10987

        num_layers_per_pipeline_rank = (                                       # trace_info : t_9995, t_10997
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_9990, t_10992
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:# trace_info : t_9996, t_10998
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:    # trace_info : t_9998, t_11000
                offset = pipeline_rank * num_layers_per_pipeline_rank          # trace_info : t_10003, t_11005
            else:
                offset = 0

        return offset                                                          # trace_info : t_10004, t_11006

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18410, t_18905, t_22140, t_22633, t_25868, ...

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)           # trace_info : t_18411, t_18906, t_22141, t_22634, t_25869, ...

        # Self attention.
        attention_output_with_bias = self.self_attention(                      # trace_info : t_18429, t_18435, t_18924, t_18930, t_22159, ...
            input_layernorm_output,                                            # trace_info : t_18430, t_18925, t_22160, t_22653, t_25888, ...
            attention_mask=attention_mask,                                     # trace_info : t_18431, t_18926, t_22161, t_22654, t_25889, ...
            inference_params=inference_params,                                 # trace_info : t_18432, t_18927, t_22162, t_22655, t_25890, ...
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_18433, t_18928, t_22163, t_22656, t_25891, ...
            packed_seq_params=packed_seq_params,                               # trace_info : t_18434, t_18929, t_22164, t_22657, t_25892, ...
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18704, t_18712, t_19197, t_19205, t_22432, ...
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18705, t_18710, t_19198, t_19203, t_22433, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18709, t_19202, t_22437, t_22930, t_26165, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18713, t_19206, t_22441, t_22934, t_26169, ...

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)# trace_info : t_18714, t_19207, t_22442, t_22935, t_26170, ...

        # Cross attention.
        attention_output_with_bias = self.cross_attention(                     # trace_info : t_18716, t_18721, t_19209, t_19214, t_22444, ...
            pre_cross_attn_layernorm_output,                                   # trace_info : t_18717, t_19210, t_22445, t_22938, t_26173, ...
            attention_mask=context_mask,                                       # trace_info : t_18718, t_19211, t_22446, t_22939, t_26174, ...
            key_value_states=context,                                          # trace_info : t_18719, t_19212, t_22447, t_22940, t_26175, ...
            inference_params=inference_params,                                 # trace_info : t_18720, t_19213, t_22448, t_22941, t_26176, ...
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:# trace_info : t_18723, t_19216, t_22451, t_22944, t_26179, ...
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18724, t_18730, t_19217, t_19223, t_22452, ...
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18725, t_18728, t_19218, t_19221, t_22453, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18727, t_19220, t_22455, t_22948, t_26183, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18731, t_19224, t_22459, t_22952, t_26187, ...

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)       # trace_info : t_18732, t_19225, t_22460, t_22953, t_26188, ...

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)              # trace_info : t_18750, t_19243, t_22478, t_22971, t_26206, ...

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18872, t_18880, t_19365, t_19373, t_22600, ...
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18873, t_18878, t_19366, t_19371, t_22601, ...
                mlp_output_with_bias, residual, self.hidden_dropout            # trace_info : t_18877, t_19370, t_22605, t_23098, t_26333, ...
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(                                         # trace_info : t_18881, t_18883, t_19374, t_19376, t_22609, ...
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True# trace_info : t_18882, t_19375, t_22610, t_23103, t_26338, ...
        )

        return output, context                                                 # trace_info : t_18886, t_19379, t_22614, t_23107, t_26342, ...

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
