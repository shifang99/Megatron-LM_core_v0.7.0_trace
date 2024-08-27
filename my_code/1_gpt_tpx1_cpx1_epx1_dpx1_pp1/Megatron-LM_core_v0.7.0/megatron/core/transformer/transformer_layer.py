# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC                                                            # trace_info : t_6376
from dataclasses import dataclass, field                                       # trace_info : t_6377
from typing import Dict, Optional, Union                                       # trace_info : t_6378
                                                                               # trace_info : t_6379
import torch                                                                   # trace_info : t_6380
                                                                               # trace_info : t_6381
from megatron.core import parallel_state                                       # trace_info : t_6382
from megatron.core.dist_checkpointing.mapping import ShardedStateDict          # trace_info : t_6383
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping        # trace_info : t_6384
from megatron.core.transformer.enums import AttnMaskType                       # trace_info : t_6385
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
        super().__init__(config=config)                                        # trace_info : t_6614, t_7616
        self.submodules_config = submodules                                    # trace_info : t_6617, t_7619

        self.layer_number = layer_number + self._get_layer_offset()            # trace_info : t_6618, t_7620
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout# trace_info : t_6639, t_7641

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(                                   # trace_info : t_6640, t_6645, t_7642, t_7647
            submodules.input_layernorm,                                        # trace_info : t_6641, t_7643
            config=self.config,                                                # trace_info : t_6642, t_7644
            hidden_size=self.config.hidden_size,                               # trace_info : t_6643, t_7645
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_6644, t_7646
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(                                    # trace_info : t_6682, t_6684, t_7684, t_7686
            submodules.self_attention, config=self.config, layer_number=layer_number,# trace_info : t_6683, t_7685
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)            # trace_info : t_7165, t_8167

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(                          # trace_info : t_7168, t_7173, t_8170, t_8175
            submodules.pre_cross_attn_layernorm,                               # trace_info : t_7169, t_8171
            config=self.config,                                                # trace_info : t_7170, t_8172
            hidden_size=self.config.hidden_size,                               # trace_info : t_7171, t_8173
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_7172, t_8174
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(                                   # trace_info : t_7189, t_7191, t_8191, t_8193
            submodules.cross_attention, config=self.config, layer_number=layer_number,# trace_info : t_7190, t_8192
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config,)# trace_info : t_7207, t_8209

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(                                 # trace_info : t_7224, t_7229, t_8226, t_8231
            submodules.pre_mlp_layernorm,                                      # trace_info : t_7225, t_8227
            config=self.config,                                                # trace_info : t_7226, t_8228
            hidden_size=self.config.hidden_size,                               # trace_info : t_7227, t_8229
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_7228, t_8230
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)            # trace_info : t_7266, t_8268
        if hasattr(self.mlp, 'set_layer_number'):                              # trace_info : t_7592, t_8594
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)                        # trace_info : t_7593, t_8595

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad                 # trace_info : t_7596, t_8598

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()      # trace_info : t_6619, t_7621

        num_layers_per_pipeline_rank = (                                       # trace_info : t_6629, t_7631
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_6624, t_7626
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:# trace_info : t_6630, t_7632
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:    # trace_info : t_6632, t_7634
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0                                                     # trace_info : t_6637, t_7639

        return offset                                                          # trace_info : t_6638, t_7640

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
        residual = hidden_states                                               # trace_info : t_15271, t_15774, t_18912, t_19413, t_22551, ...

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)           # trace_info : t_15272, t_15775, t_18913, t_19414, t_22552, ...

        # Self attention.
        attention_output_with_bias = self.self_attention(                      # trace_info : t_15290, t_15296, t_15793, t_15799, t_18931, ...
            input_layernorm_output,                                            # trace_info : t_15291, t_15794, t_18932, t_19433, t_22571, ...
            attention_mask=attention_mask,                                     # trace_info : t_15292, t_15795, t_18933, t_19434, t_22572, ...
            inference_params=inference_params,                                 # trace_info : t_15293, t_15796, t_18934, t_19435, t_22573, ...
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_15294, t_15797, t_18935, t_19436, t_22574, ...
            packed_seq_params=packed_seq_params,                               # trace_info : t_15295, t_15798, t_18936, t_19437, t_22575, ...
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_15569, t_15577, t_16070, t_16078, t_19208, ...
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_15570, t_15575, t_16071, t_16076, t_19209, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_15574, t_16075, t_19213, t_19714, t_22852, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_15578, t_16079, t_19217, t_19718, t_22856, ...

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)# trace_info : t_15579, t_16080, t_19218, t_19719, t_22857, ...

        # Cross attention.
        attention_output_with_bias = self.cross_attention(                     # trace_info : t_15581, t_15586, t_16082, t_16087, t_19220, ...
            pre_cross_attn_layernorm_output,                                   # trace_info : t_15582, t_16083, t_19221, t_19722, t_22860, ...
            attention_mask=context_mask,                                       # trace_info : t_15583, t_16084, t_19222, t_19723, t_22861, ...
            key_value_states=context,                                          # trace_info : t_15584, t_16085, t_19223, t_19724, t_22862, ...
            inference_params=inference_params,                                 # trace_info : t_15585, t_16086, t_19224, t_19725, t_22863, ...
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:# trace_info : t_15588, t_16089, t_19227, t_19728, t_22866, ...
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_15589, t_15595, t_16090, t_16096, t_19228, ...
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_15590, t_15593, t_16091, t_16094, t_19229, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_15592, t_16093, t_19231, t_19732, t_22870, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_15596, t_16097, t_19235, t_19736, t_22874, ...

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)       # trace_info : t_15597, t_16098, t_19236, t_19737, t_22875, ...

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)              # trace_info : t_15615, t_16116, t_19254, t_19755, t_22893, ...

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_15741, t_15749, t_16242, t_16250, t_19380, ...
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_15742, t_15747, t_16243, t_16248, t_19381, ...
                mlp_output_with_bias, residual, self.hidden_dropout            # trace_info : t_15746, t_16247, t_19385, t_19886, t_23024, ...
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(                                         # trace_info : t_15750, t_15752, t_16251, t_16253, t_19389, ...
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True# trace_info : t_15751, t_16252, t_19390, t_19891, t_23029, ...
        )

        return output, context                                                 # trace_info : t_15755, t_16256, t_19394, t_19895, t_23033, ...

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
