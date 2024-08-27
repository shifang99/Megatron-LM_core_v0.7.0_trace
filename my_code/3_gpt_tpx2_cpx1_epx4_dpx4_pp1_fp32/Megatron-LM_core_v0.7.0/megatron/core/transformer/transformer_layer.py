# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC                                                            # trace_info : t_9330
from dataclasses import dataclass, field                                       # trace_info : t_9331
from typing import Dict, Optional, Union                                       # trace_info : t_9332
                                                                               # trace_info : t_9333
import torch                                                                   # trace_info : t_9334
                                                                               # trace_info : t_9335
from megatron.core import parallel_state                                       # trace_info : t_9336
from megatron.core.dist_checkpointing.mapping import ShardedStateDict          # trace_info : t_9337
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping        # trace_info : t_9338
from megatron.core.transformer.enums import AttnMaskType                       # trace_info : t_9339
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
        super().__init__(config=config)                                        # trace_info : t_9568, t_10708
        self.submodules_config = submodules                                    # trace_info : t_9571, t_10711

        self.layer_number = layer_number + self._get_layer_offset()            # trace_info : t_9572, t_10712
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout# trace_info : t_9593, t_10733

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(                                   # trace_info : t_9594, t_9599, t_10734, t_10739
            submodules.input_layernorm,                                        # trace_info : t_9595, t_10735
            config=self.config,                                                # trace_info : t_9596, t_10736
            hidden_size=self.config.hidden_size,                               # trace_info : t_9597, t_10737
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_9598, t_10738
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(                                    # trace_info : t_9636, t_9638, t_10776, t_10778
            submodules.self_attention, config=self.config, layer_number=layer_number,# trace_info : t_9637, t_10777
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)            # trace_info : t_10117, t_11257

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(                          # trace_info : t_10120, t_10125, t_11260, t_11265
            submodules.pre_cross_attn_layernorm,                               # trace_info : t_10121, t_11261
            config=self.config,                                                # trace_info : t_10122, t_11262
            hidden_size=self.config.hidden_size,                               # trace_info : t_10123, t_11263
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10124, t_11264
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(                                   # trace_info : t_10141, t_10143, t_11281, t_11283
            submodules.cross_attention, config=self.config, layer_number=layer_number,# trace_info : t_10142, t_11282
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config,)# trace_info : t_10159, t_11299

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(                                 # trace_info : t_10176, t_10181, t_11316, t_11321
            submodules.pre_mlp_layernorm,                                      # trace_info : t_10177, t_11317
            config=self.config,                                                # trace_info : t_10178, t_11318
            hidden_size=self.config.hidden_size,                               # trace_info : t_10179, t_11319
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10180, t_11320
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)            # trace_info : t_10218, t_11358
        if hasattr(self.mlp, 'set_layer_number'):                              # trace_info : t_10680, t_11820
            self.mlp.set_layer_number(self.layer_number)                       # trace_info : t_10681, t_11821

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)                        # trace_info : t_10685, t_11825

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad                 # trace_info : t_10688, t_11828

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()      # trace_info : t_9573, t_10713

        num_layers_per_pipeline_rank = (                                       # trace_info : t_9583, t_10723
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_9578, t_10718
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:# trace_info : t_9584, t_10724
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:    # trace_info : t_9586, t_10726
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0                                                     # trace_info : t_9591, t_10731

        return offset                                                          # trace_info : t_9592, t_10732

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
        residual = hidden_states                                               # trace_info : t_18343, t_19106, t_22696, t_23451, t_27041, ...

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)           # trace_info : t_18344, t_19107, t_22697, t_23452, t_27042, ...

        # Self attention.
        attention_output_with_bias = self.self_attention(                      # trace_info : t_18362, t_18368, t_19125, t_19131, t_22715, ...
            input_layernorm_output,                                            # trace_info : t_18363, t_19126, t_22716, t_23471, t_27061, ...
            attention_mask=attention_mask,                                     # trace_info : t_18364, t_19127, t_22717, t_23472, t_27062, ...
            inference_params=inference_params,                                 # trace_info : t_18365, t_19128, t_22718, t_23473, t_27063, ...
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_18366, t_19129, t_22719, t_23474, t_27064, ...
            packed_seq_params=packed_seq_params,                               # trace_info : t_18367, t_19130, t_22720, t_23475, t_27065, ...
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18625, t_18633, t_19383, t_19391, t_22973, ...
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18626, t_18631, t_19384, t_19389, t_22974, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18630, t_19388, t_22978, t_23733, t_27323, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18634, t_19392, t_22982, t_23737, t_27327, ...

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)# trace_info : t_18635, t_19393, t_22983, t_23738, t_27328, ...

        # Cross attention.
        attention_output_with_bias = self.cross_attention(                     # trace_info : t_18637, t_18642, t_19395, t_19400, t_22985, ...
            pre_cross_attn_layernorm_output,                                   # trace_info : t_18638, t_19396, t_22986, t_23741, t_27331, ...
            attention_mask=context_mask,                                       # trace_info : t_18639, t_19397, t_22987, t_23742, t_27332, ...
            key_value_states=context,                                          # trace_info : t_18640, t_19398, t_22988, t_23743, t_27333, ...
            inference_params=inference_params,                                 # trace_info : t_18641, t_19399, t_22989, t_23744, t_27334, ...
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:# trace_info : t_18644, t_19402, t_22992, t_23747, t_27337, ...
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18645, t_18651, t_19403, t_19409, t_22993, ...
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18646, t_18649, t_19404, t_19407, t_22994, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18648, t_19406, t_22996, t_23751, t_27341, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18652, t_19410, t_23000, t_23755, t_27345, ...

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)       # trace_info : t_18653, t_19411, t_23001, t_23756, t_27346, ...

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)              # trace_info : t_18671, t_19429, t_23019, t_23774, t_27364, ...

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_19073, t_19081, t_19828, t_19836, t_23418, ...
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_19074, t_19079, t_19829, t_19834, t_23419, ...
                mlp_output_with_bias, residual, self.hidden_dropout            # trace_info : t_19078, t_19833, t_23423, t_24178, t_27768, ...
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(                                         # trace_info : t_19082, t_19084, t_19837, t_19839, t_23427, ...
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True# trace_info : t_19083, t_19838, t_23428, t_24183, t_27773, ...
        )

        return output, context                                                 # trace_info : t_19087, t_19842, t_23432, t_24187, t_27777, ...

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
