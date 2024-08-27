# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC                                                            # trace_info : t_9682
from dataclasses import dataclass, field                                       # trace_info : t_9683
from typing import Dict, Optional, Union                                       # trace_info : t_9684
                                                                               # trace_info : t_9685
import torch                                                                   # trace_info : t_9686
                                                                               # trace_info : t_9687
from megatron.core import parallel_state                                       # trace_info : t_9688
from megatron.core.dist_checkpointing.mapping import ShardedStateDict          # trace_info : t_9689
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping        # trace_info : t_9690
from megatron.core.transformer.enums import AttnMaskType                       # trace_info : t_9691
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
        super().__init__(config=config)                                        # trace_info : t_9920, t_10699
        self.submodules_config = submodules                                    # trace_info : t_9923, t_10702

        self.layer_number = layer_number + self._get_layer_offset()            # trace_info : t_9924, t_10703
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout# trace_info : t_9945, t_10724

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(                                   # trace_info : t_9946, t_9951, t_10725, t_10730
            submodules.input_layernorm,                                        # trace_info : t_9947, t_10726
            config=self.config,                                                # trace_info : t_9948, t_10727
            hidden_size=self.config.hidden_size,                               # trace_info : t_9949, t_10728
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_9950, t_10729
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(                                    # trace_info : t_9967, t_9969, t_10746, t_10748
            submodules.self_attention, config=self.config, layer_number=layer_number,# trace_info : t_9968, t_10747
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)            # trace_info : t_10355, t_11133

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(                          # trace_info : t_10358, t_10363, t_11136, t_11141
            submodules.pre_cross_attn_layernorm,                               # trace_info : t_10359, t_11137
            config=self.config,                                                # trace_info : t_10360, t_11138
            hidden_size=self.config.hidden_size,                               # trace_info : t_10361, t_11139
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10362, t_11140
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(                                   # trace_info : t_10379, t_10381, t_11157, t_11159
            submodules.cross_attention, config=self.config, layer_number=layer_number,# trace_info : t_10380, t_11158
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config,)# trace_info : t_10397, t_11175

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(                                 # trace_info : t_10414, t_10419, t_11192, t_11197
            submodules.pre_mlp_layernorm,                                      # trace_info : t_10415, t_11193
            config=self.config,                                                # trace_info : t_10416, t_11194
            hidden_size=self.config.hidden_size,                               # trace_info : t_10417, t_11195
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10418, t_11196
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)            # trace_info : t_10435, t_11213
        if hasattr(self.mlp, 'set_layer_number'):                              # trace_info : t_10675, t_11453
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)                        # trace_info : t_10676, t_11454

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad                 # trace_info : t_10679, t_11457

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()      # trace_info : t_9925, t_10704

        num_layers_per_pipeline_rank = (                                       # trace_info : t_9935, t_10714
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_9930, t_10709
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:# trace_info : t_9936, t_10715
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:    # trace_info : t_9938, t_10717
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0                                                     # trace_info : t_9943, t_10722

        return offset                                                          # trace_info : t_9944, t_10723

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
        residual = hidden_states                                               # trace_info : t_18263, t_18470, t_21449, t_21656, t_24635, ...

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)           # trace_info : t_18264, t_18471, t_21450, t_21657, t_24636, ...

        # Self attention.
        attention_output_with_bias = self.self_attention(                      # trace_info : t_18266, t_18272, t_18473, t_18479, t_21452, ...
            input_layernorm_output,                                            # trace_info : t_18267, t_18474, t_21453, t_21660, t_24639, ...
            attention_mask=attention_mask,                                     # trace_info : t_18268, t_18475, t_21454, t_21661, t_24640, ...
            inference_params=inference_params,                                 # trace_info : t_18269, t_18476, t_21455, t_21662, t_24641, ...
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_18270, t_18477, t_21456, t_21663, t_24642, ...
            packed_seq_params=packed_seq_params,                               # trace_info : t_18271, t_18478, t_21457, t_21664, t_24643, ...
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18383, t_18391, t_18590, t_18598, t_21569, ...
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18384, t_18389, t_18591, t_18596, t_21570, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18388, t_18595, t_21574, t_21781, t_24760, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18392, t_18599, t_21578, t_21785, t_24764, ...

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)# trace_info : t_18393, t_18600, t_21579, t_21786, t_24765, ...

        # Cross attention.
        attention_output_with_bias = self.cross_attention(                     # trace_info : t_18395, t_18400, t_18602, t_18607, t_21581, ...
            pre_cross_attn_layernorm_output,                                   # trace_info : t_18396, t_18603, t_21582, t_21789, t_24768, ...
            attention_mask=context_mask,                                       # trace_info : t_18397, t_18604, t_21583, t_21790, t_24769, ...
            key_value_states=context,                                          # trace_info : t_18398, t_18605, t_21584, t_21791, t_24770, ...
            inference_params=inference_params,                                 # trace_info : t_18399, t_18606, t_21585, t_21792, t_24771, ...
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:# trace_info : t_18402, t_18609, t_21588, t_21795, t_24774, ...
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18403, t_18409, t_18610, t_18616, t_21589, ...
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18404, t_18407, t_18611, t_18614, t_21590, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18406, t_18613, t_21592, t_21799, t_24778, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18410, t_18617, t_21596, t_21803, t_24782, ...

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)       # trace_info : t_18411, t_18618, t_21597, t_21804, t_24783, ...

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)              # trace_info : t_18413, t_18620, t_21599, t_21806, t_24785, ...

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18437, t_18445, t_18644, t_18652, t_21623, ...
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18438, t_18443, t_18645, t_18650, t_21624, ...
                mlp_output_with_bias, residual, self.hidden_dropout            # trace_info : t_18442, t_18649, t_21628, t_21835, t_24814, ...
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(                                         # trace_info : t_18446, t_18448, t_18653, t_18655, t_21632, ...
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True# trace_info : t_18447, t_18654, t_21633, t_21840, t_24819, ...
        )

        return output, context                                                 # trace_info : t_18451, t_18658, t_21637, t_21844, t_24823, ...

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
