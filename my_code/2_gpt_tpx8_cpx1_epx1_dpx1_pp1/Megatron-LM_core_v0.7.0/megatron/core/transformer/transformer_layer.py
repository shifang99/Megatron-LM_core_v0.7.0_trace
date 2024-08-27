# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC                                                            # trace_info : t_11243
from dataclasses import dataclass, field                                       # trace_info : t_11244
from typing import Dict, Optional, Union                                       # trace_info : t_11245
                                                                               # trace_info : t_11246
import torch                                                                   # trace_info : t_11247
                                                                               # trace_info : t_11248
from megatron.core import parallel_state                                       # trace_info : t_11249
from megatron.core.dist_checkpointing.mapping import ShardedStateDict          # trace_info : t_11250
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping        # trace_info : t_11251
from megatron.core.transformer.enums import AttnMaskType                       # trace_info : t_11252
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
        super().__init__(config=config)                                        # trace_info : t_11481, t_12483
        self.submodules_config = submodules                                    # trace_info : t_11484, t_12486

        self.layer_number = layer_number + self._get_layer_offset()            # trace_info : t_11485, t_12487
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout# trace_info : t_11506, t_12508

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(                                   # trace_info : t_11507, t_11512, t_12509, t_12514
            submodules.input_layernorm,                                        # trace_info : t_11508, t_12510
            config=self.config,                                                # trace_info : t_11509, t_12511
            hidden_size=self.config.hidden_size,                               # trace_info : t_11510, t_12512
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_11511, t_12513
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(                                    # trace_info : t_11549, t_11551, t_12551, t_12553
            submodules.self_attention, config=self.config, layer_number=layer_number,# trace_info : t_11550, t_12552
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)            # trace_info : t_12032, t_13034

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(                          # trace_info : t_12035, t_12040, t_13037, t_13042
            submodules.pre_cross_attn_layernorm,                               # trace_info : t_12036, t_13038
            config=self.config,                                                # trace_info : t_12037, t_13039
            hidden_size=self.config.hidden_size,                               # trace_info : t_12038, t_13040
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_12039, t_13041
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(                                   # trace_info : t_12056, t_12058, t_13058, t_13060
            submodules.cross_attention, config=self.config, layer_number=layer_number,# trace_info : t_12057, t_13059
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config,)# trace_info : t_12074, t_13076

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(                                 # trace_info : t_12091, t_12096, t_13093, t_13098
            submodules.pre_mlp_layernorm,                                      # trace_info : t_12092, t_13094
            config=self.config,                                                # trace_info : t_12093, t_13095
            hidden_size=self.config.hidden_size,                               # trace_info : t_12094, t_13096
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_12095, t_13097
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)            # trace_info : t_12133, t_13135
        if hasattr(self.mlp, 'set_layer_number'):                              # trace_info : t_12459, t_13461
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)                        # trace_info : t_12460, t_13462

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad                 # trace_info : t_12463, t_13465

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()      # trace_info : t_11486, t_12488

        num_layers_per_pipeline_rank = (                                       # trace_info : t_11496, t_12498
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_11491, t_12493
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:# trace_info : t_11497, t_12499
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:    # trace_info : t_11499, t_12501
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0                                                     # trace_info : t_11504, t_12506

        return offset                                                          # trace_info : t_11505, t_12507

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
        residual = hidden_states                                               # trace_info : t_20144, t_20631, t_23756, t_24241, t_27366, ...

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)           # trace_info : t_20145, t_20632, t_23757, t_24242, t_27367, ...

        # Self attention.
        attention_output_with_bias = self.self_attention(                      # trace_info : t_20163, t_20169, t_20650, t_20656, t_23775, ...
            input_layernorm_output,                                            # trace_info : t_20164, t_20651, t_23776, t_24261, t_27386, ...
            attention_mask=attention_mask,                                     # trace_info : t_20165, t_20652, t_23777, t_24262, t_27387, ...
            inference_params=inference_params,                                 # trace_info : t_20166, t_20653, t_23778, t_24263, t_27388, ...
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_20167, t_20654, t_23779, t_24264, t_27389, ...
            packed_seq_params=packed_seq_params,                               # trace_info : t_20168, t_20655, t_23780, t_24265, t_27390, ...
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_20430, t_20438, t_20915, t_20923, t_24040, ...
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_20431, t_20436, t_20916, t_20921, t_24041, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_20435, t_20920, t_24045, t_24530, t_27655, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_20439, t_20924, t_24049, t_24534, t_27659, ...

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)# trace_info : t_20440, t_20925, t_24050, t_24535, t_27660, ...

        # Cross attention.
        attention_output_with_bias = self.cross_attention(                     # trace_info : t_20442, t_20447, t_20927, t_20932, t_24052, ...
            pre_cross_attn_layernorm_output,                                   # trace_info : t_20443, t_20928, t_24053, t_24538, t_27663, ...
            attention_mask=context_mask,                                       # trace_info : t_20444, t_20929, t_24054, t_24539, t_27664, ...
            key_value_states=context,                                          # trace_info : t_20445, t_20930, t_24055, t_24540, t_27665, ...
            inference_params=inference_params,                                 # trace_info : t_20446, t_20931, t_24056, t_24541, t_27666, ...
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:# trace_info : t_20449, t_20934, t_24059, t_24544, t_27669, ...
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_20450, t_20456, t_20935, t_20941, t_24060, ...
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_20451, t_20454, t_20936, t_20939, t_24061, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_20453, t_20938, t_24063, t_24548, t_27673, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_20457, t_20942, t_24067, t_24552, t_27677, ...

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)       # trace_info : t_20458, t_20943, t_24068, t_24553, t_27678, ...

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)              # trace_info : t_20476, t_20961, t_24086, t_24571, t_27696, ...

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_20598, t_20606, t_21083, t_21091, t_24208, ...
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_20599, t_20604, t_21084, t_21089, t_24209, ...
                mlp_output_with_bias, residual, self.hidden_dropout            # trace_info : t_20603, t_21088, t_24213, t_24698, t_27823, ...
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(                                         # trace_info : t_20607, t_20609, t_21092, t_21094, t_24217, ...
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True# trace_info : t_20608, t_21093, t_24218, t_24703, t_27828, ...
        )

        return output, context                                                 # trace_info : t_20612, t_21097, t_24222, t_24707, t_27832, ...

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
