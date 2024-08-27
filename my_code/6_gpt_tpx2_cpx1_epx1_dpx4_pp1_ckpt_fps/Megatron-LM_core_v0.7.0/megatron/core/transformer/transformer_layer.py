# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC                                                            # trace_info : t_9405
from dataclasses import dataclass, field                                       # trace_info : t_9406
from typing import Dict, Optional, Union                                       # trace_info : t_9407
                                                                               # trace_info : t_9408
import torch                                                                   # trace_info : t_9409
                                                                               # trace_info : t_9410
from megatron.core import parallel_state                                       # trace_info : t_9411
from megatron.core.dist_checkpointing.mapping import ShardedStateDict          # trace_info : t_9412
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping        # trace_info : t_9413
from megatron.core.transformer.enums import AttnMaskType                       # trace_info : t_9414
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
        super().__init__(config=config)                                        # trace_info : t_9643, t_10645
        self.submodules_config = submodules                                    # trace_info : t_9646, t_10648

        self.layer_number = layer_number + self._get_layer_offset()            # trace_info : t_9647, t_10649
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout# trace_info : t_9668, t_10670

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(                                   # trace_info : t_9669, t_9674, t_10671, t_10676
            submodules.input_layernorm,                                        # trace_info : t_9670, t_10672
            config=self.config,                                                # trace_info : t_9671, t_10673
            hidden_size=self.config.hidden_size,                               # trace_info : t_9672, t_10674
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_9673, t_10675
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(                                    # trace_info : t_9711, t_9713, t_10713, t_10715
            submodules.self_attention, config=self.config, layer_number=layer_number,# trace_info : t_9712, t_10714
        )

        ## [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)            # trace_info : t_10194, t_11196

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(                          # trace_info : t_10197, t_10202, t_11199, t_11204
            submodules.pre_cross_attn_layernorm,                               # trace_info : t_10198, t_11200
            config=self.config,                                                # trace_info : t_10199, t_11201
            hidden_size=self.config.hidden_size,                               # trace_info : t_10200, t_11202
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10201, t_11203
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(                                   # trace_info : t_10218, t_10220, t_11220, t_11222
            submodules.cross_attention, config=self.config, layer_number=layer_number,# trace_info : t_10219, t_11221
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config,)# trace_info : t_10236, t_11238

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(                                 # trace_info : t_10253, t_10258, t_11255, t_11260
            submodules.pre_mlp_layernorm,                                      # trace_info : t_10254, t_11256
            config=self.config,                                                # trace_info : t_10255, t_11257
            hidden_size=self.config.hidden_size,                               # trace_info : t_10256, t_11258
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10257, t_11259
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)            # trace_info : t_10295, t_11297
        if hasattr(self.mlp, 'set_layer_number'):                              # trace_info : t_10621, t_11623
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)                        # trace_info : t_10622, t_11624

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad                 # trace_info : t_10625, t_11627

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()      # trace_info : t_9648, t_10650, t_25287, t_27081, t_92880, ...

        num_layers_per_pipeline_rank = (                                       # trace_info : t_9658, t_10660, t_25297, t_27091, t_92890, ...
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_9653, t_10655, t_25292, t_27086, t_92885, ...
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:# trace_info : t_9659, t_10661, t_25298, t_27092, t_92891, ...
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:    # trace_info : t_9661, t_10663, t_25300, t_27094, t_92893, ...
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0                                                     # trace_info : t_9666, t_10668, t_25305, t_27099, t_92898, ...

        return offset                                                          # trace_info : t_9667, t_10669, t_25306, t_27100, t_92899, ...

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
        residual = hidden_states                                               # trace_info : t_18423, t_18918, t_22062, t_22555, t_89669, ...

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)           # trace_info : t_18424, t_18919, t_22063, t_22556, t_89670, ...

        # Self attention.
        attention_output_with_bias = self.self_attention(                      # trace_info : t_18442, t_18448, t_18937, t_18943, t_22081, ...
            input_layernorm_output,                                            # trace_info : t_18443, t_18938, t_22082, t_22575, t_89689, ...
            attention_mask=attention_mask,                                     # trace_info : t_18444, t_18939, t_22083, t_22576, t_89690, ...
            inference_params=inference_params,                                 # trace_info : t_18445, t_18940, t_22084, t_22577, t_89691, ...
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_18446, t_18941, t_22085, t_22578, t_89692, ...
            packed_seq_params=packed_seq_params,                               # trace_info : t_18447, t_18942, t_22086, t_22579, t_89693, ...
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18717, t_18725, t_19210, t_19218, t_22354, ...
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18718, t_18723, t_19211, t_19216, t_22355, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18722, t_19215, t_22359, t_22852, t_89966, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18726, t_19219, t_22363, t_22856, t_89970, ...

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)# trace_info : t_18727, t_19220, t_22364, t_22857, t_89971, ...

        # Cross attention.
        attention_output_with_bias = self.cross_attention(                     # trace_info : t_18729, t_18734, t_19222, t_19227, t_22366, ...
            pre_cross_attn_layernorm_output,                                   # trace_info : t_18730, t_19223, t_22367, t_22860, t_89974, ...
            attention_mask=context_mask,                                       # trace_info : t_18731, t_19224, t_22368, t_22861, t_89975, ...
            key_value_states=context,                                          # trace_info : t_18732, t_19225, t_22369, t_22862, t_89976, ...
            inference_params=inference_params,                                 # trace_info : t_18733, t_19226, t_22370, t_22863, t_89977, ...
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:# trace_info : t_18736, t_19229, t_22373, t_22866, t_89980, ...
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18737, t_18743, t_19230, t_19236, t_22374, ...
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18738, t_18741, t_19231, t_19234, t_22375, ...
                attention_output_with_bias, residual, self.hidden_dropout      # trace_info : t_18740, t_19233, t_22377, t_22870, t_89984, ...
            )

        # Residual connection.
        residual = hidden_states                                               # trace_info : t_18744, t_19237, t_22381, t_22874, t_89988, ...

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)       # trace_info : t_18745, t_19238, t_22382, t_22875, t_89989, ...

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)              # trace_info : t_18763, t_19256, t_22400, t_22893, t_90007, ...

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():                             # trace_info : t_18885, t_18893, t_19378, t_19386, t_22522, ...
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(# trace_info : t_18886, t_18891, t_19379, t_19384, t_22523, ...
                mlp_output_with_bias, residual, self.hidden_dropout            # trace_info : t_18890, t_19383, t_22527, t_23020, t_90134, ...
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(                                         # trace_info : t_18894, t_18896, t_19387, t_19389, t_22531, ...
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True# trace_info : t_18895, t_19388, t_22532, t_23025, t_90139, ...
        )

        return output, context                                                 # trace_info : t_18899, t_19392, t_22536, t_23029, t_90143, ...

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)# trace_info : t_25316, t_27110, t_92909, t_94703
        prefixed_map = {                                                       # trace_info : t_26721, t_26723, t_28515, t_28517, t_94314, ...
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()# trace_info : t_26722, t_28516, t_94315, t_96109
        }
        if prefixed_map:                                                       # trace_info : t_26724, t_28518, t_94317, t_96111
            apply_prefix_mapping(sharded_state_dict, prefixed_map)             # trace_info : t_26725, t_28519, t_94318, t_96112
        return sharded_state_dict                                              # trace_info : t_26927, t_28721, t_94520, t_96314
