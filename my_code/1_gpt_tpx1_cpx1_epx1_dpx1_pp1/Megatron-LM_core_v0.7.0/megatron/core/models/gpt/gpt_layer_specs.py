# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(                                                # trace_info : t_6328, t_6330
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm# trace_info : t_6329
    )
    return ModuleSpec(                                                         # trace_info : t_6345, t_6386
        module=TransformerLayer,                                               # trace_info : t_6346
        submodules=TransformerLayerSubmodules(                                 # trace_info : t_6347, t_6375
            input_layernorm=FusedLayerNorm,                                    # trace_info : t_6348
            self_attention=ModuleSpec(                                         # trace_info : t_6349, t_6364
                module=SelfAttention,                                          # trace_info : t_6350
                params={"attn_mask_type": AttnMaskType.causal},                # trace_info : t_6351
                submodules=SelfAttentionSubmodules(                            # trace_info : t_6352, t_6358
                    linear_qkv=ColumnParallelLinear,                           # trace_info : t_6353
                    core_attention=DotProductAttention,                        # trace_info : t_6354
                    linear_proj=RowParallelLinear,                             # trace_info : t_6355
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,# trace_info : t_6356
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,# trace_info : t_6357
                ),
            ),
            self_attn_bda=get_bias_dropout_add,                                # trace_info : t_6368
            pre_mlp_layernorm=FusedLayerNorm,                                  # trace_info : t_6369
            mlp=mlp,                                                           # trace_info : t_6370
            mlp_bda=get_bias_dropout_add,                                      # trace_info : t_6371
            sharded_state_dict_keys_map={                                      # trace_info : t_6374
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',   # trace_info : t_6372
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',            # trace_info : t_6373
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    if num_experts is None:                                                    # trace_info : t_6331
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(                                                     # trace_info : t_6332, t_6340
            module=MLP,                                                        # trace_info : t_6333
            submodules=MLPSubmodules(                                          # trace_info : t_6334, t_6337
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,# trace_info : t_6335
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,# trace_info : t_6336
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )
