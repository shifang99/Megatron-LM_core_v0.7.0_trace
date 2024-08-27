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
    mlp = _get_mlp_module_spec(                                                # trace_info : t_9638, t_9640
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm# trace_info : t_9639
    )
    return ModuleSpec(                                                         # trace_info : t_9655, t_9692
        module=TransformerLayer,                                               # trace_info : t_9656
        submodules=TransformerLayerSubmodules(                                 # trace_info : t_9657, t_9681
            self_attention=ModuleSpec(                                         # trace_info : t_9658, t_9673
                module=SelfAttention,                                          # trace_info : t_9659
                params={"attn_mask_type": AttnMaskType.causal},                # trace_info : t_9660
                submodules=SelfAttentionSubmodules(                            # trace_info : t_9661, t_9667
                    linear_qkv=TELayerNormColumnParallelLinear,                # trace_info : t_9662
                    core_attention=TEDotProductAttention,                      # trace_info : t_9663
                    linear_proj=TERowParallelLinear,                           # trace_info : t_9664
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,        # trace_info : t_9665
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,        # trace_info : t_9666
                ),
            ),
            self_attn_bda=get_bias_dropout_add,                                # trace_info : t_9677
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,           # trace_info : t_9678
            mlp=mlp,                                                           # trace_info : t_9679
            mlp_bda=get_bias_dropout_add,                                      # trace_info : t_9680
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    if num_experts is None:                                                    # trace_info : t_9641
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(                                                     # trace_info : t_9642, t_9650
            module=MLP,                                                        # trace_info : t_9643
            submodules=MLPSubmodules(                                          # trace_info : t_9644, t_9647
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,# trace_info : t_9645
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,# trace_info : t_9646
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
