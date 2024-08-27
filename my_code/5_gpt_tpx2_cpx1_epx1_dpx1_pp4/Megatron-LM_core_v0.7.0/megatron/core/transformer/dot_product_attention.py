# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        super().__init__(config=config)                                        # trace_info : t_10126, t_11128

        self.config: TransformerConfig = config                                # trace_info : t_10129, t_11131

        assert (
            self.config.context_parallel_size == 1                             # trace_info : t_10130, t_11132
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None                                    # trace_info : t_10131, t_11133
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)                               # trace_info : t_10132, t_11134
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_10133, t_11135
        self.attention_type = attention_type  # unused for now                 # trace_info : t_10134, t_11136

        projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_10135, t_11137

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_10136, t_11138
        self.hidden_size_per_partition = divide(projection_size, world_size)   # trace_info : t_10142, t_11144
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)# trace_info : t_10146, t_11148
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_10150, t_11152
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_10154, t_11156

        coeff = None                                                           # trace_info : t_10158, t_11160
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)      # trace_info : t_10159, t_11161
        if self.config.apply_query_key_layer_scaling:                          # trace_info : t_10160, t_11162
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(                       # trace_info : t_10161, t_10169, t_11163, t_11171
            input_in_fp16=self.config.fp16,                                    # trace_info : t_10162, t_11164
            input_in_bf16=self.config.bf16,                                    # trace_info : t_10163, t_11165
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_10164, t_11166
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,    # trace_info : t_10165, t_11167
            mask_func=attention_mask_func,                                     # trace_info : t_10166, t_11168
            softmax_in_fp32=self.config.attention_softmax_in_fp32,             # trace_info : t_10167, t_11169
            scale=coeff,                                                       # trace_info : t_10168, t_11170
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(                             # trace_info : t_10184, t_10186, t_11186, t_11188
            self.config.attention_dropout if attention_dropout is None else attention_dropout# trace_info : t_10185, t_11187
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        assert packed_seq_params is None, (                                    # trace_info : t_18534, t_19029, t_22264, t_22757, t_25992, ...
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:# trace_info : t_18535, t_19030, t_22265, t_22758, t_25993, ...
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (                                                        # trace_info : t_18540, t_19035, t_22270, t_22763, t_25998, ...
            query.size(1),                                                     # trace_info : t_18536, t_19031, t_22266, t_22759, t_25994, ...
            query.size(2),                                                     # trace_info : t_18537, t_19032, t_22267, t_22760, t_25995, ...
            query.size(0),                                                     # trace_info : t_18538, t_19033, t_22268, t_22761, t_25996, ...
            key.size(0),                                                       # trace_info : t_18539, t_19034, t_22269, t_22762, t_25997, ...
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)# trace_info : t_18541, t_19036, t_22271, t_22764, t_25999, ...
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)    # trace_info : t_18542, t_19037, t_22272, t_22765, t_26000, ...

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(# trace_info : t_18543, t_18547, t_19038, t_19042, t_22273, ...
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",# trace_info : t_18546, t_19041, t_22276, t_22769, t_26004, ...
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(                                         # trace_info : t_18554, t_18560, t_19047, t_19053, t_22282, ...
            matmul_input_buffer,                                               # trace_info : t_18555, t_19048, t_22283, t_22776, t_26011, ...
            query.transpose(0, 1),  # [b * np, sq, hn]                         # trace_info : t_18556, t_19049, t_22284, t_22777, t_26012, ...
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]           # trace_info : t_18557, t_19050, t_22285, t_22778, t_26013, ...
            beta=0.0,                                                          # trace_info : t_18558, t_19051, t_22286, t_22779, t_26014, ...
            alpha=(1.0 / self.norm_factor),                                    # trace_info : t_18559, t_19052, t_22287, t_22780, t_26015, ...
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)                    # trace_info : t_18561, t_19054, t_22289, t_22782, t_26017, ...

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)# trace_info : t_18562, t_19055, t_22290, t_22783, t_26018, ...

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:                                  # trace_info : t_18594, t_19087, t_22322, t_22815, t_26050, ...
            with tensor_parallel.get_cuda_rng_tracker().fork():                # trace_info : t_18595, t_18616, t_19088, t_19109, t_22323, ...
                attention_probs = self.attention_dropout(attention_probs)      # trace_info : t_18615, t_19108, t_22343, t_22836, t_26071, ...
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (                                                        # trace_info : t_18633, t_19126, t_22361, t_22854, t_26089, ...
            value.size(1),                                                     # trace_info : t_18629, t_19122, t_22357, t_22850, t_26085, ...
            value.size(2),                                                     # trace_info : t_18630, t_19123, t_22358, t_22851, t_26086, ...
            query.size(0),                                                     # trace_info : t_18631, t_19124, t_22359, t_22852, t_26087, ...
            value.size(3),                                                     # trace_info : t_18632, t_19125, t_22360, t_22853, t_26088, ...
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1) # trace_info : t_18634, t_19127, t_22362, t_22855, t_26090, ...

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)# trace_info : t_18635, t_19128, t_22363, t_22856, t_26091, ...

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))            # trace_info : t_18636, t_19129, t_22364, t_22857, t_26092, ...

        # change view [b, np, sq, hn]
        context = context.view(*output_size)                                   # trace_info : t_18637, t_19130, t_22365, t_22858, t_26093, ...

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()                     # trace_info : t_18638, t_19131, t_22366, t_22859, t_26094, ...

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)# trace_info : t_18639, t_19132, t_22367, t_22860, t_26095, ...
        context = context.view(*new_context_shape)                             # trace_info : t_18640, t_19133, t_22368, t_22861, t_26096, ...

        return context                                                         # trace_info : t_18641, t_19134, t_22369, t_22862, t_26097, ...
