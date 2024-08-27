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
        super().__init__(config=config)                                        # trace_info : t_9714, t_10854

        self.config: TransformerConfig = config                                # trace_info : t_9717, t_10857

        assert (
            self.config.context_parallel_size == 1                             # trace_info : t_9718, t_10858
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None                                    # trace_info : t_9719, t_10859
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)                               # trace_info : t_9720, t_10860
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_9721, t_10861
        self.attention_type = attention_type  # unused for now                 # trace_info : t_9722, t_10862

        projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_9723, t_10863

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_9724, t_10864
        self.hidden_size_per_partition = divide(projection_size, world_size)   # trace_info : t_9730, t_10870
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)# trace_info : t_9734, t_10874
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_9738, t_10878
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_9742, t_10882

        coeff = None                                                           # trace_info : t_9746, t_10886
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)      # trace_info : t_9747, t_10887
        if self.config.apply_query_key_layer_scaling:                          # trace_info : t_9748, t_10888
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(                       # trace_info : t_9749, t_9757, t_10889, t_10897
            input_in_fp16=self.config.fp16,                                    # trace_info : t_9750, t_10890
            input_in_bf16=self.config.bf16,                                    # trace_info : t_9751, t_10891
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_9752, t_10892
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,    # trace_info : t_9753, t_10893
            mask_func=attention_mask_func,                                     # trace_info : t_9754, t_10894
            softmax_in_fp32=self.config.attention_softmax_in_fp32,             # trace_info : t_9755, t_10895
            scale=coeff,                                                       # trace_info : t_9756, t_10896
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(                             # trace_info : t_9770, t_9772, t_10910, t_10912
            self.config.attention_dropout if attention_dropout is None else attention_dropout# trace_info : t_9771, t_10911
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
        assert packed_seq_params is None, (                                    # trace_info : t_18492, t_19253, t_22843, t_23598, t_27188, ...
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
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:# trace_info : t_18493, t_19254, t_22844, t_23599, t_27189, ...
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (                                                        # trace_info : t_18498, t_19259, t_22849, t_23604, t_27194, ...
            query.size(1),                                                     # trace_info : t_18494, t_19255, t_22845, t_23600, t_27190, ...
            query.size(2),                                                     # trace_info : t_18495, t_19256, t_22846, t_23601, t_27191, ...
            query.size(0),                                                     # trace_info : t_18496, t_19257, t_22847, t_23602, t_27192, ...
            key.size(0),                                                       # trace_info : t_18497, t_19258, t_22848, t_23603, t_27193, ...
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)# trace_info : t_18499, t_19260, t_22850, t_23605, t_27195, ...
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)    # trace_info : t_18500, t_19261, t_22851, t_23606, t_27196, ...

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(# trace_info : t_18501, t_18505, t_19262, t_19266, t_22852, ...
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",# trace_info : t_18504, t_19265, t_22855, t_23610, t_27200, ...
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(                                         # trace_info : t_18513, t_18519, t_19271, t_19277, t_22861, ...
            matmul_input_buffer,                                               # trace_info : t_18514, t_19272, t_22862, t_23617, t_27207, ...
            query.transpose(0, 1),  # [b * np, sq, hn]                         # trace_info : t_18515, t_19273, t_22863, t_23618, t_27208, ...
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]           # trace_info : t_18516, t_19274, t_22864, t_23619, t_27209, ...
            beta=0.0,                                                          # trace_info : t_18517, t_19275, t_22865, t_23620, t_27210, ...
            alpha=(1.0 / self.norm_factor),                                    # trace_info : t_18518, t_19276, t_22866, t_23621, t_27211, ...
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)                    # trace_info : t_18520, t_19278, t_22868, t_23623, t_27213, ...

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)# trace_info : t_18521, t_19279, t_22869, t_23624, t_27214, ...

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:                                  # trace_info : t_18541, t_19299, t_22889, t_23644, t_27234, ...
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)          # trace_info : t_18542, t_19300, t_22890, t_23645, t_27235, ...

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (                                                        # trace_info : t_18547, t_19305, t_22895, t_23650, t_27240, ...
            value.size(1),                                                     # trace_info : t_18543, t_19301, t_22891, t_23646, t_27236, ...
            value.size(2),                                                     # trace_info : t_18544, t_19302, t_22892, t_23647, t_27237, ...
            query.size(0),                                                     # trace_info : t_18545, t_19303, t_22893, t_23648, t_27238, ...
            value.size(3),                                                     # trace_info : t_18546, t_19304, t_22894, t_23649, t_27239, ...
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1) # trace_info : t_18548, t_19306, t_22896, t_23651, t_27241, ...

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)# trace_info : t_18549, t_19307, t_22897, t_23652, t_27242, ...

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))            # trace_info : t_18550, t_19308, t_22898, t_23653, t_27243, ...

        # change view [b, np, sq, hn]
        context = context.view(*output_size)                                   # trace_info : t_18551, t_19309, t_22899, t_23654, t_27244, ...

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()                     # trace_info : t_18552, t_19310, t_22900, t_23655, t_27245, ...

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)# trace_info : t_18553, t_19311, t_22901, t_23656, t_27246, ...
        context = context.view(*new_context_shape)                             # trace_info : t_18554, t_19312, t_22902, t_23657, t_27247, ...

        return context                                                         # trace_info : t_18555, t_19313, t_22903, t_23658, t_27248, ...
