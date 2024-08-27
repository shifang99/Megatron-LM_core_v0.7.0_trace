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
        super().__init__(config=config)                                        # trace_info : t_6760, t_7762

        self.config: TransformerConfig = config                                # trace_info : t_6763, t_7765

        assert (
            self.config.context_parallel_size == 1                             # trace_info : t_6764, t_7766
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None                                    # trace_info : t_6765, t_7767
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)                               # trace_info : t_6766, t_7768
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_6767, t_7769
        self.attention_type = attention_type  # unused for now                 # trace_info : t_6768, t_7770

        projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_6769, t_7771

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_6770, t_7772
        self.hidden_size_per_partition = divide(projection_size, world_size)   # trace_info : t_6776, t_7778
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)# trace_info : t_6780, t_7782
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_6784, t_7786
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_6788, t_7790

        coeff = None                                                           # trace_info : t_6792, t_7794
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)      # trace_info : t_6793, t_7795
        if self.config.apply_query_key_layer_scaling:                          # trace_info : t_6794, t_7796
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(                       # trace_info : t_6795, t_6803, t_7797, t_7805
            input_in_fp16=self.config.fp16,                                    # trace_info : t_6796, t_7798
            input_in_bf16=self.config.bf16,                                    # trace_info : t_6797, t_7799
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_6798, t_7800
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,    # trace_info : t_6799, t_7801
            mask_func=attention_mask_func,                                     # trace_info : t_6800, t_7802
            softmax_in_fp32=self.config.attention_softmax_in_fp32,             # trace_info : t_6801, t_7803
            scale=coeff,                                                       # trace_info : t_6802, t_7804
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(                             # trace_info : t_6818, t_6820, t_7820, t_7822
            self.config.attention_dropout if attention_dropout is None else attention_dropout# trace_info : t_6819, t_7821
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
        assert packed_seq_params is None, (                                    # trace_info : t_15403, t_15906, t_19044, t_19545, t_22683, ...
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
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:# trace_info : t_15404, t_15907, t_19045, t_19546, t_22684, ...
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (                                                        # trace_info : t_15409, t_15912, t_19050, t_19551, t_22689, ...
            query.size(1),                                                     # trace_info : t_15405, t_15908, t_19046, t_19547, t_22685, ...
            query.size(2),                                                     # trace_info : t_15406, t_15909, t_19047, t_19548, t_22686, ...
            query.size(0),                                                     # trace_info : t_15407, t_15910, t_19048, t_19549, t_22687, ...
            key.size(0),                                                       # trace_info : t_15408, t_15911, t_19049, t_19550, t_22688, ...
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)# trace_info : t_15410, t_15913, t_19051, t_19552, t_22690, ...
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)    # trace_info : t_15411, t_15914, t_19052, t_19553, t_22691, ...

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(# trace_info : t_15412, t_15416, t_15915, t_15919, t_19053, ...
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",# trace_info : t_15415, t_15918, t_19056, t_19557, t_22695, ...
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(                                         # trace_info : t_15423, t_15429, t_15924, t_15930, t_19062, ...
            matmul_input_buffer,                                               # trace_info : t_15424, t_15925, t_19063, t_19564, t_22702, ...
            query.transpose(0, 1),  # [b * np, sq, hn]                         # trace_info : t_15425, t_15926, t_19064, t_19565, t_22703, ...
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]           # trace_info : t_15426, t_15927, t_19065, t_19566, t_22704, ...
            beta=0.0,                                                          # trace_info : t_15427, t_15928, t_19066, t_19567, t_22705, ...
            alpha=(1.0 / self.norm_factor),                                    # trace_info : t_15428, t_15929, t_19067, t_19568, t_22706, ...
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)                    # trace_info : t_15430, t_15931, t_19069, t_19570, t_22708, ...

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)# trace_info : t_15431, t_15932, t_19070, t_19571, t_22709, ...

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:                                  # trace_info : t_15463, t_15964, t_19102, t_19603, t_22741, ...
            with tensor_parallel.get_cuda_rng_tracker().fork():                # trace_info : t_15464, t_15485, t_15965, t_15986, t_19103, ...
                attention_probs = self.attention_dropout(attention_probs)      # trace_info : t_15484, t_15985, t_19123, t_19624, t_22762, ...
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (                                                        # trace_info : t_15502, t_16003, t_19141, t_19642, t_22780, ...
            value.size(1),                                                     # trace_info : t_15498, t_15999, t_19137, t_19638, t_22776, ...
            value.size(2),                                                     # trace_info : t_15499, t_16000, t_19138, t_19639, t_22777, ...
            query.size(0),                                                     # trace_info : t_15500, t_16001, t_19139, t_19640, t_22778, ...
            value.size(3),                                                     # trace_info : t_15501, t_16002, t_19140, t_19641, t_22779, ...
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1) # trace_info : t_15503, t_16004, t_19142, t_19643, t_22781, ...

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)# trace_info : t_15504, t_16005, t_19143, t_19644, t_22782, ...

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))            # trace_info : t_15505, t_16006, t_19144, t_19645, t_22783, ...

        # change view [b, np, sq, hn]
        context = context.view(*output_size)                                   # trace_info : t_15506, t_16007, t_19145, t_19646, t_22784, ...

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()                     # trace_info : t_15507, t_16008, t_19146, t_19647, t_22785, ...

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)# trace_info : t_15508, t_16009, t_19147, t_19648, t_22786, ...
        context = context.view(*new_context_shape)                             # trace_info : t_15509, t_16010, t_19148, t_19649, t_22787, ...

        return context                                                         # trace_info : t_15510, t_16011, t_19149, t_19650, t_22788, ...
