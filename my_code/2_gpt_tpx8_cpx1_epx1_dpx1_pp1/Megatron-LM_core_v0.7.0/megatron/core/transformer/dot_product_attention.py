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
        super().__init__(config=config)                                        # trace_info : t_11627, t_12629

        self.config: TransformerConfig = config                                # trace_info : t_11630, t_12632

        assert (
            self.config.context_parallel_size == 1                             # trace_info : t_11631, t_12633
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None                                    # trace_info : t_11632, t_12634
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)                               # trace_info : t_11633, t_12635
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_11634, t_12636
        self.attention_type = attention_type  # unused for now                 # trace_info : t_11635, t_12637

        projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_11636, t_12638

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_11637, t_12639
        self.hidden_size_per_partition = divide(projection_size, world_size)   # trace_info : t_11643, t_12645
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)# trace_info : t_11647, t_12649
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_11651, t_12653
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_11655, t_12657

        coeff = None                                                           # trace_info : t_11659, t_12661
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)      # trace_info : t_11660, t_12662
        if self.config.apply_query_key_layer_scaling:                          # trace_info : t_11661, t_12663
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(                       # trace_info : t_11662, t_11670, t_12664, t_12672
            input_in_fp16=self.config.fp16,                                    # trace_info : t_11663, t_12665
            input_in_bf16=self.config.bf16,                                    # trace_info : t_11664, t_12666
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_11665, t_12667
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,    # trace_info : t_11666, t_12668
            mask_func=attention_mask_func,                                     # trace_info : t_11667, t_12669
            softmax_in_fp32=self.config.attention_softmax_in_fp32,             # trace_info : t_11668, t_12670
            scale=coeff,                                                       # trace_info : t_11669, t_12671
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(                             # trace_info : t_11685, t_11687, t_12687, t_12689
            self.config.attention_dropout if attention_dropout is None else attention_dropout# trace_info : t_11686, t_12688
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
        assert packed_seq_params is None, (                                    # trace_info : t_20268, t_20755, t_23880, t_24365, t_27490, ...
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
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:# trace_info : t_20269, t_20756, t_23881, t_24366, t_27491, ...
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (                                                        # trace_info : t_20274, t_20761, t_23886, t_24371, t_27496, ...
            query.size(1),                                                     # trace_info : t_20270, t_20757, t_23882, t_24367, t_27492, ...
            query.size(2),                                                     # trace_info : t_20271, t_20758, t_23883, t_24368, t_27493, ...
            query.size(0),                                                     # trace_info : t_20272, t_20759, t_23884, t_24369, t_27494, ...
            key.size(0),                                                       # trace_info : t_20273, t_20760, t_23885, t_24370, t_27495, ...
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)# trace_info : t_20275, t_20762, t_23887, t_24372, t_27497, ...
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)    # trace_info : t_20276, t_20763, t_23888, t_24373, t_27498, ...

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(# trace_info : t_20277, t_20281, t_20764, t_20768, t_23889, ...
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",# trace_info : t_20280, t_20767, t_23892, t_24377, t_27502, ...
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(                                         # trace_info : t_20288, t_20294, t_20773, t_20779, t_23898, ...
            matmul_input_buffer,                                               # trace_info : t_20289, t_20774, t_23899, t_24384, t_27509, ...
            query.transpose(0, 1),  # [b * np, sq, hn]                         # trace_info : t_20290, t_20775, t_23900, t_24385, t_27510, ...
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]           # trace_info : t_20291, t_20776, t_23901, t_24386, t_27511, ...
            beta=0.0,                                                          # trace_info : t_20292, t_20777, t_23902, t_24387, t_27512, ...
            alpha=(1.0 / self.norm_factor),                                    # trace_info : t_20293, t_20778, t_23903, t_24388, t_27513, ...
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)                    # trace_info : t_20295, t_20780, t_23905, t_24390, t_27515, ...

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)# trace_info : t_20296, t_20781, t_23906, t_24391, t_27516, ...

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:                                  # trace_info : t_20320, t_20805, t_23930, t_24415, t_27540, ...
            with tensor_parallel.get_cuda_rng_tracker().fork():                # trace_info : t_20321, t_20342, t_20806, t_20827, t_23931, ...
                attention_probs = self.attention_dropout(attention_probs)      # trace_info : t_20341, t_20826, t_23951, t_24436, t_27561, ...
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (                                                        # trace_info : t_20359, t_20844, t_23969, t_24454, t_27579, ...
            value.size(1),                                                     # trace_info : t_20355, t_20840, t_23965, t_24450, t_27575, ...
            value.size(2),                                                     # trace_info : t_20356, t_20841, t_23966, t_24451, t_27576, ...
            query.size(0),                                                     # trace_info : t_20357, t_20842, t_23967, t_24452, t_27577, ...
            value.size(3),                                                     # trace_info : t_20358, t_20843, t_23968, t_24453, t_27578, ...
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1) # trace_info : t_20360, t_20845, t_23970, t_24455, t_27580, ...

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)# trace_info : t_20361, t_20846, t_23971, t_24456, t_27581, ...

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))            # trace_info : t_20362, t_20847, t_23972, t_24457, t_27582, ...

        # change view [b, np, sq, hn]
        context = context.view(*output_size)                                   # trace_info : t_20363, t_20848, t_23973, t_24458, t_27583, ...

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()                     # trace_info : t_20364, t_20849, t_23974, t_24459, t_27584, ...

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)# trace_info : t_20365, t_20850, t_23975, t_24460, t_27585, ...
        context = context.view(*new_context_shape)                             # trace_info : t_20366, t_20851, t_23976, t_24461, t_27586, ...

        return context                                                         # trace_info : t_20367, t_20852, t_23977, t_24462, t_27587, ...
