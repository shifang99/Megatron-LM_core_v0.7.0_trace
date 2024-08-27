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
        super().__init__(config=config)                                        # trace_info : t_9789, t_10791

        self.config: TransformerConfig = config                                # trace_info : t_9792, t_10794

        assert (
            self.config.context_parallel_size == 1                             # trace_info : t_9793, t_10795
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None                                    # trace_info : t_9794, t_10796
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)                               # trace_info : t_9795, t_10797
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_9796, t_10798
        self.attention_type = attention_type  # unused for now                 # trace_info : t_9797, t_10799

        projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_9798, t_10800

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_9799, t_10801
        self.hidden_size_per_partition = divide(projection_size, world_size)   # trace_info : t_9805, t_10807
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)# trace_info : t_9809, t_10811
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_9813, t_10815
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_9817, t_10819

        coeff = None                                                           # trace_info : t_9821, t_10823
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)      # trace_info : t_9822, t_10824
        if self.config.apply_query_key_layer_scaling:                          # trace_info : t_9823, t_10825
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(                       # trace_info : t_9824, t_9832, t_10826, t_10834
            input_in_fp16=self.config.fp16,                                    # trace_info : t_9825, t_10827
            input_in_bf16=self.config.bf16,                                    # trace_info : t_9826, t_10828
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_9827, t_10829
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,    # trace_info : t_9828, t_10830
            mask_func=attention_mask_func,                                     # trace_info : t_9829, t_10831
            softmax_in_fp32=self.config.attention_softmax_in_fp32,             # trace_info : t_9830, t_10832
            scale=coeff,                                                       # trace_info : t_9831, t_10833
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(                             # trace_info : t_9847, t_9849, t_10849, t_10851
            self.config.attention_dropout if attention_dropout is None else attention_dropout# trace_info : t_9848, t_10850
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
        assert packed_seq_params is None, (                                    # trace_info : t_18547, t_19042, t_22186, t_22679, t_89793, ...
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
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:# trace_info : t_18548, t_19043, t_22187, t_22680, t_89794, ...
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (                                                        # trace_info : t_18553, t_19048, t_22192, t_22685, t_89799, ...
            query.size(1),                                                     # trace_info : t_18549, t_19044, t_22188, t_22681, t_89795, ...
            query.size(2),                                                     # trace_info : t_18550, t_19045, t_22189, t_22682, t_89796, ...
            query.size(0),                                                     # trace_info : t_18551, t_19046, t_22190, t_22683, t_89797, ...
            key.size(0),                                                       # trace_info : t_18552, t_19047, t_22191, t_22684, t_89798, ...
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)# trace_info : t_18554, t_19049, t_22193, t_22686, t_89800, ...
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)    # trace_info : t_18555, t_19050, t_22194, t_22687, t_89801, ...

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(# trace_info : t_18556, t_18560, t_19051, t_19055, t_22195, ...
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",# trace_info : t_18559, t_19054, t_22198, t_22691, t_89805, ...
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(                                         # trace_info : t_18567, t_18573, t_19060, t_19066, t_22204, ...
            matmul_input_buffer,                                               # trace_info : t_18568, t_19061, t_22205, t_22698, t_89812, ...
            query.transpose(0, 1),  # [b * np, sq, hn]                         # trace_info : t_18569, t_19062, t_22206, t_22699, t_89813, ...
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]           # trace_info : t_18570, t_19063, t_22207, t_22700, t_89814, ...
            beta=0.0,                                                          # trace_info : t_18571, t_19064, t_22208, t_22701, t_89815, ...
            alpha=(1.0 / self.norm_factor),                                    # trace_info : t_18572, t_19065, t_22209, t_22702, t_89816, ...
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)                    # trace_info : t_18574, t_19067, t_22211, t_22704, t_89818, ...

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)# trace_info : t_18575, t_19068, t_22212, t_22705, t_89819, ...

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:                                  # trace_info : t_18607, t_19100, t_22244, t_22737, t_89851, ...
            with tensor_parallel.get_cuda_rng_tracker().fork():                # trace_info : t_18608, t_18629, t_19101, t_19122, t_22245, ...
                attention_probs = self.attention_dropout(attention_probs)      # trace_info : t_18628, t_19121, t_22265, t_22758, t_89872, ...
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (                                                        # trace_info : t_18646, t_19139, t_22283, t_22776, t_89890, ...
            value.size(1),                                                     # trace_info : t_18642, t_19135, t_22279, t_22772, t_89886, ...
            value.size(2),                                                     # trace_info : t_18643, t_19136, t_22280, t_22773, t_89887, ...
            query.size(0),                                                     # trace_info : t_18644, t_19137, t_22281, t_22774, t_89888, ...
            value.size(3),                                                     # trace_info : t_18645, t_19138, t_22282, t_22775, t_89889, ...
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1) # trace_info : t_18647, t_19140, t_22284, t_22777, t_89891, ...

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)# trace_info : t_18648, t_19141, t_22285, t_22778, t_89892, ...

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))            # trace_info : t_18649, t_19142, t_22286, t_22779, t_89893, ...

        # change view [b, np, sq, hn]
        context = context.view(*output_size)                                   # trace_info : t_18650, t_19143, t_22287, t_22780, t_89894, ...

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()                     # trace_info : t_18651, t_19144, t_22288, t_22781, t_89895, ...

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)# trace_info : t_18652, t_19145, t_22289, t_22782, t_89896, ...
        context = context.view(*new_context_shape)                             # trace_info : t_18653, t_19146, t_22290, t_22783, t_89897, ...

        return context                                                         # trace_info : t_18654, t_19147, t_22291, t_22784, t_89898, ...
