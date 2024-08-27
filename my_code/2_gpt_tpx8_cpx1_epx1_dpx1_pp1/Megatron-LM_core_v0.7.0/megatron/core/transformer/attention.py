# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass                                              # trace_info : t_11226
from importlib.metadata import version                                         # trace_info : t_11227
from typing import Union                                                       # trace_info : t_11228
                                                                               # trace_info : t_11229
import torch                                                                   # trace_info : t_11230
from pkg_resources import packaging

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

from .enums import AttnMaskType
from .transformer_config import TransformerConfig


@dataclass
class SelfAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class CrossAttentionSubmodules:
    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class Attention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
    ):
        super().__init__(config=config)                                        # trace_info : t_11577, t_12579

        self.config = config                                                   # trace_info : t_11580, t_12582
        self.layer_number = layer_number                                       # trace_info : t_11581, t_12583
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_11582, t_12584
        self.attention_type = attention_type                                   # trace_info : t_11583, t_12585

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_11584, t_12586
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups# trace_info : t_11585, t_12587

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_11586, t_12588
        self.hidden_size_per_attention_head = divide(                          # trace_info : t_11592, t_11594, t_12594, t_12596
            self.query_projection_size, self.config.num_attention_heads        # trace_info : t_11593, t_12595
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_11598, t_12600
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_11602, t_12604

        self.core_attention = build_module(                                    # trace_info : t_11606, t_11612, t_12608, t_12614
            submodules.core_attention,                                         # trace_info : t_11607, t_12609
            config=self.config,                                                # trace_info : t_11608, t_12610
            layer_number=self.layer_number,                                    # trace_info : t_11609, t_12611
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_11610, t_12612
            attention_type=self.attention_type,                                # trace_info : t_11611, t_12613
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'# trace_info : t_11688, t_12690

        # Output.
        self.linear_proj = build_module(                                       # trace_info : t_11689, t_11700, t_12691, t_12702
            submodules.linear_proj,                                            # trace_info : t_11690, t_12692
            self.query_projection_size,                                        # trace_info : t_11691, t_12693
            self.config.hidden_size,                                           # trace_info : t_11692, t_12694
            config=self.config,                                                # trace_info : t_11693, t_12695
            init_method=self.config.output_layer_init_method,                  # trace_info : t_11694, t_12696
            bias=self.config.add_bias_linear,                                  # trace_info : t_11695, t_12697
            input_is_parallel=True,                                            # trace_info : t_11696, t_12698
            skip_bias_add=True,                                                # trace_info : t_11697, t_12699
            is_expert=False,                                                   # trace_info : t_11698, t_12700
            tp_comm_buffer_name='proj',                                        # trace_info : t_11699, t_12701
        )

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_length, batch_size, dtype):
        """Allocate memory to store kv cache during inference."""

        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _adjust_key_value_for_inference(self, inference_params, key, value, rotary_pos_emb):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params, as well as
        adjusted rotary_pos_emb.

        Returns a tuple: (key, value, rotary_pos_emb)

        """
        attn_mask_type = self.attn_mask_type                                   # trace_info : t_20254, t_20741, t_23866, t_24351, t_27476, ...
        if inference_params is None:                                           # trace_info : t_20255, t_20742, t_23867, t_24352, t_27477, ...
            return key, value, rotary_pos_emb, attn_mask_type                  # trace_info : t_20256, t_20743, t_23868, t_24353, t_27478, ...

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, key.dtype
            )
            inference_value_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, value.dtype
            )
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
            is_first_step = True
        else:
            # Get the pre-allocated buffers for this layer
            inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                self.layer_number
            ]
            attn_mask_type = AttnMaskType.no_mask

        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
        inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
        key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
        value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        return key, value, rotary_pos_emb, attn_mask_type

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):# trace_info : t_20170, t_20657, t_23782, t_24267, t_27392, ...
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)# trace_info : t_20171, t_20658, t_23783, t_24268, t_27393, ...

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(# trace_info : t_20251, t_20253, t_20738, t_20740, t_23863, ...
            inference_params, key, value, rotary_pos_emb                       # trace_info : t_20252, t_20739, t_23864, t_24349, t_27474, ...
        )

        if packed_seq_params is not None:                                      # trace_info : t_20257, t_20744, t_23869, t_24354, t_27479, ...
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:                                         # trace_info : t_20258, t_20745, t_23870, t_24355, t_27480, ...
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
            )
            key = apply_rotary_pos_emb(
                key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
            )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:                   # trace_info : t_20259, t_20746, t_23871, t_24356, t_27481, ...
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(                               # trace_info : t_20260, t_20267, t_20747, t_20754, t_23872, ...
                query,                                                         # trace_info : t_20261, t_20748, t_23873, t_24358, t_27483, ...
                key,                                                           # trace_info : t_20262, t_20749, t_23874, t_24359, t_27484, ...
                value,                                                         # trace_info : t_20263, t_20750, t_23875, t_24360, t_27485, ...
                attention_mask,                                                # trace_info : t_20264, t_20751, t_23876, t_24361, t_27486, ...
                attn_mask_type=attn_mask_type,                                 # trace_info : t_20265, t_20752, t_23877, t_24362, t_27487, ...
                packed_seq_params=packed_seq_params,                           # trace_info : t_20266, t_20753, t_23878, t_24363, t_27488, ...
            )

        if packed_seq_params is not None:                                      # trace_info : t_20368, t_20853, t_23978, t_24463, t_27588, ...
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)                         # trace_info : t_20369, t_20854, t_23979, t_24464, t_27589, ...

        return output, bias                                                    # trace_info : t_20429, t_20914, t_24039, t_24524, t_27649, ...


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(                                                      # trace_info : t_11570, t_11576, t_12572, t_12578
            config=config,                                                     # trace_info : t_11571, t_12573
            submodules=submodules,                                             # trace_info : t_11572, t_12574
            layer_number=layer_number,                                         # trace_info : t_11573, t_12575
            attn_mask_type=attn_mask_type,                                     # trace_info : t_11574, t_12576
            attention_type="self",                                             # trace_info : t_11575, t_12577
        )

        self.linear_qkv = build_module(                                        # trace_info : t_11830, t_11841, t_12832, t_12843
            submodules.linear_qkv,                                             # trace_info : t_11831, t_12833
            self.config.hidden_size,                                           # trace_info : t_11832, t_12834
            self.query_projection_size + 2 * self.kv_projection_size,          # trace_info : t_11833, t_12835
            config=self.config,                                                # trace_info : t_11834, t_12836
            init_method=self.config.init_method,                               # trace_info : t_11835, t_12837
            gather_output=False,                                               # trace_info : t_11836, t_12838
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,      # trace_info : t_11837, t_12839
            skip_bias_add=False,                                               # trace_info : t_11838, t_12840
            is_expert=False,                                                   # trace_info : t_11839, t_12841
            tp_comm_buffer_name='qkv',                                         # trace_info : t_11840, t_12842
        )

        if submodules.q_layernorm is not None:                                 # trace_info : t_11988, t_12990
            self.q_layernorm = build_module(                                   # trace_info : t_11989, t_11994, t_12991, t_12996
                submodules.q_layernorm,                                        # trace_info : t_11990, t_12992
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_11991, t_12993
                config=self.config,                                            # trace_info : t_11992, t_12994
                eps=self.config.layernorm_epsilon,                             # trace_info : t_11993, t_12995
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:                                 # trace_info : t_12010, t_13012
            self.k_layernorm = build_module(                                   # trace_info : t_12011, t_12016, t_13013, t_13018
                submodules.k_layernorm,                                        # trace_info : t_12012, t_13014
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_12013, t_13015
                config=self.config,                                            # trace_info : t_12014, t_13016
                eps=self.config.layernorm_epsilon,                             # trace_info : t_12015, t_13017
            )
        else:
            self.k_layernorm = None

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably not
        required; transmitting hashes is sufficient."""

        if not self.config.qk_layernorm:
            return

        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        rank = get_data_parallel_rank()
        inputs = torch.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())

        def _compare(srcs, tgts, names, parallelism):
            assert len(srcs) == len(tgts) == len(names)
            for src, tgt, name in zip(srcs, tgts, names):
                assert torch.all(
                    src == tgt
                ), f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. Diff: {torch.norm(src - tgt)}"

        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )

        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())

        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)                          # trace_info : t_20172, t_20659, t_23784, t_24269, t_27394, ...

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (                           # trace_info : t_20225, t_20230, t_20712, t_20717, t_23837, ...
            self.num_query_groups_per_partition,                               # trace_info : t_20226, t_20713, t_23838, t_24323, t_27448, ...
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)# trace_info : t_20227, t_20229, t_20714, t_20716, t_23839, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_20228, t_20715, t_23840, t_24325, t_27450, ...
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)                          # trace_info : t_20231, t_20718, t_23843, t_24328, t_27453, ...

        split_arg_list = [                                                     # trace_info : t_20239, t_20726, t_23851, t_24336, t_27461, ...
            (
                self.num_attention_heads_per_partition                         # trace_info : t_20232, t_20234, t_20236, t_20719, t_20721, ...
                // self.num_query_groups_per_partition                         # trace_info : t_20233, t_20720, t_23845, t_24330, t_27455, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_20235, t_20722, t_23847, t_24332, t_27457, ...
            ),
            self.hidden_size_per_attention_head,                               # trace_info : t_20237, t_20724, t_23849, t_24334, t_27459, ...
            self.hidden_size_per_attention_head,                               # trace_info : t_20238, t_20725, t_23850, t_24335, t_27460, ...
        ]

        if SplitAlongDim is not None:                                          # trace_info : t_20240, t_20727, t_23852, t_24337, t_27462, ...

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,) # trace_info : t_20241, t_20728, t_23853, t_24338, t_27463, ...
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)# trace_info : t_20242, t_20729, t_23854, t_24339, t_27464, ...

        if self.q_layernorm is not None:                                       # trace_info : t_20243, t_20730, t_23855, t_24340, t_27465, ...
            query = self.q_layernorm(query)                                    # trace_info : t_20244, t_20731, t_23856, t_24341, t_27466, ...

        if self.k_layernorm is not None:                                       # trace_info : t_20246, t_20733, t_23858, t_24343, t_27468, ...
            key = self.k_layernorm(key)                                        # trace_info : t_20247, t_20734, t_23859, t_24344, t_27469, ...

        if self.config.test_mode:                                              # trace_info : t_20249, t_20736, t_23861, t_24346, t_27471, ...
            self.run_realtime_tests()

        return query, key, value                                               # trace_info : t_20250, t_20737, t_23862, t_24347, t_27472, ...


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
        )

        if self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Group query attention is not currently supported in cross attention."
            )
        assert self.query_projection_size == self.kv_projection_size

        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_module(
            submodules.linear_kv,
            self.config.hidden_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
