# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass                                              # trace_info : t_9668
from importlib.metadata import version                                         # trace_info : t_9669
from typing import Union                                                       # trace_info : t_9670
                                                                               # trace_info : t_9671
import torch                                                                   # trace_info : t_9672
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
        super().__init__(config=config)                                        # trace_info : t_9995, t_10774

        self.config = config                                                   # trace_info : t_9998, t_10777
        self.layer_number = layer_number                                       # trace_info : t_9999, t_10778
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_10000, t_10779
        self.attention_type = attention_type                                   # trace_info : t_10001, t_10780

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_10002, t_10781
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups# trace_info : t_10003, t_10782

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_10004, t_10783
        self.hidden_size_per_attention_head = divide(                          # trace_info : t_10010, t_10012, t_10789, t_10791
            self.query_projection_size, self.config.num_attention_heads        # trace_info : t_10011, t_10790
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_10016, t_10795
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_10020, t_10799

        self.core_attention = build_module(                                    # trace_info : t_10024, t_10030, t_10803, t_10809
            submodules.core_attention,                                         # trace_info : t_10025, t_10804
            config=self.config,                                                # trace_info : t_10026, t_10805
            layer_number=self.layer_number,                                    # trace_info : t_10027, t_10806
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_10028, t_10807
            attention_type=self.attention_type,                                # trace_info : t_10029, t_10808
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'# trace_info : t_10097, t_10875

        # Output.
        self.linear_proj = build_module(                                       # trace_info : t_10098, t_10109, t_10876, t_10887
            submodules.linear_proj,                                            # trace_info : t_10099, t_10877
            self.query_projection_size,                                        # trace_info : t_10100, t_10878
            self.config.hidden_size,                                           # trace_info : t_10101, t_10879
            config=self.config,                                                # trace_info : t_10102, t_10880
            init_method=self.config.output_layer_init_method,                  # trace_info : t_10103, t_10881
            bias=self.config.add_bias_linear,                                  # trace_info : t_10104, t_10882
            input_is_parallel=True,                                            # trace_info : t_10105, t_10883
            skip_bias_add=True,                                                # trace_info : t_10106, t_10884
            is_expert=False,                                                   # trace_info : t_10107, t_10885
            tp_comm_buffer_name='proj',                                        # trace_info : t_10108, t_10886
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
        attn_mask_type = self.attn_mask_type                                   # trace_info : t_18311, t_18518, t_21497, t_21704, t_24683, ...
        if inference_params is None:                                           # trace_info : t_18312, t_18519, t_21498, t_21705, t_24684, ...
            return key, value, rotary_pos_emb, attn_mask_type                  # trace_info : t_18313, t_18520, t_21499, t_21706, t_24685, ...

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
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):# trace_info : t_18273, t_18480, t_21459, t_21666, t_24645, ...
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)# trace_info : t_18274, t_18481, t_21460, t_21667, t_24646, ...

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(# trace_info : t_18308, t_18310, t_18515, t_18517, t_21494, ...
            inference_params, key, value, rotary_pos_emb                       # trace_info : t_18309, t_18516, t_21495, t_21702, t_24681, ...
        )

        if packed_seq_params is not None:                                      # trace_info : t_18314, t_18521, t_21500, t_21707, t_24686, ...
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:                                         # trace_info : t_18315, t_18522, t_21501, t_21708, t_24687, ...
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

        if self.checkpoint_core_attention and self.training:                   # trace_info : t_18316, t_18523, t_21502, t_21709, t_24688, ...
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(                               # trace_info : t_18317, t_18324, t_18524, t_18531, t_21503, ...
                query,                                                         # trace_info : t_18318, t_18525, t_21504, t_21711, t_24690, ...
                key,                                                           # trace_info : t_18319, t_18526, t_21505, t_21712, t_24691, ...
                value,                                                         # trace_info : t_18320, t_18527, t_21506, t_21713, t_24692, ...
                attention_mask,                                                # trace_info : t_18321, t_18528, t_21507, t_21714, t_24693, ...
                attn_mask_type=attn_mask_type,                                 # trace_info : t_18322, t_18529, t_21508, t_21715, t_24694, ...
                packed_seq_params=packed_seq_params,                           # trace_info : t_18323, t_18530, t_21509, t_21716, t_24695, ...
            )

        if packed_seq_params is not None:                                      # trace_info : t_18374, t_18581, t_21560, t_21767, t_24746, ...
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)                         # trace_info : t_18375, t_18582, t_21561, t_21768, t_24747, ...

        return output, bias                                                    # trace_info : t_18382, t_18589, t_21568, t_21775, t_24754, ...


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
        super().__init__(                                                      # trace_info : t_9988, t_9994, t_10767, t_10773
            config=config,                                                     # trace_info : t_9989, t_10768
            submodules=submodules,                                             # trace_info : t_9990, t_10769
            layer_number=layer_number,                                         # trace_info : t_9991, t_10770
            attn_mask_type=attn_mask_type,                                     # trace_info : t_9992, t_10771
            attention_type="self",                                             # trace_info : t_9993, t_10772
        )

        self.linear_qkv = build_module(                                        # trace_info : t_10208, t_10219, t_10986, t_10997
            submodules.linear_qkv,                                             # trace_info : t_10209, t_10987
            self.config.hidden_size,                                           # trace_info : t_10210, t_10988
            self.query_projection_size + 2 * self.kv_projection_size,          # trace_info : t_10211, t_10989
            config=self.config,                                                # trace_info : t_10212, t_10990
            init_method=self.config.init_method,                               # trace_info : t_10213, t_10991
            gather_output=False,                                               # trace_info : t_10214, t_10992
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,      # trace_info : t_10215, t_10993
            skip_bias_add=False,                                               # trace_info : t_10216, t_10994
            is_expert=False,                                                   # trace_info : t_10217, t_10995
            tp_comm_buffer_name='qkv',                                         # trace_info : t_10218, t_10996
        )

        if submodules.q_layernorm is not None:                                 # trace_info : t_10311, t_11089
            self.q_layernorm = build_module(                                   # trace_info : t_10312, t_10317, t_11090, t_11095
                submodules.q_layernorm,                                        # trace_info : t_10313, t_11091
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10314, t_11092
                config=self.config,                                            # trace_info : t_10315, t_11093
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10316, t_11094
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:                                 # trace_info : t_10333, t_11111
            self.k_layernorm = build_module(                                   # trace_info : t_10334, t_10339, t_11112, t_11117
                submodules.k_layernorm,                                        # trace_info : t_10335, t_11113
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10336, t_11114
                config=self.config,                                            # trace_info : t_10337, t_11115
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10338, t_11116
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
        mixed_qkv, _ = self.linear_qkv(hidden_states)                          # trace_info : t_18275, t_18482, t_21461, t_21668, t_24647, ...

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (                           # trace_info : t_18282, t_18287, t_18489, t_18494, t_21468, ...
            self.num_query_groups_per_partition,                               # trace_info : t_18283, t_18490, t_21469, t_21676, t_24655, ...
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)# trace_info : t_18284, t_18286, t_18491, t_18493, t_21470, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18285, t_18492, t_21471, t_21678, t_24657, ...
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)                          # trace_info : t_18288, t_18495, t_21474, t_21681, t_24660, ...

        split_arg_list = [                                                     # trace_info : t_18296, t_18503, t_21482, t_21689, t_24668, ...
            (
                self.num_attention_heads_per_partition                         # trace_info : t_18289, t_18291, t_18293, t_18496, t_18498, ...
                // self.num_query_groups_per_partition                         # trace_info : t_18290, t_18497, t_21476, t_21683, t_24662, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18292, t_18499, t_21478, t_21685, t_24664, ...
            ),
            self.hidden_size_per_attention_head,                               # trace_info : t_18294, t_18501, t_21480, t_21687, t_24666, ...
            self.hidden_size_per_attention_head,                               # trace_info : t_18295, t_18502, t_21481, t_21688, t_24667, ...
        ]

        if SplitAlongDim is not None:                                          # trace_info : t_18297, t_18504, t_21483, t_21690, t_24669, ...

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,) # trace_info : t_18298, t_18505, t_21484, t_21691, t_24670, ...
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)# trace_info : t_18299, t_18506, t_21485, t_21692, t_24671, ...

        if self.q_layernorm is not None:                                       # trace_info : t_18300, t_18507, t_21486, t_21693, t_24672, ...
            query = self.q_layernorm(query)                                    # trace_info : t_18301, t_18508, t_21487, t_21694, t_24673, ...

        if self.k_layernorm is not None:                                       # trace_info : t_18303, t_18510, t_21489, t_21696, t_24675, ...
            key = self.k_layernorm(key)                                        # trace_info : t_18304, t_18511, t_21490, t_21697, t_24676, ...

        if self.config.test_mode:                                              # trace_info : t_18306, t_18513, t_21492, t_21699, t_24678, ...
            self.run_realtime_tests()

        return query, key, value                                               # trace_info : t_18307, t_18514, t_21493, t_21700, t_24679, ...


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
