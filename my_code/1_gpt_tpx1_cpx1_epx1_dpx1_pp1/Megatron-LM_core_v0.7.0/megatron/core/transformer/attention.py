# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass                                              # trace_info : t_6359
from importlib.metadata import version                                         # trace_info : t_6360
from typing import Union                                                       # trace_info : t_6361
                                                                               # trace_info : t_6362
import torch                                                                   # trace_info : t_6363
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
        super().__init__(config=config)                                        # trace_info : t_6710, t_7712

        self.config = config                                                   # trace_info : t_6713, t_7715
        self.layer_number = layer_number                                       # trace_info : t_6714, t_7716
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_6715, t_7717
        self.attention_type = attention_type                                   # trace_info : t_6716, t_7718

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_6717, t_7719
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups# trace_info : t_6718, t_7720

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_6719, t_7721
        self.hidden_size_per_attention_head = divide(                          # trace_info : t_6725, t_6727, t_7727, t_7729
            self.query_projection_size, self.config.num_attention_heads        # trace_info : t_6726, t_7728
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_6731, t_7733
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_6735, t_7737

        self.core_attention = build_module(                                    # trace_info : t_6739, t_6745, t_7741, t_7747
            submodules.core_attention,                                         # trace_info : t_6740, t_7742
            config=self.config,                                                # trace_info : t_6741, t_7743
            layer_number=self.layer_number,                                    # trace_info : t_6742, t_7744
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_6743, t_7745
            attention_type=self.attention_type,                                # trace_info : t_6744, t_7746
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'# trace_info : t_6821, t_7823

        # Output.
        self.linear_proj = build_module(                                       # trace_info : t_6822, t_6833, t_7824, t_7835
            submodules.linear_proj,                                            # trace_info : t_6823, t_7825
            self.query_projection_size,                                        # trace_info : t_6824, t_7826
            self.config.hidden_size,                                           # trace_info : t_6825, t_7827
            config=self.config,                                                # trace_info : t_6826, t_7828
            init_method=self.config.output_layer_init_method,                  # trace_info : t_6827, t_7829
            bias=self.config.add_bias_linear,                                  # trace_info : t_6828, t_7830
            input_is_parallel=True,                                            # trace_info : t_6829, t_7831
            skip_bias_add=True,                                                # trace_info : t_6830, t_7832
            is_expert=False,                                                   # trace_info : t_6831, t_7833
            tp_comm_buffer_name='proj',                                        # trace_info : t_6832, t_7834
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
        attn_mask_type = self.attn_mask_type                                   # trace_info : t_15389, t_15892, t_19030, t_19531, t_22669, ...
        if inference_params is None:                                           # trace_info : t_15390, t_15893, t_19031, t_19532, t_22670, ...
            return key, value, rotary_pos_emb, attn_mask_type                  # trace_info : t_15391, t_15894, t_19032, t_19533, t_22671, ...

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
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):# trace_info : t_15297, t_15800, t_18938, t_19439, t_22577, ...
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)# trace_info : t_15298, t_15801, t_18939, t_19440, t_22578, ...

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(# trace_info : t_15386, t_15388, t_15889, t_15891, t_19027, ...
            inference_params, key, value, rotary_pos_emb                       # trace_info : t_15387, t_15890, t_19028, t_19529, t_22667, ...
        )

        if packed_seq_params is not None:                                      # trace_info : t_15392, t_15895, t_19033, t_19534, t_22672, ...
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:                                         # trace_info : t_15393, t_15896, t_19034, t_19535, t_22673, ...
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

        if self.checkpoint_core_attention and self.training:                   # trace_info : t_15394, t_15897, t_19035, t_19536, t_22674, ...
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(                               # trace_info : t_15395, t_15402, t_15898, t_15905, t_19036, ...
                query,                                                         # trace_info : t_15396, t_15899, t_19037, t_19538, t_22676, ...
                key,                                                           # trace_info : t_15397, t_15900, t_19038, t_19539, t_22677, ...
                value,                                                         # trace_info : t_15398, t_15901, t_19039, t_19540, t_22678, ...
                attention_mask,                                                # trace_info : t_15399, t_15902, t_19040, t_19541, t_22679, ...
                attn_mask_type=attn_mask_type,                                 # trace_info : t_15400, t_15903, t_19041, t_19542, t_22680, ...
                packed_seq_params=packed_seq_params,                           # trace_info : t_15401, t_15904, t_19042, t_19543, t_22681, ...
            )

        if packed_seq_params is not None:                                      # trace_info : t_15511, t_16012, t_19150, t_19651, t_22789, ...
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)                         # trace_info : t_15512, t_16013, t_19151, t_19652, t_22790, ...

        return output, bias                                                    # trace_info : t_15568, t_16069, t_19207, t_19708, t_22846, ...


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
        super().__init__(                                                      # trace_info : t_6703, t_6709, t_7705, t_7711
            config=config,                                                     # trace_info : t_6704, t_7706
            submodules=submodules,                                             # trace_info : t_6705, t_7707
            layer_number=layer_number,                                         # trace_info : t_6706, t_7708
            attn_mask_type=attn_mask_type,                                     # trace_info : t_6707, t_7709
            attention_type="self",                                             # trace_info : t_6708, t_7710
        )

        self.linear_qkv = build_module(                                        # trace_info : t_6963, t_6974, t_7965, t_7976
            submodules.linear_qkv,                                             # trace_info : t_6964, t_7966
            self.config.hidden_size,                                           # trace_info : t_6965, t_7967
            self.query_projection_size + 2 * self.kv_projection_size,          # trace_info : t_6966, t_7968
            config=self.config,                                                # trace_info : t_6967, t_7969
            init_method=self.config.init_method,                               # trace_info : t_6968, t_7970
            gather_output=False,                                               # trace_info : t_6969, t_7971
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,      # trace_info : t_6970, t_7972
            skip_bias_add=False,                                               # trace_info : t_6971, t_7973
            is_expert=False,                                                   # trace_info : t_6972, t_7974
            tp_comm_buffer_name='qkv',                                         # trace_info : t_6973, t_7975
        )

        if submodules.q_layernorm is not None:                                 # trace_info : t_7121, t_8123
            self.q_layernorm = build_module(                                   # trace_info : t_7122, t_7127, t_8124, t_8129
                submodules.q_layernorm,                                        # trace_info : t_7123, t_8125
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_7124, t_8126
                config=self.config,                                            # trace_info : t_7125, t_8127
                eps=self.config.layernorm_epsilon,                             # trace_info : t_7126, t_8128
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:                                 # trace_info : t_7143, t_8145
            self.k_layernorm = build_module(                                   # trace_info : t_7144, t_7149, t_8146, t_8151
                submodules.k_layernorm,                                        # trace_info : t_7145, t_8147
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_7146, t_8148
                config=self.config,                                            # trace_info : t_7147, t_8149
                eps=self.config.layernorm_epsilon,                             # trace_info : t_7148, t_8150
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
        mixed_qkv, _ = self.linear_qkv(hidden_states)                          # trace_info : t_15299, t_15802, t_18940, t_19441, t_22579, ...

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (                           # trace_info : t_15360, t_15365, t_15863, t_15868, t_19001, ...
            self.num_query_groups_per_partition,                               # trace_info : t_15361, t_15864, t_19002, t_19503, t_22641, ...
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)# trace_info : t_15362, t_15364, t_15865, t_15867, t_19003, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_15363, t_15866, t_19004, t_19505, t_22643, ...
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)                          # trace_info : t_15366, t_15869, t_19007, t_19508, t_22646, ...

        split_arg_list = [                                                     # trace_info : t_15374, t_15877, t_19015, t_19516, t_22654, ...
            (
                self.num_attention_heads_per_partition                         # trace_info : t_15367, t_15369, t_15371, t_15870, t_15872, ...
                // self.num_query_groups_per_partition                         # trace_info : t_15368, t_15871, t_19009, t_19510, t_22648, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_15370, t_15873, t_19011, t_19512, t_22650, ...
            ),
            self.hidden_size_per_attention_head,                               # trace_info : t_15372, t_15875, t_19013, t_19514, t_22652, ...
            self.hidden_size_per_attention_head,                               # trace_info : t_15373, t_15876, t_19014, t_19515, t_22653, ...
        ]

        if SplitAlongDim is not None:                                          # trace_info : t_15375, t_15878, t_19016, t_19517, t_22655, ...

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,) # trace_info : t_15376, t_15879, t_19017, t_19518, t_22656, ...
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)# trace_info : t_15377, t_15880, t_19018, t_19519, t_22657, ...

        if self.q_layernorm is not None:                                       # trace_info : t_15378, t_15881, t_19019, t_19520, t_22658, ...
            query = self.q_layernorm(query)                                    # trace_info : t_15379, t_15882, t_19020, t_19521, t_22659, ...

        if self.k_layernorm is not None:                                       # trace_info : t_15381, t_15884, t_19022, t_19523, t_22661, ...
            key = self.k_layernorm(key)                                        # trace_info : t_15382, t_15885, t_19023, t_19524, t_22662, ...

        if self.config.test_mode:                                              # trace_info : t_15384, t_15887, t_19025, t_19526, t_22664, ...
            self.run_realtime_tests()

        return query, key, value                                               # trace_info : t_15385, t_15888, t_19026, t_19527, t_22665, ...


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
