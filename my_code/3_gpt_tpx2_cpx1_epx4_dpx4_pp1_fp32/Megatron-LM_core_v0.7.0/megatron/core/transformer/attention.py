# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass                                              # trace_info : t_9313
from importlib.metadata import version                                         # trace_info : t_9314
from typing import Union                                                       # trace_info : t_9315
                                                                               # trace_info : t_9316
import torch                                                                   # trace_info : t_9317
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
        super().__init__(config=config)                                        # trace_info : t_9664, t_10804

        self.config = config                                                   # trace_info : t_9667, t_10807
        self.layer_number = layer_number                                       # trace_info : t_9668, t_10808
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_9669, t_10809
        self.attention_type = attention_type                                   # trace_info : t_9670, t_10810

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_9671, t_10811
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups# trace_info : t_9672, t_10812

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_9673, t_10813
        self.hidden_size_per_attention_head = divide(                          # trace_info : t_9679, t_9681, t_10819, t_10821
            self.query_projection_size, self.config.num_attention_heads        # trace_info : t_9680, t_10820
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_9685, t_10825
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_9689, t_10829

        self.core_attention = build_module(                                    # trace_info : t_9693, t_9699, t_10833, t_10839
            submodules.core_attention,                                         # trace_info : t_9694, t_10834
            config=self.config,                                                # trace_info : t_9695, t_10835
            layer_number=self.layer_number,                                    # trace_info : t_9696, t_10836
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_9697, t_10837
            attention_type=self.attention_type,                                # trace_info : t_9698, t_10838
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'# trace_info : t_9773, t_10913

        # Output.
        self.linear_proj = build_module(                                       # trace_info : t_9774, t_9785, t_10914, t_10925
            submodules.linear_proj,                                            # trace_info : t_9775, t_10915
            self.query_projection_size,                                        # trace_info : t_9776, t_10916
            self.config.hidden_size,                                           # trace_info : t_9777, t_10917
            config=self.config,                                                # trace_info : t_9778, t_10918
            init_method=self.config.output_layer_init_method,                  # trace_info : t_9779, t_10919
            bias=self.config.add_bias_linear,                                  # trace_info : t_9780, t_10920
            input_is_parallel=True,                                            # trace_info : t_9781, t_10921
            skip_bias_add=True,                                                # trace_info : t_9782, t_10922
            is_expert=False,                                                   # trace_info : t_9783, t_10923
            tp_comm_buffer_name='proj',                                        # trace_info : t_9784, t_10924
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
        attn_mask_type = self.attn_mask_type                                   # trace_info : t_18478, t_19239, t_22829, t_23584, t_27174, ...
        if inference_params is None:                                           # trace_info : t_18479, t_19240, t_22830, t_23585, t_27175, ...
            return key, value, rotary_pos_emb, attn_mask_type                  # trace_info : t_18480, t_19241, t_22831, t_23586, t_27176, ...

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
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):# trace_info : t_18369, t_19132, t_22722, t_23477, t_27067, ...
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)# trace_info : t_18370, t_19133, t_22723, t_23478, t_27068, ...

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(# trace_info : t_18475, t_18477, t_19236, t_19238, t_22826, ...
            inference_params, key, value, rotary_pos_emb                       # trace_info : t_18476, t_19237, t_22827, t_23582, t_27172, ...
        )

        if packed_seq_params is not None:                                      # trace_info : t_18481, t_19242, t_22832, t_23587, t_27177, ...
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:                                         # trace_info : t_18482, t_19243, t_22833, t_23588, t_27178, ...
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

        if self.checkpoint_core_attention and self.training:                   # trace_info : t_18483, t_19244, t_22834, t_23589, t_27179, ...
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(                               # trace_info : t_18484, t_18491, t_19245, t_19252, t_22835, ...
                query,                                                         # trace_info : t_18485, t_19246, t_22836, t_23591, t_27181, ...
                key,                                                           # trace_info : t_18486, t_19247, t_22837, t_23592, t_27182, ...
                value,                                                         # trace_info : t_18487, t_19248, t_22838, t_23593, t_27183, ...
                attention_mask,                                                # trace_info : t_18488, t_19249, t_22839, t_23594, t_27184, ...
                attn_mask_type=attn_mask_type,                                 # trace_info : t_18489, t_19250, t_22840, t_23595, t_27185, ...
                packed_seq_params=packed_seq_params,                           # trace_info : t_18490, t_19251, t_22841, t_23596, t_27186, ...
            )

        if packed_seq_params is not None:                                      # trace_info : t_18556, t_19314, t_22904, t_23659, t_27249, ...
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)                         # trace_info : t_18557, t_19315, t_22905, t_23660, t_27250, ...

        return output, bias                                                    # trace_info : t_18624, t_19382, t_22972, t_23727, t_27317, ...


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
        super().__init__(                                                      # trace_info : t_9657, t_9663, t_10797, t_10803
            config=config,                                                     # trace_info : t_9658, t_10798
            submodules=submodules,                                             # trace_info : t_9659, t_10799
            layer_number=layer_number,                                         # trace_info : t_9660, t_10800
            attn_mask_type=attn_mask_type,                                     # trace_info : t_9661, t_10801
            attention_type="self",                                             # trace_info : t_9662, t_10802
        )

        self.linear_qkv = build_module(                                        # trace_info : t_9915, t_9926, t_11055, t_11066
            submodules.linear_qkv,                                             # trace_info : t_9916, t_11056
            self.config.hidden_size,                                           # trace_info : t_9917, t_11057
            self.query_projection_size + 2 * self.kv_projection_size,          # trace_info : t_9918, t_11058
            config=self.config,                                                # trace_info : t_9919, t_11059
            init_method=self.config.init_method,                               # trace_info : t_9920, t_11060
            gather_output=False,                                               # trace_info : t_9921, t_11061
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,      # trace_info : t_9922, t_11062
            skip_bias_add=False,                                               # trace_info : t_9923, t_11063
            is_expert=False,                                                   # trace_info : t_9924, t_11064
            tp_comm_buffer_name='qkv',                                         # trace_info : t_9925, t_11065
        )

        if submodules.q_layernorm is not None:                                 # trace_info : t_10073, t_11213
            self.q_layernorm = build_module(                                   # trace_info : t_10074, t_10079, t_11214, t_11219
                submodules.q_layernorm,                                        # trace_info : t_10075, t_11215
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10076, t_11216
                config=self.config,                                            # trace_info : t_10077, t_11217
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10078, t_11218
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:                                 # trace_info : t_10095, t_11235
            self.k_layernorm = build_module(                                   # trace_info : t_10096, t_10101, t_11236, t_11241
                submodules.k_layernorm,                                        # trace_info : t_10097, t_11237
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10098, t_11238
                config=self.config,                                            # trace_info : t_10099, t_11239
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10100, t_11240
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
        mixed_qkv, _ = self.linear_qkv(hidden_states)                          # trace_info : t_18371, t_19134, t_22724, t_23479, t_27069, ...

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (                           # trace_info : t_18449, t_18454, t_19210, t_19215, t_22800, ...
            self.num_query_groups_per_partition,                               # trace_info : t_18450, t_19211, t_22801, t_23556, t_27146, ...
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)# trace_info : t_18451, t_18453, t_19212, t_19214, t_22802, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18452, t_19213, t_22803, t_23558, t_27148, ...
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)                          # trace_info : t_18455, t_19216, t_22806, t_23561, t_27151, ...

        split_arg_list = [                                                     # trace_info : t_18463, t_19224, t_22814, t_23569, t_27159, ...
            (
                self.num_attention_heads_per_partition                         # trace_info : t_18456, t_18458, t_18460, t_19217, t_19219, ...
                // self.num_query_groups_per_partition                         # trace_info : t_18457, t_19218, t_22808, t_23563, t_27153, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18459, t_19220, t_22810, t_23565, t_27155, ...
            ),
            self.hidden_size_per_attention_head,                               # trace_info : t_18461, t_19222, t_22812, t_23567, t_27157, ...
            self.hidden_size_per_attention_head,                               # trace_info : t_18462, t_19223, t_22813, t_23568, t_27158, ...
        ]

        if SplitAlongDim is not None:                                          # trace_info : t_18464, t_19225, t_22815, t_23570, t_27160, ...

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,) # trace_info : t_18465, t_19226, t_22816, t_23571, t_27161, ...
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)# trace_info : t_18466, t_19227, t_22817, t_23572, t_27162, ...

        if self.q_layernorm is not None:                                       # trace_info : t_18467, t_19228, t_22818, t_23573, t_27163, ...
            query = self.q_layernorm(query)                                    # trace_info : t_18468, t_19229, t_22819, t_23574, t_27164, ...

        if self.k_layernorm is not None:                                       # trace_info : t_18470, t_19231, t_22821, t_23576, t_27166, ...
            key = self.k_layernorm(key)                                        # trace_info : t_18471, t_19232, t_22822, t_23577, t_27167, ...

        if self.config.test_mode:                                              # trace_info : t_18473, t_19234, t_22824, t_23579, t_27169, ...
            self.run_realtime_tests()

        return query, key, value                                               # trace_info : t_18474, t_19235, t_22825, t_23580, t_27170, ...


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
