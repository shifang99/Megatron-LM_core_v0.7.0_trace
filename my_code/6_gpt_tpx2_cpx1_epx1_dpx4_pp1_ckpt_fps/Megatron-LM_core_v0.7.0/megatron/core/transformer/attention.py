# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass                                              # trace_info : t_9388
from importlib.metadata import version                                         # trace_info : t_9389
from typing import Union                                                       # trace_info : t_9390
                                                                               # trace_info : t_9391
import torch                                                                   # trace_info : t_9392
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
        super().__init__(config=config)                                        # trace_info : t_9739, t_10741

        self.config = config                                                   # trace_info : t_9742, t_10744
        self.layer_number = layer_number                                       # trace_info : t_9743, t_10745
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_9744, t_10746
        self.attention_type = attention_type                                   # trace_info : t_9745, t_10747

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_9746, t_10748
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups# trace_info : t_9747, t_10749

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_9748, t_10750
        self.hidden_size_per_attention_head = divide(                          # trace_info : t_9754, t_9756, t_10756, t_10758
            self.query_projection_size, self.config.num_attention_heads        # trace_info : t_9755, t_10757
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_9760, t_10762
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_9764, t_10766

        self.core_attention = build_module(                                    # trace_info : t_9768, t_9774, t_10770, t_10776
            submodules.core_attention,                                         # trace_info : t_9769, t_10771
            config=self.config,                                                # trace_info : t_9770, t_10772
            layer_number=self.layer_number,                                    # trace_info : t_9771, t_10773
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_9772, t_10774
            attention_type=self.attention_type,                                # trace_info : t_9773, t_10775
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'# trace_info : t_9850, t_10852

        # Output.
        self.linear_proj = build_module(                                       # trace_info : t_9851, t_9862, t_10853, t_10864
            submodules.linear_proj,                                            # trace_info : t_9852, t_10854
            self.query_projection_size,                                        # trace_info : t_9853, t_10855
            self.config.hidden_size,                                           # trace_info : t_9854, t_10856
            config=self.config,                                                # trace_info : t_9855, t_10857
            init_method=self.config.output_layer_init_method,                  # trace_info : t_9856, t_10858
            bias=self.config.add_bias_linear,                                  # trace_info : t_9857, t_10859
            input_is_parallel=True,                                            # trace_info : t_9858, t_10860
            skip_bias_add=True,                                                # trace_info : t_9859, t_10861
            is_expert=False,                                                   # trace_info : t_9860, t_10862
            tp_comm_buffer_name='proj',                                        # trace_info : t_9861, t_10863
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
        attn_mask_type = self.attn_mask_type                                   # trace_info : t_18533, t_19028, t_22172, t_22665, t_89779, ...
        if inference_params is None:                                           # trace_info : t_18534, t_19029, t_22173, t_22666, t_89780, ...
            return key, value, rotary_pos_emb, attn_mask_type                  # trace_info : t_18535, t_19030, t_22174, t_22667, t_89781, ...

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
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):# trace_info : t_18449, t_18944, t_22088, t_22581, t_89695, ...
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)# trace_info : t_18450, t_18945, t_22089, t_22582, t_89696, ...

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(# trace_info : t_18530, t_18532, t_19025, t_19027, t_22169, ...
            inference_params, key, value, rotary_pos_emb                       # trace_info : t_18531, t_19026, t_22170, t_22663, t_89777, ...
        )

        if packed_seq_params is not None:                                      # trace_info : t_18536, t_19031, t_22175, t_22668, t_89782, ...
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:                                         # trace_info : t_18537, t_19032, t_22176, t_22669, t_89783, ...
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

        if self.checkpoint_core_attention and self.training:                   # trace_info : t_18538, t_19033, t_22177, t_22670, t_89784, ...
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(                               # trace_info : t_18539, t_18546, t_19034, t_19041, t_22178, ...
                query,                                                         # trace_info : t_18540, t_19035, t_22179, t_22672, t_89786, ...
                key,                                                           # trace_info : t_18541, t_19036, t_22180, t_22673, t_89787, ...
                value,                                                         # trace_info : t_18542, t_19037, t_22181, t_22674, t_89788, ...
                attention_mask,                                                # trace_info : t_18543, t_19038, t_22182, t_22675, t_89789, ...
                attn_mask_type=attn_mask_type,                                 # trace_info : t_18544, t_19039, t_22183, t_22676, t_89790, ...
                packed_seq_params=packed_seq_params,                           # trace_info : t_18545, t_19040, t_22184, t_22677, t_89791, ...
            )

        if packed_seq_params is not None:                                      # trace_info : t_18655, t_19148, t_22292, t_22785, t_89899, ...
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)                         # trace_info : t_18656, t_19149, t_22293, t_22786, t_89900, ...

        return output, bias                                                    # trace_info : t_18716, t_19209, t_22353, t_22846, t_89960, ...


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
        super().__init__(                                                      # trace_info : t_9732, t_9738, t_10734, t_10740
            config=config,                                                     # trace_info : t_9733, t_10735
            submodules=submodules,                                             # trace_info : t_9734, t_10736
            layer_number=layer_number,                                         # trace_info : t_9735, t_10737
            attn_mask_type=attn_mask_type,                                     # trace_info : t_9736, t_10738
            attention_type="self",                                             # trace_info : t_9737, t_10739
        )

        self.linear_qkv = build_module(                                        # trace_info : t_9992, t_10003, t_10994, t_11005
            submodules.linear_qkv,                                             # trace_info : t_9993, t_10995
            self.config.hidden_size,                                           # trace_info : t_9994, t_10996
            self.query_projection_size + 2 * self.kv_projection_size,          # trace_info : t_9995, t_10997
            config=self.config,                                                # trace_info : t_9996, t_10998
            init_method=self.config.init_method,                               # trace_info : t_9997, t_10999
            gather_output=False,                                               # trace_info : t_9998, t_11000
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,      # trace_info : t_9999, t_11001
            skip_bias_add=False,                                               # trace_info : t_10000, t_11002
            is_expert=False,                                                   # trace_info : t_10001, t_11003
            tp_comm_buffer_name='qkv',                                         # trace_info : t_10002, t_11004
        )

        if submodules.q_layernorm is not None:                                 # trace_info : t_10150, t_11152
            self.q_layernorm = build_module(                                   # trace_info : t_10151, t_10156, t_11153, t_11158
                submodules.q_layernorm,                                        # trace_info : t_10152, t_11154
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10153, t_11155
                config=self.config,                                            # trace_info : t_10154, t_11156
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10155, t_11157
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:                                 # trace_info : t_10172, t_11174
            self.k_layernorm = build_module(                                   # trace_info : t_10173, t_10178, t_11175, t_11180
                submodules.k_layernorm,                                        # trace_info : t_10174, t_11176
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10175, t_11177
                config=self.config,                                            # trace_info : t_10176, t_11178
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10177, t_11179
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
        mixed_qkv, _ = self.linear_qkv(hidden_states)                          # trace_info : t_18451, t_18946, t_22090, t_22583, t_89697, ...

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (                           # trace_info : t_18504, t_18509, t_18999, t_19004, t_22143, ...
            self.num_query_groups_per_partition,                               # trace_info : t_18505, t_19000, t_22144, t_22637, t_89751, ...
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)# trace_info : t_18506, t_18508, t_19001, t_19003, t_22145, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18507, t_19002, t_22146, t_22639, t_89753, ...
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)                          # trace_info : t_18510, t_19005, t_22149, t_22642, t_89756, ...

        split_arg_list = [                                                     # trace_info : t_18518, t_19013, t_22157, t_22650, t_89764, ...
            (
                self.num_attention_heads_per_partition                         # trace_info : t_18511, t_18513, t_18515, t_19006, t_19008, ...
                // self.num_query_groups_per_partition                         # trace_info : t_18512, t_19007, t_22151, t_22644, t_89758, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18514, t_19009, t_22153, t_22646, t_89760, ...
            ),
            self.hidden_size_per_attention_head,                               # trace_info : t_18516, t_19011, t_22155, t_22648, t_89762, ...
            self.hidden_size_per_attention_head,                               # trace_info : t_18517, t_19012, t_22156, t_22649, t_89763, ...
        ]

        if SplitAlongDim is not None:                                          # trace_info : t_18519, t_19014, t_22158, t_22651, t_89765, ...

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,) # trace_info : t_18520, t_19015, t_22159, t_22652, t_89766, ...
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)# trace_info : t_18521, t_19016, t_22160, t_22653, t_89767, ...

        if self.q_layernorm is not None:                                       # trace_info : t_18522, t_19017, t_22161, t_22654, t_89768, ...
            query = self.q_layernorm(query)                                    # trace_info : t_18523, t_19018, t_22162, t_22655, t_89769, ...

        if self.k_layernorm is not None:                                       # trace_info : t_18525, t_19020, t_22164, t_22657, t_89771, ...
            key = self.k_layernorm(key)                                        # trace_info : t_18526, t_19021, t_22165, t_22658, t_89772, ...

        if self.config.test_mode:                                              # trace_info : t_18528, t_19023, t_22167, t_22660, t_89774, ...
            self.run_realtime_tests()

        return query, key, value                                               # trace_info : t_18529, t_19024, t_22168, t_22661, t_89775, ...


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
