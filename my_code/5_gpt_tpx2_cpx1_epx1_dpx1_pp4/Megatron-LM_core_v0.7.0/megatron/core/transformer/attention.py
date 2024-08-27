# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass                                              # trace_info : t_9725
from importlib.metadata import version                                         # trace_info : t_9726
from typing import Union                                                       # trace_info : t_9727
                                                                               # trace_info : t_9728
import torch                                                                   # trace_info : t_9729
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
        super().__init__(config=config)                                        # trace_info : t_10076, t_11078

        self.config = config                                                   # trace_info : t_10079, t_11081
        self.layer_number = layer_number                                       # trace_info : t_10080, t_11082
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_10081, t_11083
        self.attention_type = attention_type                                   # trace_info : t_10082, t_11084

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads# trace_info : t_10083, t_11085
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups# trace_info : t_10084, t_11086

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()     # trace_info : t_10085, t_11087
        self.hidden_size_per_attention_head = divide(                          # trace_info : t_10091, t_10093, t_11093, t_11095
            self.query_projection_size, self.config.num_attention_heads        # trace_info : t_10092, t_11094
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)# trace_info : t_10097, t_11099
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)# trace_info : t_10101, t_11103

        self.core_attention = build_module(                                    # trace_info : t_10105, t_10111, t_11107, t_11113
            submodules.core_attention,                                         # trace_info : t_10106, t_11108
            config=self.config,                                                # trace_info : t_10107, t_11109
            layer_number=self.layer_number,                                    # trace_info : t_10108, t_11110
            attn_mask_type=self.attn_mask_type,                                # trace_info : t_10109, t_11111
            attention_type=self.attention_type,                                # trace_info : t_10110, t_11112
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'# trace_info : t_10187, t_11189

        # Output.
        self.linear_proj = build_module(                                       # trace_info : t_10188, t_10199, t_11190, t_11201
            submodules.linear_proj,                                            # trace_info : t_10189, t_11191
            self.query_projection_size,                                        # trace_info : t_10190, t_11192
            self.config.hidden_size,                                           # trace_info : t_10191, t_11193
            config=self.config,                                                # trace_info : t_10192, t_11194
            init_method=self.config.output_layer_init_method,                  # trace_info : t_10193, t_11195
            bias=self.config.add_bias_linear,                                  # trace_info : t_10194, t_11196
            input_is_parallel=True,                                            # trace_info : t_10195, t_11197
            skip_bias_add=True,                                                # trace_info : t_10196, t_11198
            is_expert=False,                                                   # trace_info : t_10197, t_11199
            tp_comm_buffer_name='proj',                                        # trace_info : t_10198, t_11200
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
        attn_mask_type = self.attn_mask_type                                   # trace_info : t_18520, t_19015, t_22250, t_22743, t_25978, ...
        if inference_params is None:                                           # trace_info : t_18521, t_19016, t_22251, t_22744, t_25979, ...
            return key, value, rotary_pos_emb, attn_mask_type                  # trace_info : t_18522, t_19017, t_22252, t_22745, t_25980, ...

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
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):# trace_info : t_18436, t_18931, t_22166, t_22659, t_25894, ...
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)# trace_info : t_18437, t_18932, t_22167, t_22660, t_25895, ...

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(# trace_info : t_18517, t_18519, t_19012, t_19014, t_22247, ...
            inference_params, key, value, rotary_pos_emb                       # trace_info : t_18518, t_19013, t_22248, t_22741, t_25976, ...
        )

        if packed_seq_params is not None:                                      # trace_info : t_18523, t_19018, t_22253, t_22746, t_25981, ...
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:                                         # trace_info : t_18524, t_19019, t_22254, t_22747, t_25982, ...
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

        if self.checkpoint_core_attention and self.training:                   # trace_info : t_18525, t_19020, t_22255, t_22748, t_25983, ...
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(                               # trace_info : t_18526, t_18533, t_19021, t_19028, t_22256, ...
                query,                                                         # trace_info : t_18527, t_19022, t_22257, t_22750, t_25985, ...
                key,                                                           # trace_info : t_18528, t_19023, t_22258, t_22751, t_25986, ...
                value,                                                         # trace_info : t_18529, t_19024, t_22259, t_22752, t_25987, ...
                attention_mask,                                                # trace_info : t_18530, t_19025, t_22260, t_22753, t_25988, ...
                attn_mask_type=attn_mask_type,                                 # trace_info : t_18531, t_19026, t_22261, t_22754, t_25989, ...
                packed_seq_params=packed_seq_params,                           # trace_info : t_18532, t_19027, t_22262, t_22755, t_25990, ...
            )

        if packed_seq_params is not None:                                      # trace_info : t_18642, t_19135, t_22370, t_22863, t_26098, ...
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)                         # trace_info : t_18643, t_19136, t_22371, t_22864, t_26099, ...

        return output, bias                                                    # trace_info : t_18703, t_19196, t_22431, t_22924, t_26159, ...


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
        super().__init__(                                                      # trace_info : t_10069, t_10075, t_11071, t_11077
            config=config,                                                     # trace_info : t_10070, t_11072
            submodules=submodules,                                             # trace_info : t_10071, t_11073
            layer_number=layer_number,                                         # trace_info : t_10072, t_11074
            attn_mask_type=attn_mask_type,                                     # trace_info : t_10073, t_11075
            attention_type="self",                                             # trace_info : t_10074, t_11076
        )

        self.linear_qkv = build_module(                                        # trace_info : t_10329, t_10340, t_11331, t_11342
            submodules.linear_qkv,                                             # trace_info : t_10330, t_11332
            self.config.hidden_size,                                           # trace_info : t_10331, t_11333
            self.query_projection_size + 2 * self.kv_projection_size,          # trace_info : t_10332, t_11334
            config=self.config,                                                # trace_info : t_10333, t_11335
            init_method=self.config.init_method,                               # trace_info : t_10334, t_11336
            gather_output=False,                                               # trace_info : t_10335, t_11337
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,      # trace_info : t_10336, t_11338
            skip_bias_add=False,                                               # trace_info : t_10337, t_11339
            is_expert=False,                                                   # trace_info : t_10338, t_11340
            tp_comm_buffer_name='qkv',                                         # trace_info : t_10339, t_11341
        )

        if submodules.q_layernorm is not None:                                 # trace_info : t_10487, t_11489
            self.q_layernorm = build_module(                                   # trace_info : t_10488, t_10493, t_11490, t_11495
                submodules.q_layernorm,                                        # trace_info : t_10489, t_11491
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10490, t_11492
                config=self.config,                                            # trace_info : t_10491, t_11493
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10492, t_11494
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:                                 # trace_info : t_10509, t_11511
            self.k_layernorm = build_module(                                   # trace_info : t_10510, t_10515, t_11512, t_11517
                submodules.k_layernorm,                                        # trace_info : t_10511, t_11513
                hidden_size=self.hidden_size_per_attention_head,               # trace_info : t_10512, t_11514
                config=self.config,                                            # trace_info : t_10513, t_11515
                eps=self.config.layernorm_epsilon,                             # trace_info : t_10514, t_11516
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
        mixed_qkv, _ = self.linear_qkv(hidden_states)                          # trace_info : t_18438, t_18933, t_22168, t_22661, t_25896, ...

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (                           # trace_info : t_18491, t_18496, t_18986, t_18991, t_22221, ...
            self.num_query_groups_per_partition,                               # trace_info : t_18492, t_18987, t_22222, t_22715, t_25950, ...
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)# trace_info : t_18493, t_18495, t_18988, t_18990, t_22223, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18494, t_18989, t_22224, t_22717, t_25952, ...
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)                          # trace_info : t_18497, t_18992, t_22227, t_22720, t_25955, ...

        split_arg_list = [                                                     # trace_info : t_18505, t_19000, t_22235, t_22728, t_25963, ...
            (
                self.num_attention_heads_per_partition                         # trace_info : t_18498, t_18500, t_18502, t_18993, t_18995, ...
                // self.num_query_groups_per_partition                         # trace_info : t_18499, t_18994, t_22229, t_22722, t_25957, ...
                * self.hidden_size_per_attention_head                          # trace_info : t_18501, t_18996, t_22231, t_22724, t_25959, ...
            ),
            self.hidden_size_per_attention_head,                               # trace_info : t_18503, t_18998, t_22233, t_22726, t_25961, ...
            self.hidden_size_per_attention_head,                               # trace_info : t_18504, t_18999, t_22234, t_22727, t_25962, ...
        ]

        if SplitAlongDim is not None:                                          # trace_info : t_18506, t_19001, t_22236, t_22729, t_25964, ...

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,) # trace_info : t_18507, t_19002, t_22237, t_22730, t_25965, ...
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)# trace_info : t_18508, t_19003, t_22238, t_22731, t_25966, ...

        if self.q_layernorm is not None:                                       # trace_info : t_18509, t_19004, t_22239, t_22732, t_25967, ...
            query = self.q_layernorm(query)                                    # trace_info : t_18510, t_19005, t_22240, t_22733, t_25968, ...

        if self.k_layernorm is not None:                                       # trace_info : t_18512, t_19007, t_22242, t_22735, t_25970, ...
            key = self.k_layernorm(key)                                        # trace_info : t_18513, t_19008, t_22243, t_22736, t_25971, ...

        if self.config.test_mode:                                              # trace_info : t_18515, t_19010, t_22245, t_22738, t_25973, ...
            self.run_realtime_tests()

        return query, key, value                                               # trace_info : t_18516, t_19011, t_22246, t_22739, t_25974, ...


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
