# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


class GPTModel(LanguageModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)                                        # trace_info : t_6404

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec       # trace_info : t_6408
        self.vocab_size = vocab_size                                           # trace_info : t_6409
        self.max_sequence_length = max_sequence_length                         # trace_info : t_6410
        self.pre_process = pre_process                                         # trace_info : t_6411
        self.post_process = post_process                                       # trace_info : t_6412
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy                     # trace_info : t_6413
        self.parallel_output = parallel_output                                 # trace_info : t_6414
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights# trace_info : t_6415
        self.position_embedding_type = position_embedding_type                 # trace_info : t_6416

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder                         # trace_info : t_6417

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length                     # trace_info : t_6418
        self.rotary_percent = rotary_percent                                   # trace_info : t_6419

        if self.pre_process:                                                   # trace_info : t_6420
            self.embedding = LanguageModelEmbedding(                           # trace_info : t_6421, t_6426
                config=self.config,                                            # trace_info : t_6422
                vocab_size=self.vocab_size,                                    # trace_info : t_6423
                max_sequence_length=self.max_sequence_length,                  # trace_info : t_6424
                position_embedding_type=position_embedding_type,               # trace_info : t_6425
            )

        if self.position_embedding_type == 'rope':                             # trace_info : t_6542
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Transformer.
        self.decoder = TransformerBlock(                                       # trace_info : t_6543, t_6548
            config=self.config,                                                # trace_info : t_6544
            spec=transformer_layer_spec,                                       # trace_info : t_6545
            pre_process=self.pre_process,                                      # trace_info : t_6546
            post_process=self.post_process,                                    # trace_info : t_6547
        )

        # Output
        if post_process:                                                       # trace_info : t_8623
            if self.config.defer_embedding_wgrad_compute:                      # trace_info : t_8624
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None                        # trace_info : t_8625
                self.grad_output_buffer = None                                 # trace_info : t_8626

            self.output_layer = tensor_parallel.ColumnParallelLinear(          # trace_info : t_8627, t_8639
                config.hidden_size,                                            # trace_info : t_8628
                self.vocab_size,                                               # trace_info : t_8629
                config=config,                                                 # trace_info : t_8630
                init_method=config.init_method,                                # trace_info : t_8631
                bias=False,                                                    # trace_info : t_8632
                skip_bias_add=False,                                           # trace_info : t_8633
                gather_output=not self.parallel_output,                        # trace_info : t_8634
                skip_weight_param_allocation=self.pre_process                  # trace_info : t_8635
                and self.share_embeddings_and_output_weights,                  # trace_info : t_8636
                embedding_activation_buffer=self.embedding_activation_buffer,  # trace_info : t_8637
                grad_output_buffer=self.grad_output_buffer,                    # trace_info : t_8638
            )

        if self.pre_process or self.post_process:                              # trace_info : t_8683
            self.setup_embeddings_and_output_layer()                           # trace_info : t_8684

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):                                 # trace_info : t_15024, t_18665, t_22304
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'# trace_info : t_15025, t_18666, t_22305
        self.decoder.set_input_tensor(input_tensor[0])                         # trace_info : t_15026, t_18667, t_22306

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:                                          # trace_info : t_15207, t_18848, t_22487
            pass
        elif self.pre_process:                                                 # trace_info : t_15208, t_18849, t_22488
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)# trace_info : t_15209, t_18850, t_22489
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None                                                  # trace_info : t_15236, t_18877, t_22516
        if self.position_embedding_type == 'rope':                             # trace_info : t_15237, t_18878, t_22517
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(                                          # trace_info : t_15238, t_15244, t_15246, t_18879, t_18885, ...
            hidden_states=decoder_input,                                       # trace_info : t_15239, t_18880, t_22519
            attention_mask=attention_mask,                                     # trace_info : t_15240, t_18881, t_22520
            inference_params=inference_params,                                 # trace_info : t_15241, t_18882, t_22521
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_15242, t_18883, t_22522
            packed_seq_params=packed_seq_params,                               # trace_info : t_15243, t_18884, t_22523
            **(extra_block_kwargs or {}),                                      # trace_info : t_15245, t_18886, t_22525
        )

        if not self.post_process:                                              # trace_info : t_16268, t_19907, t_23546
            return hidden_states

        # logits and loss
        output_weight = None                                                   # trace_info : t_16269, t_19908, t_23547
        if self.share_embeddings_and_output_weights:                           # trace_info : t_16270, t_19909, t_23548
            output_weight = self.shared_embedding_or_output_weight()           # trace_info : t_16271, t_19910, t_23549
        logits, _ = self.output_layer(hidden_states, weight=output_weight)     # trace_info : t_16274, t_19913, t_23552

        if labels is None:                                                     # trace_info : t_16334, t_19973, t_23612
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)                # trace_info : t_16335, t_19974, t_23613

        return loss                                                            # trace_info : t_16403, t_20042, t_23681

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """ Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the _extra_state key
        # but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        return sharded_state_dict
