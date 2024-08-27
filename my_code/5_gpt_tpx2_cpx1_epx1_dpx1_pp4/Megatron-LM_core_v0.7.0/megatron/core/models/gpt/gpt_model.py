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
        super().__init__(config=config)                                        # trace_info : t_9770

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec       # trace_info : t_9774
        self.vocab_size = vocab_size                                           # trace_info : t_9775
        self.max_sequence_length = max_sequence_length                         # trace_info : t_9776
        self.pre_process = pre_process                                         # trace_info : t_9777
        self.post_process = post_process                                       # trace_info : t_9778
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy                     # trace_info : t_9779
        self.parallel_output = parallel_output                                 # trace_info : t_9780
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights# trace_info : t_9781
        self.position_embedding_type = position_embedding_type                 # trace_info : t_9782

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder                         # trace_info : t_9783

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length                     # trace_info : t_9784
        self.rotary_percent = rotary_percent                                   # trace_info : t_9785

        if self.pre_process:                                                   # trace_info : t_9786
            self.embedding = LanguageModelEmbedding(                           # trace_info : t_9787, t_9792
                config=self.config,                                            # trace_info : t_9788
                vocab_size=self.vocab_size,                                    # trace_info : t_9789
                max_sequence_length=self.max_sequence_length,                  # trace_info : t_9790
                position_embedding_type=position_embedding_type,               # trace_info : t_9791
            )

        if self.position_embedding_type == 'rope':                             # trace_info : t_9908
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Transformer.
        self.decoder = TransformerBlock(                                       # trace_info : t_9909, t_9914
            config=self.config,                                                # trace_info : t_9910
            spec=transformer_layer_spec,                                       # trace_info : t_9911
            pre_process=self.pre_process,                                      # trace_info : t_9912
            post_process=self.post_process,                                    # trace_info : t_9913
        )

        # Output
        if post_process:                                                       # trace_info : t_11968
            if self.config.defer_embedding_wgrad_compute:
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
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:                              # trace_info : t_11969
            self.setup_embeddings_and_output_layer()                           # trace_info : t_11970

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):                                 # trace_info : t_18163, t_21893, t_25621
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'# trace_info : t_18164, t_21894, t_25622
        self.decoder.set_input_tensor(input_tensor[0])                         # trace_info : t_18165, t_21895, t_25623

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
        if decoder_input is not None:                                          # trace_info : t_18339, t_22069, t_25797
            pass
        elif self.pre_process:                                                 # trace_info : t_18340, t_22070, t_25798
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)# trace_info : t_18341, t_22071, t_25799
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None                                                  # trace_info : t_18375, t_22105, t_25833
        if self.position_embedding_type == 'rope':                             # trace_info : t_18376, t_22106, t_25834
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(                                          # trace_info : t_18377, t_18383, t_18385, t_22107, t_22113, ...
            hidden_states=decoder_input,                                       # trace_info : t_18378, t_22108, t_25836
            attention_mask=attention_mask,                                     # trace_info : t_18379, t_22109, t_25837
            inference_params=inference_params,                                 # trace_info : t_18380, t_22110, t_25838
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_18381, t_22111, t_25839
            packed_seq_params=packed_seq_params,                               # trace_info : t_18382, t_22112, t_25840
            **(extra_block_kwargs or {}),                                      # trace_info : t_18384, t_22114, t_25842
        )

        if not self.post_process:                                              # trace_info : t_19390, t_23118, t_26846
            return hidden_states                                               # trace_info : t_19391, t_23119, t_26847

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss

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
