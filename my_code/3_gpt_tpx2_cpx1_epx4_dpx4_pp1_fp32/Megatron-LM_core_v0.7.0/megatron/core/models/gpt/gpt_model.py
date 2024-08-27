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
        super().__init__(config=config)                                        # trace_info : t_9358

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec       # trace_info : t_9362
        self.vocab_size = vocab_size                                           # trace_info : t_9363
        self.max_sequence_length = max_sequence_length                         # trace_info : t_9364
        self.pre_process = pre_process                                         # trace_info : t_9365
        self.post_process = post_process                                       # trace_info : t_9366
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy                     # trace_info : t_9367
        self.parallel_output = parallel_output                                 # trace_info : t_9368
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights# trace_info : t_9369
        self.position_embedding_type = position_embedding_type                 # trace_info : t_9370

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder                         # trace_info : t_9371

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length                     # trace_info : t_9372
        self.rotary_percent = rotary_percent                                   # trace_info : t_9373

        if self.pre_process:                                                   # trace_info : t_9374
            self.embedding = LanguageModelEmbedding(                           # trace_info : t_9375, t_9380
                config=self.config,                                            # trace_info : t_9376
                vocab_size=self.vocab_size,                                    # trace_info : t_9377
                max_sequence_length=self.max_sequence_length,                  # trace_info : t_9378
                position_embedding_type=position_embedding_type,               # trace_info : t_9379
            )

        if self.position_embedding_type == 'rope':                             # trace_info : t_9496
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Transformer.
        self.decoder = TransformerBlock(                                       # trace_info : t_9497, t_9502
            config=self.config,                                                # trace_info : t_9498
            spec=transformer_layer_spec,                                       # trace_info : t_9499
            pre_process=self.pre_process,                                      # trace_info : t_9500
            post_process=self.post_process,                                    # trace_info : t_9501
        )

        # Output
        if post_process:                                                       # trace_info : t_11853
            if self.config.defer_embedding_wgrad_compute:                      # trace_info : t_11854
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
                self.embedding_activation_buffer = None                        # trace_info : t_11855
                self.grad_output_buffer = None                                 # trace_info : t_11856

            self.output_layer = tensor_parallel.ColumnParallelLinear(          # trace_info : t_11857, t_11869
                config.hidden_size,                                            # trace_info : t_11858
                self.vocab_size,                                               # trace_info : t_11859
                config=config,                                                 # trace_info : t_11860
                init_method=config.init_method,                                # trace_info : t_11861
                bias=False,                                                    # trace_info : t_11862
                skip_bias_add=False,                                           # trace_info : t_11863
                gather_output=not self.parallel_output,                        # trace_info : t_11864
                skip_weight_param_allocation=self.pre_process                  # trace_info : t_11865
                and self.share_embeddings_and_output_weights,                  # trace_info : t_11866
                embedding_activation_buffer=self.embedding_activation_buffer,  # trace_info : t_11867
                grad_output_buffer=self.grad_output_buffer,                    # trace_info : t_11868
            )

        if self.pre_process or self.post_process:                              # trace_info : t_11913
            self.setup_embeddings_and_output_layer()                           # trace_info : t_11914

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):                                 # trace_info : t_18058, t_22411, t_26756
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'# trace_info : t_18059, t_22412, t_26757
        self.decoder.set_input_tensor(input_tensor[0])                         # trace_info : t_18060, t_22413, t_26758

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
        if decoder_input is not None:                                          # trace_info : t_18211, t_22564, t_26909
            pass
        elif self.pre_process:                                                 # trace_info : t_18212, t_22565, t_26910
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)# trace_info : t_18213, t_22566, t_26911
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None                                                  # trace_info : t_18304, t_22657, t_27002
        if self.position_embedding_type == 'rope':                             # trace_info : t_18305, t_22658, t_27003
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(                                          # trace_info : t_18306, t_18312, t_18314, t_22659, t_22665, ...
            hidden_states=decoder_input,                                       # trace_info : t_18307, t_22660, t_27005
            attention_mask=attention_mask,                                     # trace_info : t_18308, t_22661, t_27006
            inference_params=inference_params,                                 # trace_info : t_18309, t_22662, t_27007
            rotary_pos_emb=rotary_pos_emb,                                     # trace_info : t_18310, t_22663, t_27008
            packed_seq_params=packed_seq_params,                               # trace_info : t_18311, t_22664, t_27009
            **(extra_block_kwargs or {}),                                      # trace_info : t_18313, t_22666, t_27011
        )

        if not self.post_process:                                              # trace_info : t_19854, t_24199, t_28544
            return hidden_states

        # logits and loss
        output_weight = None                                                   # trace_info : t_19855, t_24200, t_28545
        if self.share_embeddings_and_output_weights:                           # trace_info : t_19856, t_24201, t_28546
            output_weight = self.shared_embedding_or_output_weight()           # trace_info : t_19857, t_24202, t_28547
        logits, _ = self.output_layer(hidden_states, weight=output_weight)     # trace_info : t_19860, t_24205, t_28550

        if labels is None:                                                     # trace_info : t_19935, t_24280, t_28625
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)                # trace_info : t_19936, t_24281, t_28626

        return loss                                                            # trace_info : t_20004, t_24349, t_28694

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
