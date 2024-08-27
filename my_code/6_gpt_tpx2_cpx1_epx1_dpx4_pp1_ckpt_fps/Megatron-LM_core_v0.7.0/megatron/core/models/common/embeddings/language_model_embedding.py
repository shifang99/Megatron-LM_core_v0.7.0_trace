# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class LanguageModelEmbedding(MegatronModule):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head . Defaults to 0.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        num_tokentypes: int = 0,
    ):
        super().__init__(config=config)                                        # trace_info : t_9456

        self.config: TransformerConfig = config                                # trace_info : t_9459
        self.vocab_size: int = vocab_size                                      # trace_info : t_9460
        self.max_sequence_length: int = max_sequence_length                    # trace_info : t_9461
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'# trace_info : t_9462
        self.num_tokentypes = num_tokentypes                                   # trace_info : t_9463

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(         # trace_info : t_9464, t_9469
            num_embeddings=self.vocab_size,                                    # trace_info : t_9465
            embedding_dim=self.config.hidden_size,                             # trace_info : t_9466
            init_method=self.config.init_method,                               # trace_info : t_9467
            config=self.config,                                                # trace_info : t_9468
        )

        # Position embedding (serial).
        if self.add_position_embedding:                                        # trace_info : t_9561
            self.position_embeddings = torch.nn.Embedding(                     # trace_info : t_9562, t_9564
                self.max_sequence_length, self.config.hidden_size              # trace_info : t_9563
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:                             # trace_info : t_9565
                self.config.init_method(self.position_embeddings.weight)       # trace_info : t_9566

        if self.num_tokentypes > 0:                                            # trace_info : t_9568
            self.tokentype_embeddings = torch.nn.Embedding(
                self.num_tokentypes, self.config.hidden_size
            )
            # Initialize the token-type embeddings.
            if self.config.perform_initialization:
                self.config.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None                                   # trace_info : t_9569

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)  # trace_info : t_9570

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)                      # trace_info : t_18355, t_21994, t_89601
        if self.add_position_embedding:                                        # trace_info : t_18378, t_22017, t_89624
            position_embeddings = self.position_embeddings(position_ids)       # trace_info : t_18379, t_22018, t_89625
            embeddings = word_embeddings + position_embeddings                 # trace_info : t_18380, t_22019, t_89626
        else:
            embeddings = word_embeddings

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()                   # trace_info : t_18381, t_22020, t_89627

        if tokentype_ids is not None:                                          # trace_info : t_18382, t_22021, t_89628
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None                           # trace_info : t_18383, t_22022, t_89629

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:                               # trace_info : t_18384, t_22023, t_89630
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:                                      # trace_info : t_18385, t_22024, t_89631
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)                    # trace_info : t_18386, t_22025, t_89632

        return embeddings                                                      # trace_info : t_18387, t_22026, t_89633
