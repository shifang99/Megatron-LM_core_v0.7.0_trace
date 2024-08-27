import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


class LanguageModule(MegatronModule):
    """Base language module that has common helper functions used across GPT, BERT etc.

    Args:
        config (TransformerConfig): Input transformer config for the model
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)                                        # trace_info : t_9434

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()                           # trace_info : t_19464, t_23101, t_90708
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels)# trace_info : t_19465, t_23102, t_90709

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()                               # trace_info : t_19529, t_23166, t_90773
        return loss                                                            # trace_info : t_19530, t_23167, t_90774

    def setup_embeddings_and_output_layer(self) -> None:
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initalizes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.
        """

        # Set `is_embedding_or_output_parameter` attribute.
        if self.pre_process:                                                   # trace_info : t_11714
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True# trace_info : t_11715
        if self.post_process and self.output_layer.weight is not None:         # trace_info : t_11716
            self.output_layer.weight.is_embedding_or_output_parameter = True

        if not self.share_embeddings_and_output_weights:                       # trace_info : t_11717
            return

        if self.pre_process and self.post_process:                             # trace_info : t_11718
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.shared_embedding_or_output_weight().zero_out_wgrad = True     # trace_info : t_11719
            return                                                             # trace_info : t_11722

        if self.pre_process and not self.post_process:
            assert parallel_state.is_pipeline_first_stage()
            self.shared_embedding_or_output_weight().shared_embedding = True

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.output_layer.weight.data.fill_(0)
            self.output_layer.weight.shared = True
            self.output_layer.weight.shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                weight = self.shared_embedding_or_output_weight()
                weight.data = weight.data.cuda()
                torch.distributed.all_reduce(
                    weight.data, group=parallel_state.get_embedding_group()
                )

        elif not getattr(LanguageModule, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            LanguageModule.embedding_warning_printed = True

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the emedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre processing it returns the input embeddings weight while during post processing it returns the final output layers weight
        """
        if self.pre_process:                                                   # trace_info : t_11720, t_19408, t_23045, t_90652
            return self.embedding.word_embeddings.weight                       # trace_info : t_11721, t_19409, t_23046, t_90653
        elif self.post_process:
            return self.output_layer.weight
        return None

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """ Sharded state dict implementation that handles the output layer weights tying.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the LanguageModel
        """
        assert not sharded_offsets, "Unexpected sharded offsets"               # trace_info : t_25054, t_92647
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)# trace_info : t_25055, t_92648

        first_stage_word_emb_key = f'{prefix}embedding.word_embeddings.weight' # trace_info : t_29083, t_96676
        output_layer_weight_key = f'{prefix}output_layer.weight'               # trace_info : t_29084, t_96677
        output_layer_bias_key = f'{prefix}output_layer.bias'                   # trace_info : t_29085, t_96678

        if self.share_embeddings_and_output_weights:                           # trace_info : t_29086, t_96679
            self.tie_embeddings_and_output_weights_state_dict(                 # trace_info : t_29087, t_29089, t_96680, t_96682
                sharded_state_dict, output_layer_weight_key, first_stage_word_emb_key# trace_info : t_29088, t_96681
            )
        elif self.post_process:
            # Make sure the output layer follows the embeddings padding logic
            sharded_state_dict[output_layer_weight_key].allow_shape_mismatch = True

        # Regardless of sharing the output weights with embeddings, we must handle the bias padding
        if self.post_process and output_layer_bias_key in sharded_state_dict:  # trace_info : t_29093, t_96686
            sharded_state_dict[output_layer_bias_key].allow_shape_mismatch = True

        return sharded_state_dict                                              # trace_info : t_29094, t_96687

    def tie_embeddings_and_output_weights_state_dict(
        self,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
    ) -> None:
        """Ties the embedding and output weights in a given sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            output_layer_weight_key (str): key of the output layer weight in the state dict.
                This entry will be replaced with a tied version
            first_stage_word_emb_key (str): this must be the same as the
                ShardedTensor.key of the first stage word embeddings.

        Returns: None, acts in-place
        """
        if not self.post_process:                                              # trace_info : t_29090, t_96683
            # No output layer
            assert output_layer_weight_key not in sharded_state_dict, sharded_state_dict.keys()
            return

        if self.pre_process:                                                   # trace_info : t_29091, t_96684
            # Output layer is equivalent to the embedding already
            return                                                             # trace_info : t_29092, t_96685

        # Replace the default output layer with a one sharing the weights with the embedding
        del sharded_state_dict[output_layer_weight_key]
        tensor = self.shared_embedding_or_output_weight()
        last_stage_word_emb_replica_id = (
            1,  # copy of first stage embedding
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=first_stage_word_emb_key,
            replica_id=last_stage_word_emb_replica_id,
            allow_shape_mismatch=True,
        )
