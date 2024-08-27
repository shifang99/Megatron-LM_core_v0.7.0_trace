# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .. import parallel_state
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
    """

    if (                                                                       # trace_info : t_20553, t_24898, t_29243
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)         # trace_info : t_20549, t_24894, t_29239
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_20554, t_24899, t_29244
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    if (                                                                       # trace_info : t_20563, t_24908, t_29253
        parallel_state.is_rank_in_position_embedding_group()                   # trace_info : t_20560, t_24905, t_29250
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_20564, t_24909, t_29254
        and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())


def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce both word and position embeddings.
    """
    _allreduce_word_embedding_grads(model, config)                             # trace_info : t_20548, t_24893, t_29238
    _allreduce_position_embedding_grads(model, config)                         # trace_info : t_20559, t_24904, t_29249


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (         # trace_info : t_20258, t_20265, t_24603, t_24610, t_28948, ...
        config.sequence_parallel or config.qk_layernorm                        # trace_info : t_20264, t_24609, t_28954
    ):
        grads = []                                                             # trace_info : t_20266, t_24611, t_28956
        for model_chunk in model:                                              # trace_info : t_20267, t_20486, t_24612, t_24831, t_28957, ...
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():# trace_info : t_20268, t_20282, t_20289, t_20296, t_20303, ...
                if (                                                           # trace_info : t_20277, t_20279, t_20284, t_20286, t_20291, ...
                    param.requires_grad                                        # trace_info : t_20276, t_20283, t_20290, t_20297, t_20304, ...
                    and getattr(param, 'sequence_parallel', False)             # trace_info : t_20278, t_20285, t_20292, t_20299, t_20306, ...
                    or 'q_layernorm' in name                                   # trace_info : t_20280, t_20287, t_20308, t_20322, t_20329, ...
                    or 'k_layernorm' in name                                   # trace_info : t_20281, t_20288, t_20309, t_20323, t_20330, ...
                ):
                    grad = param.main_grad                                     # trace_info : t_20294, t_20301, t_20315, t_20336, t_20343, ...
                    grads.append(grad.data)                                    # trace_info : t_20295, t_20302, t_20316, t_20337, t_20344, ...
        if grads:                                                              # trace_info : t_20487, t_24832, t_29177
            coalesced = _flatten_dense_tensors(grads)                          # trace_info : t_20488, t_24833, t_29178
            torch.distributed.all_reduce(                                      # trace_info : t_20489, t_20494, t_24834, t_24839, t_29179, ...
                coalesced, group=parallel_state.get_tensor_model_parallel_group()# trace_info : t_20490, t_24835, t_29180
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):# trace_info : t_20495, t_20497, t_20499, t_20501, t_20503, ...
                buf.copy_(synced)                                              # trace_info : t_20496, t_20498, t_20500, t_20502, t_20504, ...


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])                                        # trace_info : t_20173, t_24518, t_28863

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:                                              # trace_info : t_20182, t_24527, t_28872
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)# trace_info : t_20183, t_24528, t_28873
    for model_chunk in model:                                                  # trace_info : t_20190, t_20237, t_24535, t_24582, t_28880, ...
        model_chunk.finish_grad_sync()                                         # trace_info : t_20191, t_24536, t_28881
    if config.timers is not None:                                              # trace_info : t_20238, t_24583, t_28928
        config.timers('all-grads-sync').stop()                                 # trace_info : t_20239, t_24584, t_28929

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:                                              # trace_info : t_20247, t_24592, t_28937
        config.timers('layernorm-grads-all-reduce', log_level=1).start(        # trace_info : t_20248, t_20255, t_24593, t_24600, t_28938, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_20254, t_24599, t_28944
        )
    _allreduce_layernorm_grads(model, config)                                  # trace_info : t_20257, t_24602, t_28947
    if config.timers is not None:                                              # trace_info : t_20528, t_24873, t_29218
        config.timers('layernorm-grads-all-reduce').stop()                     # trace_info : t_20529, t_24874, t_29219

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:                                              # trace_info : t_20537, t_24882, t_29227
        config.timers('embedding-grads-all-reduce', log_level=1).start(        # trace_info : t_20538, t_20545, t_24883, t_24890, t_29228, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_20544, t_24889, t_29234
        )
    _allreduce_embedding_grads(model, config)                                  # trace_info : t_20547, t_24892, t_29237
    if config.timers is not None:                                              # trace_info : t_20569, t_24914, t_29259
        config.timers('embedding-grads-all-reduce').stop()                     # trace_info : t_20570, t_24915, t_29260

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:                                                 # trace_info : t_20578, t_24923, t_29268
        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        torch.distributed.broadcast(
            num_tokens,
            src=parallel_state.get_pipeline_model_parallel_last_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
