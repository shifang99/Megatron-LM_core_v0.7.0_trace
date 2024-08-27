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

    if (                                                                       # trace_info : t_19086, t_22272, t_25458
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)         # trace_info : t_19082, t_22268, t_25454
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_19087, t_22273, t_25459
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
    if (                                                                       # trace_info : t_19096, t_22282, t_25468
        parallel_state.is_rank_in_position_embedding_group()                   # trace_info : t_19093, t_22279, t_25465
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_19097, t_22283, t_25469
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
    _allreduce_word_embedding_grads(model, config)                             # trace_info : t_19081, t_22267, t_25453
    _allreduce_position_embedding_grads(model, config)                         # trace_info : t_19092, t_22278, t_25464


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (         # trace_info : t_19051, t_19058, t_19060, t_22237, t_22244, ...
        config.sequence_parallel or config.qk_layernorm                        # trace_info : t_19057, t_19059, t_22243, t_22245, t_25429, ...
    ):
        grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if (
                    param.requires_grad
                    and getattr(param, 'sequence_parallel', False)
                    or 'q_layernorm' in name
                    or 'k_layernorm' in name
                ):
                    grad = param.main_grad
                    grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])                                        # trace_info : t_18991, t_22177, t_25363

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:                                              # trace_info : t_19000, t_22186, t_25372
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)# trace_info : t_19001, t_22187, t_25373
    for model_chunk in model:                                                  # trace_info : t_19008, t_19030, t_22194, t_22216, t_25380, ...
        model_chunk.finish_grad_sync()                                         # trace_info : t_19009, t_22195, t_25381
    if config.timers is not None:                                              # trace_info : t_19031, t_22217, t_25403
        config.timers('all-grads-sync').stop()                                 # trace_info : t_19032, t_22218, t_25404

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:                                              # trace_info : t_19040, t_22226, t_25412
        config.timers('layernorm-grads-all-reduce', log_level=1).start(        # trace_info : t_19041, t_19048, t_22227, t_22234, t_25413, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_19047, t_22233, t_25419
        )
    _allreduce_layernorm_grads(model, config)                                  # trace_info : t_19050, t_22236, t_25422
    if config.timers is not None:                                              # trace_info : t_19061, t_22247, t_25433
        config.timers('layernorm-grads-all-reduce').stop()                     # trace_info : t_19062, t_22248, t_25434

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:                                              # trace_info : t_19070, t_22256, t_25442
        config.timers('embedding-grads-all-reduce', log_level=1).start(        # trace_info : t_19071, t_19078, t_22257, t_22264, t_25443, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_19077, t_22263, t_25449
        )
    _allreduce_embedding_grads(model, config)                                  # trace_info : t_19080, t_22266, t_25452
    if config.timers is not None:                                              # trace_info : t_19102, t_22288, t_25474
        config.timers('embedding-grads-all-reduce').stop()                     # trace_info : t_19103, t_22289, t_25475

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:                                                 # trace_info : t_19111, t_22297, t_25483
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
