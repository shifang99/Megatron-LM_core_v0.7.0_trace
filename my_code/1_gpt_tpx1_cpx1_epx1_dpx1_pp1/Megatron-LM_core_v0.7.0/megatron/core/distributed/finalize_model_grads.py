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

    if (                                                                       # trace_info : t_16683, t_20322, t_23961
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)         # trace_info : t_16679, t_20318, t_23957
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_16684, t_20323, t_23962
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
    if (                                                                       # trace_info : t_16693, t_20332, t_23971
        parallel_state.is_rank_in_position_embedding_group()                   # trace_info : t_16690, t_20329, t_23968
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_16694, t_20333, t_23972
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
    _allreduce_word_embedding_grads(model, config)                             # trace_info : t_16678, t_20317, t_23956
    _allreduce_position_embedding_grads(model, config)                         # trace_info : t_16689, t_20328, t_23967


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (         # trace_info : t_16652, t_20291, t_23930
        config.sequence_parallel or config.qk_layernorm
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

    config = get_model_config(model[0])                                        # trace_info : t_16593, t_20232, t_23871

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:                                              # trace_info : t_16602, t_20241, t_23880
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)# trace_info : t_16603, t_20242, t_23881
    for model_chunk in model:                                                  # trace_info : t_16610, t_16631, t_20249, t_20270, t_23888, ...
        model_chunk.finish_grad_sync()                                         # trace_info : t_16611, t_20250, t_23889
    if config.timers is not None:                                              # trace_info : t_16632, t_20271, t_23910
        config.timers('all-grads-sync').stop()                                 # trace_info : t_16633, t_20272, t_23911

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:                                              # trace_info : t_16641, t_20280, t_23919
        config.timers('layernorm-grads-all-reduce', log_level=1).start(        # trace_info : t_16642, t_16649, t_20281, t_20288, t_23920, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_16648, t_20287, t_23926
        )
    _allreduce_layernorm_grads(model, config)                                  # trace_info : t_16651, t_20290, t_23929
    if config.timers is not None:                                              # trace_info : t_16658, t_20297, t_23936
        config.timers('layernorm-grads-all-reduce').stop()                     # trace_info : t_16659, t_20298, t_23937

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:                                              # trace_info : t_16667, t_20306, t_23945
        config.timers('embedding-grads-all-reduce', log_level=1).start(        # trace_info : t_16668, t_16675, t_20307, t_20314, t_23946, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_16674, t_20313, t_23952
        )
    _allreduce_embedding_grads(model, config)                                  # trace_info : t_16677, t_20316, t_23955
    if config.timers is not None:                                              # trace_info : t_16699, t_20338, t_23977
        config.timers('embedding-grads-all-reduce').stop()                     # trace_info : t_16700, t_20339, t_23978

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:                                                 # trace_info : t_16708, t_20347, t_23986
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
