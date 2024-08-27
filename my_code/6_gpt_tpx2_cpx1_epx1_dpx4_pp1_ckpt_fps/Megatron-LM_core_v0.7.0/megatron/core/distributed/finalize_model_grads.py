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

    if (                                                                       # trace_info : t_19816, t_23453, t_91060
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)         # trace_info : t_19812, t_23449, t_91056
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_19817, t_23454, t_91061
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
    if (                                                                       # trace_info : t_19826, t_23463, t_91070
        parallel_state.is_rank_in_position_embedding_group()                   # trace_info : t_19823, t_23460, t_91067
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_19827, t_23464, t_91071
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
    _allreduce_word_embedding_grads(model, config)                             # trace_info : t_19811, t_23448, t_91055
    _allreduce_position_embedding_grads(model, config)                         # trace_info : t_19822, t_23459, t_91066


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (         # trace_info : t_19781, t_19788, t_19790, t_23418, t_23425, ...
        config.sequence_parallel or config.qk_layernorm                        # trace_info : t_19787, t_19789, t_23424, t_23426, t_91031, ...
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

    config = get_model_config(model[0])                                        # trace_info : t_19721, t_23358, t_90965

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:                                              # trace_info : t_19730, t_23367, t_90974
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)# trace_info : t_19731, t_23368, t_90975
    for model_chunk in model:                                                  # trace_info : t_19738, t_19760, t_23375, t_23397, t_90982, ...
        model_chunk.finish_grad_sync()                                         # trace_info : t_19739, t_23376, t_90983
    if config.timers is not None:                                              # trace_info : t_19761, t_23398, t_91005
        config.timers('all-grads-sync').stop()                                 # trace_info : t_19762, t_23399, t_91006

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:                                              # trace_info : t_19770, t_23407, t_91014
        config.timers('layernorm-grads-all-reduce', log_level=1).start(        # trace_info : t_19771, t_19778, t_23408, t_23415, t_91015, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_19777, t_23414, t_91021
        )
    _allreduce_layernorm_grads(model, config)                                  # trace_info : t_19780, t_23417, t_91024
    if config.timers is not None:                                              # trace_info : t_19791, t_23428, t_91035
        config.timers('layernorm-grads-all-reduce').stop()                     # trace_info : t_19792, t_23429, t_91036

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:                                              # trace_info : t_19800, t_23437, t_91044
        config.timers('embedding-grads-all-reduce', log_level=1).start(        # trace_info : t_19801, t_19808, t_23438, t_23445, t_91045, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_19807, t_23444, t_91051
        )
    _allreduce_embedding_grads(model, config)                                  # trace_info : t_19810, t_23447, t_91054
    if config.timers is not None:                                              # trace_info : t_19832, t_23469, t_91076
        config.timers('embedding-grads-all-reduce').stop()                     # trace_info : t_19833, t_23470, t_91077

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:                                                 # trace_info : t_19841, t_23478, t_91085
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
