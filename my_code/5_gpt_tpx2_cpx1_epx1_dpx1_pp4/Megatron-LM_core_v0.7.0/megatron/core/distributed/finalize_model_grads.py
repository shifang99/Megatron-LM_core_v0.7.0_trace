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

    if (                                                                       # trace_info : t_19899, t_23627, t_27355
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)         # trace_info : t_19895, t_23623, t_27351
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_19900, t_23628, t_27356
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):        # trace_info : t_19905, t_23633, t_27361
            model_module = model[0]                                            # trace_info : t_19912, t_23640, t_27368
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)# trace_info : t_19913, t_23641, t_27369
        if model_module.share_embeddings_and_output_weights:                   # trace_info : t_19929, t_23657, t_27385
            weight = model_module.shared_embedding_or_output_weight()          # trace_info : t_19930, t_23658, t_27386
            grad = weight.main_grad                                            # trace_info : t_19933, t_23661, t_27389
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())# trace_info : t_19934, t_23662, t_27390


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    if (                                                                       # trace_info : t_19941, t_23669, t_27397
        parallel_state.is_rank_in_position_embedding_group()                   # trace_info : t_19938, t_23666, t_27394
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_19942, t_23670, t_27398
        and config.pipeline_model_parallel_split_rank is not None              # trace_info : t_19947, t_23675, t_27403
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
    _allreduce_word_embedding_grads(model, config)                             # trace_info : t_19894, t_23622, t_27350
    _allreduce_position_embedding_grads(model, config)                         # trace_info : t_19937, t_23665, t_27393


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (         # trace_info : t_19864, t_19871, t_19873, t_23592, t_23599, ...
        config.sequence_parallel or config.qk_layernorm                        # trace_info : t_19870, t_19872, t_23598, t_23600, t_27326, ...
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

    config = get_model_config(model[0])                                        # trace_info : t_19805, t_23533, t_27261

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:                                              # trace_info : t_19814, t_23542, t_27270
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)# trace_info : t_19815, t_23543, t_27271
    for model_chunk in model:                                                  # trace_info : t_19822, t_19843, t_23550, t_23571, t_27278, ...
        model_chunk.finish_grad_sync()                                         # trace_info : t_19823, t_23551, t_27279
    if config.timers is not None:                                              # trace_info : t_19844, t_23572, t_27300
        config.timers('all-grads-sync').stop()                                 # trace_info : t_19845, t_23573, t_27301

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:                                              # trace_info : t_19853, t_23581, t_27309
        config.timers('layernorm-grads-all-reduce', log_level=1).start(        # trace_info : t_19854, t_19861, t_23582, t_23589, t_27310, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_19860, t_23588, t_27316
        )
    _allreduce_layernorm_grads(model, config)                                  # trace_info : t_19863, t_23591, t_27319
    if config.timers is not None:                                              # trace_info : t_19874, t_23602, t_27330
        config.timers('layernorm-grads-all-reduce').stop()                     # trace_info : t_19875, t_23603, t_27331

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:                                              # trace_info : t_19883, t_23611, t_27339
        config.timers('embedding-grads-all-reduce', log_level=1).start(        # trace_info : t_19884, t_19891, t_23612, t_23619, t_27340, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_19890, t_23618, t_27346
        )
    _allreduce_embedding_grads(model, config)                                  # trace_info : t_19893, t_23621, t_27349
    if config.timers is not None:                                              # trace_info : t_19948, t_23676, t_27404
        config.timers('embedding-grads-all-reduce').stop()                     # trace_info : t_19949, t_23677, t_27405

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:                                                 # trace_info : t_19957, t_23685, t_27413
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
