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

    if (                                                                       # trace_info : t_21520, t_25130, t_28740
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)         # trace_info : t_21516, t_25126, t_28736
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_21521, t_25131, t_28741
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
    if (                                                                       # trace_info : t_21530, t_25140, t_28750
        parallel_state.is_rank_in_position_embedding_group()                   # trace_info : t_21527, t_25137, t_28747
        and parallel_state.get_pipeline_model_parallel_world_size() > 1        # trace_info : t_21531, t_25141, t_28751
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
    _allreduce_word_embedding_grads(model, config)                             # trace_info : t_21515, t_25125, t_28735
    _allreduce_position_embedding_grads(model, config)                         # trace_info : t_21526, t_25136, t_28746


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (         # trace_info : t_21485, t_21492, t_21494, t_25095, t_25102, ...
        config.sequence_parallel or config.qk_layernorm                        # trace_info : t_21491, t_21493, t_25101, t_25103, t_28711, ...
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

    config = get_model_config(model[0])                                        # trace_info : t_21426, t_25036, t_28646

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:                                              # trace_info : t_21435, t_25045, t_28655
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)# trace_info : t_21436, t_25046, t_28656
    for model_chunk in model:                                                  # trace_info : t_21443, t_21464, t_25053, t_25074, t_28663, ...
        model_chunk.finish_grad_sync()                                         # trace_info : t_21444, t_25054, t_28664
    if config.timers is not None:                                              # trace_info : t_21465, t_25075, t_28685
        config.timers('all-grads-sync').stop()                                 # trace_info : t_21466, t_25076, t_28686

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:                                              # trace_info : t_21474, t_25084, t_28694
        config.timers('layernorm-grads-all-reduce', log_level=1).start(        # trace_info : t_21475, t_21482, t_25085, t_25092, t_28695, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_21481, t_25091, t_28701
        )
    _allreduce_layernorm_grads(model, config)                                  # trace_info : t_21484, t_25094, t_28704
    if config.timers is not None:                                              # trace_info : t_21495, t_25105, t_28715
        config.timers('layernorm-grads-all-reduce').stop()                     # trace_info : t_21496, t_25106, t_28716

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:                                              # trace_info : t_21504, t_25114, t_28724
        config.timers('embedding-grads-all-reduce', log_level=1).start(        # trace_info : t_21505, t_21512, t_25115, t_25122, t_28725, ...
            barrier=config.barrier_with_L1_time                                # trace_info : t_21511, t_25121, t_28731
        )
    _allreduce_embedding_grads(model, config)                                  # trace_info : t_21514, t_25124, t_28734
    if config.timers is not None:                                              # trace_info : t_21536, t_25146, t_28756
        config.timers('embedding-grads-all-reduce').stop()                     # trace_info : t_21537, t_25147, t_28757

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:                                                 # trace_info : t_21545, t_25155, t_28765
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
