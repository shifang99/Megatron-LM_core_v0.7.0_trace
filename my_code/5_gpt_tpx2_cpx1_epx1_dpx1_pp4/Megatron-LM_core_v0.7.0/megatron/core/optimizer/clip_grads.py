# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Gradient clipping."""

import os
from typing import List, Optional, Union

import amp_C
import torch
from apex.multi_tensor_apply import multi_tensor_applier
from torch import inf

from ..tensor_parallel import param_is_not_tensor_parallel_duplicate
from ..transformer.module import param_is_not_shared


def clip_grad_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    max_norm: Union[int, float],
    norm_type: Union[int, float] = 2,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized.
        grads_for_norm (Iterable[Tensor]): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        max_norm (float or int): max norm of the gradients.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):                                   # trace_info : t_20711, t_24439, t_28167
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):                               # trace_info : t_20712, t_24440, t_28168
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []                                                                 # trace_info : t_20713, t_24441, t_28169
    for param in parameters:                                                   # trace_info : t_20714, t_20718, t_20722, t_20726, t_20730, ...
        if param.grad is not None:                                             # trace_info : t_20715, t_20719, t_20723, t_20727, t_20731, ...
            assert param.grad.type() == 'torch.cuda.FloatTensor'               # trace_info : t_20716, t_20720, t_20724, t_20728, t_20732, ...
            grads.append(param.grad.detach())                                  # trace_info : t_20717, t_20721, t_20725, t_20729, t_20733, ...

    # Norm parameters.
    max_norm = float(max_norm)                                                 # trace_info : t_20819, t_24547, t_28275
    norm_type = float(norm_type)                                               # trace_info : t_20820, t_24548, t_28276
    total_norm = 0.0                                                           # trace_info : t_20821, t_24549, t_28277

    # Calculate norm.
    if norm_type == inf:                                                       # trace_info : t_20822, t_24550, t_28278
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:                                                   # trace_info : t_20823, t_24551, t_28279
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_20824, t_24552, t_28280
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:                                                 # trace_info : t_20825, t_24553, t_28281
                grad_norm, _ = multi_tensor_applier(                           # trace_info : t_20826, t_20831, t_24554, t_24559, t_28282, ...
                    amp_C.multi_tensor_l2norm,                                 # trace_info : t_20827, t_24555, t_28283
                    dummy_overflow_buf,                                        # trace_info : t_20828, t_24556, t_28284
                    [grads_for_norm],                                          # trace_info : t_20829, t_24557, t_28285
                    False,  # no per-parameter norm                            # trace_info : t_20830, t_24558, t_28286
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type                                # trace_info : t_20832, t_24560, t_28288

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_20833, t_20835, t_24561, t_24563, t_28289, ...
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group# trace_info : t_20834, t_24562, t_28290
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)                    # trace_info : t_20836, t_24564, t_28292

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)                              # trace_info : t_20837, t_24565, t_28293
    if clip_coeff < 1.0:                                                       # trace_info : t_20838, t_24566, t_28294
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda') # trace_info : t_20839, t_24567, t_28295
        multi_tensor_applier(                                                  # trace_info : t_20840, t_20842, t_24568, t_24570, t_28296, ...
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff# trace_info : t_20841, t_24569, t_28297
        )

    return total_norm                                                          # trace_info : t_20843, t_24571, t_28299


def count_zeros_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    model_parallel_group: torch.distributed.ProcessGroup,
) -> float:
    """Counts the number of zeros in gradients associated with the passed-in list of
    parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have the number of zeros in its corresponding
            gradient counted.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = torch.tensor([0.0], dtype=torch.float, device='cuda')
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
    )

    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
