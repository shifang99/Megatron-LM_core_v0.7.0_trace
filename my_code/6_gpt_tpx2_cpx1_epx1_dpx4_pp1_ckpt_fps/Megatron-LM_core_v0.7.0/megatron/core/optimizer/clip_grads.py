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

    if isinstance(parameters, torch.Tensor):                                   # trace_info : t_20649, t_24286, t_91893
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):                               # trace_info : t_20650, t_24287, t_91894
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []                                                                 # trace_info : t_20651, t_24288, t_91895
    for param in parameters:                                                   # trace_info : t_20652, t_20656, t_20660, t_20664, t_20668, ...
        if param.grad is not None:                                             # trace_info : t_20653, t_20657, t_20661, t_20665, t_20669, ...
            assert param.grad.type() == 'torch.cuda.FloatTensor'               # trace_info : t_20654, t_20658, t_20662, t_20666, t_20670, ...
            grads.append(param.grad.detach())                                  # trace_info : t_20655, t_20659, t_20663, t_20667, t_20671, ...

    # Norm parameters.
    max_norm = float(max_norm)                                                 # trace_info : t_20765, t_24402, t_92009
    norm_type = float(norm_type)                                               # trace_info : t_20766, t_24403, t_92010
    total_norm = 0.0                                                           # trace_info : t_20767, t_24404, t_92011

    # Calculate norm.
    if norm_type == inf:                                                       # trace_info : t_20768, t_24405, t_92012
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:                                                   # trace_info : t_20769, t_24406, t_92013
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_20770, t_24407, t_92014
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:                                                 # trace_info : t_20771, t_24408, t_92015
                grad_norm, _ = multi_tensor_applier(                           # trace_info : t_20772, t_20777, t_24409, t_24414, t_92016, ...
                    amp_C.multi_tensor_l2norm,                                 # trace_info : t_20773, t_24410, t_92017
                    dummy_overflow_buf,                                        # trace_info : t_20774, t_24411, t_92018
                    [grads_for_norm],                                          # trace_info : t_20775, t_24412, t_92019
                    False,  # no per-parameter norm                            # trace_info : t_20776, t_24413, t_92020
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type                                # trace_info : t_20778, t_24415, t_92022

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_20779, t_20781, t_24416, t_24418, t_92023, ...
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group# trace_info : t_20780, t_24417, t_92024
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)                    # trace_info : t_20782, t_24419, t_92026

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)                              # trace_info : t_20783, t_24420, t_92027
    if clip_coeff < 1.0:                                                       # trace_info : t_20784, t_24421, t_92028
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda') # trace_info : t_20785, t_24422, t_92029
        multi_tensor_applier(                                                  # trace_info : t_20786, t_20788, t_24423, t_24425, t_92030, ...
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff# trace_info : t_20787, t_24424, t_92031
        )

    return total_norm                                                          # trace_info : t_20789, t_24426, t_92033


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
