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

    if isinstance(parameters, torch.Tensor):                                   # trace_info : t_22353, t_25963, t_29573
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):                               # trace_info : t_22354, t_25964, t_29574
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []                                                                 # trace_info : t_22355, t_25965, t_29575
    for param in parameters:                                                   # trace_info : t_22356, t_22360, t_22364, t_22368, t_22372, ...
        if param.grad is not None:                                             # trace_info : t_22357, t_22361, t_22365, t_22369, t_22373, ...
            assert param.grad.type() == 'torch.cuda.FloatTensor'               # trace_info : t_22358, t_22362, t_22366, t_22370, t_22374, ...
            grads.append(param.grad.detach())                                  # trace_info : t_22359, t_22363, t_22367, t_22371, t_22375, ...

    # Norm parameters.
    max_norm = float(max_norm)                                                 # trace_info : t_22469, t_26079, t_29689
    norm_type = float(norm_type)                                               # trace_info : t_22470, t_26080, t_29690
    total_norm = 0.0                                                           # trace_info : t_22471, t_26081, t_29691

    # Calculate norm.
    if norm_type == inf:                                                       # trace_info : t_22472, t_26082, t_29692
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:                                                   # trace_info : t_22473, t_26083, t_29693
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_22474, t_26084, t_29694
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:                                                 # trace_info : t_22475, t_26085, t_29695
                grad_norm, _ = multi_tensor_applier(                           # trace_info : t_22476, t_22481, t_26086, t_26091, t_29696, ...
                    amp_C.multi_tensor_l2norm,                                 # trace_info : t_22477, t_26087, t_29697
                    dummy_overflow_buf,                                        # trace_info : t_22478, t_26088, t_29698
                    [grads_for_norm],                                          # trace_info : t_22479, t_26089, t_29699
                    False,  # no per-parameter norm                            # trace_info : t_22480, t_26090, t_29700
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type                                # trace_info : t_22482, t_26092, t_29702

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_22483, t_22485, t_26093, t_26095, t_29703, ...
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group# trace_info : t_22484, t_26094, t_29704
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)                    # trace_info : t_22486, t_26096, t_29706

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)                              # trace_info : t_22487, t_26097, t_29707
    if clip_coeff < 1.0:                                                       # trace_info : t_22488, t_26098, t_29708
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda') # trace_info : t_22489, t_26099, t_29709
        multi_tensor_applier(                                                  # trace_info : t_22490, t_22492, t_26100, t_26102, t_29710, ...
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff# trace_info : t_22491, t_26101, t_29711
        )

    return total_norm                                                          # trace_info : t_22493, t_26103, t_29713


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
