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

    if isinstance(parameters, torch.Tensor):                                   # trace_info : t_21102, t_21472, t_25447, t_25817, t_29792, ...
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):                               # trace_info : t_21103, t_21473, t_25448, t_25818, t_29793, ...
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []                                                                 # trace_info : t_21104, t_21474, t_25449, t_25819, t_29794, ...
    for param in parameters:                                                   # trace_info : t_21105, t_21109, t_21113, t_21117, t_21121, ...
        if param.grad is not None:                                             # trace_info : t_21106, t_21110, t_21114, t_21118, t_21122, ...
            assert param.grad.type() == 'torch.cuda.FloatTensor'               # trace_info : t_21107, t_21111, t_21115, t_21119, t_21123, ...
            grads.append(param.grad.detach())                                  # trace_info : t_21108, t_21112, t_21116, t_21120, t_21124, ...

    # Norm parameters.
    max_norm = float(max_norm)                                                 # trace_info : t_21194, t_21508, t_25539, t_25853, t_29884, ...
    norm_type = float(norm_type)                                               # trace_info : t_21195, t_21509, t_25540, t_25854, t_29885, ...
    total_norm = 0.0                                                           # trace_info : t_21196, t_21510, t_25541, t_25855, t_29886, ...

    # Calculate norm.
    if norm_type == inf:                                                       # trace_info : t_21197, t_21511, t_25542, t_25856, t_29887, ...
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:                                                   # trace_info : t_21198, t_21512, t_25543, t_25857, t_29888, ...
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_21199, t_21513, t_25544, t_25858, t_29889, ...
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:                                                 # trace_info : t_21200, t_21514, t_25545, t_25859, t_29890, ...
                grad_norm, _ = multi_tensor_applier(                           # trace_info : t_21201, t_21206, t_21515, t_21520, t_25546, ...
                    amp_C.multi_tensor_l2norm,                                 # trace_info : t_21202, t_21516, t_25547, t_25861, t_29892, ...
                    dummy_overflow_buf,                                        # trace_info : t_21203, t_21517, t_25548, t_25862, t_29893, ...
                    [grads_for_norm],                                          # trace_info : t_21204, t_21518, t_25549, t_25863, t_29894, ...
                    False,  # no per-parameter norm                            # trace_info : t_21205, t_21519, t_25550, t_25864, t_29895, ...
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type                                # trace_info : t_21207, t_21521, t_25552, t_25866, t_29897, ...

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_21208, t_21210, t_21522, t_21524, t_25553, ...
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group# trace_info : t_21209, t_21523, t_25554, t_25868, t_29899, ...
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)                    # trace_info : t_21211, t_21525, t_25556, t_25870, t_29901, ...

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)                              # trace_info : t_21212, t_21526, t_25557, t_25871, t_29902, ...
    if clip_coeff < 1.0:                                                       # trace_info : t_21213, t_21527, t_25558, t_25872, t_29903, ...
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda') # trace_info : t_21214, t_21528, t_25559, t_25873, t_29904, ...
        multi_tensor_applier(                                                  # trace_info : t_21215, t_21217, t_21529, t_21531, t_25560, ...
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff# trace_info : t_21216, t_21530, t_25561, t_25875, t_29906, ...
        )

    return total_norm                                                          # trace_info : t_21218, t_21532, t_25563, t_25877, t_29908, ...


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
