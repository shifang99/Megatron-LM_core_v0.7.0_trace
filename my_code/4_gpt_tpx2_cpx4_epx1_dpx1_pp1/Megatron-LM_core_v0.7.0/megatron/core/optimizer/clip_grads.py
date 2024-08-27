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

    if isinstance(parameters, torch.Tensor):                                   # trace_info : t_19919, t_23105, t_26291
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):                               # trace_info : t_19920, t_23106, t_26292
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []                                                                 # trace_info : t_19921, t_23107, t_26293
    for param in parameters:                                                   # trace_info : t_19922, t_19926, t_19930, t_19934, t_19938, ...
        if param.grad is not None:                                             # trace_info : t_19923, t_19927, t_19931, t_19935, t_19939, ...
            assert param.grad.type() == 'torch.cuda.FloatTensor'               # trace_info : t_19924, t_19928, t_19932, t_19936, t_19940, ...
            grads.append(param.grad.detach())                                  # trace_info : t_19925, t_19929, t_19933, t_19937, t_19941, ...

    # Norm parameters.
    max_norm = float(max_norm)                                                 # trace_info : t_20035, t_23221, t_26407
    norm_type = float(norm_type)                                               # trace_info : t_20036, t_23222, t_26408
    total_norm = 0.0                                                           # trace_info : t_20037, t_23223, t_26409

    # Calculate norm.
    if norm_type == inf:                                                       # trace_info : t_20038, t_23224, t_26410
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:                                                   # trace_info : t_20039, t_23225, t_26411
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_20040, t_23226, t_26412
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:                                                 # trace_info : t_20041, t_23227, t_26413
                grad_norm, _ = multi_tensor_applier(                           # trace_info : t_20042, t_20047, t_23228, t_23233, t_26414, ...
                    amp_C.multi_tensor_l2norm,                                 # trace_info : t_20043, t_23229, t_26415
                    dummy_overflow_buf,                                        # trace_info : t_20044, t_23230, t_26416
                    [grads_for_norm],                                          # trace_info : t_20045, t_23231, t_26417
                    False,  # no per-parameter norm                            # trace_info : t_20046, t_23232, t_26418
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type                                # trace_info : t_20048, t_23234, t_26420

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_20049, t_20051, t_23235, t_23237, t_26421, ...
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group# trace_info : t_20050, t_23236, t_26422
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)                    # trace_info : t_20052, t_23238, t_26424

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)                              # trace_info : t_20053, t_23239, t_26425
    if clip_coeff < 1.0:                                                       # trace_info : t_20054, t_23240, t_26426
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda') # trace_info : t_20055, t_23241, t_26427
        multi_tensor_applier(                                                  # trace_info : t_20056, t_20058, t_23242, t_23244, t_26428, ...
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff# trace_info : t_20057, t_23243, t_26429
        )

    return total_norm                                                          # trace_info : t_20059, t_23245, t_26431


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
