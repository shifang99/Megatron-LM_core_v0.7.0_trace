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

    if isinstance(parameters, torch.Tensor):                                   # trace_info : t_17516, t_21155, t_24794
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):                               # trace_info : t_17517, t_21156, t_24795
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []                                                                 # trace_info : t_17518, t_21157, t_24796
    for param in parameters:                                                   # trace_info : t_17519, t_17523, t_17527, t_17531, t_17535, ...
        if param.grad is not None:                                             # trace_info : t_17520, t_17524, t_17528, t_17532, t_17536, ...
            assert param.grad.type() == 'torch.cuda.FloatTensor'               # trace_info : t_17521, t_17525, t_17529, t_17533, t_17537, ...
            grads.append(param.grad.detach())                                  # trace_info : t_17522, t_17526, t_17530, t_17534, t_17538, ...

    # Norm parameters.
    max_norm = float(max_norm)                                                 # trace_info : t_17632, t_21271, t_24910
    norm_type = float(norm_type)                                               # trace_info : t_17633, t_21272, t_24911
    total_norm = 0.0                                                           # trace_info : t_17634, t_21273, t_24912

    # Calculate norm.
    if norm_type == inf:                                                       # trace_info : t_17635, t_21274, t_24913
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:                                                   # trace_info : t_17636, t_21275, t_24914
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_17637, t_21276, t_24915
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:                                                 # trace_info : t_17638, t_21277, t_24916
                grad_norm, _ = multi_tensor_applier(                           # trace_info : t_17639, t_17644, t_21278, t_21283, t_24917, ...
                    amp_C.multi_tensor_l2norm,                                 # trace_info : t_17640, t_21279, t_24918
                    dummy_overflow_buf,                                        # trace_info : t_17641, t_21280, t_24919
                    [grads_for_norm],                                          # trace_info : t_17642, t_21281, t_24920
                    False,  # no per-parameter norm                            # trace_info : t_17643, t_21282, t_24921
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type                                # trace_info : t_17645, t_21284, t_24923

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_17646, t_17648, t_21285, t_21287, t_24924, ...
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group# trace_info : t_17647, t_21286, t_24925
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)                    # trace_info : t_17649, t_21288, t_24927

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)                              # trace_info : t_17650, t_21289, t_24928
    if clip_coeff < 1.0:                                                       # trace_info : t_17651, t_21290, t_24929
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda') # trace_info : t_17652, t_21291, t_24930
        multi_tensor_applier(                                                  # trace_info : t_17653, t_17655, t_21292, t_21294, t_24931, ...
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff# trace_info : t_17654, t_21293, t_24932
        )

    return total_norm                                                          # trace_info : t_17656, t_21295, t_24934


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
