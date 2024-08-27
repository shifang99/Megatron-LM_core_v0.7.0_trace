# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import io
import math
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_and_expert_parallel_rank,
    get_tensor_and_expert_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from ..dist_checkpointing.mapping import ShardedStateDict
from ..transformer.utils import make_sharded_tensors_for_checkpoint
from ..utils import make_tp_sharded_tensor_for_checkpoint, prepare_input_tensors_for_wgrad_compute
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from .utils import VocabUtility, divide, split_tensor_along_last_dim

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    'tensor_model_parallel': False,
    'partition_dim': -1,
    'partition_stride': 1,
}


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (# trace_info : t_20384, t_20393, t_20400, t_20409, t_20418, ...
        get_tensor_model_parallel_rank() == 0                                  # trace_info : t_20394, t_20482, t_20498, t_20514, t_20539, ...
    )


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_9852, t_9854, t_9856, t_9858, t_10263, ...
        assert not hasattr(tensor, attribute)                                  # trace_info : t_9853, t_9855, t_9857, t_10264, t_10266, ...
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)                      # trace_info : t_9859, t_10270, t_10412, t_10469, t_10741, ...
    setattr(tensor, 'partition_dim', dim)                                      # trace_info : t_9860, t_10271, t_10413, t_10470, t_10742, ...
    setattr(tensor, 'partition_stride', stride)                                # trace_info : t_9861, t_10272, t_10414, t_10471, t_10743, ...


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):                                           # trace_info : t_12021, t_12034, t_12050, t_12066, t_12082, ...
        if not hasattr(tensor, attribute):                                     # trace_info : t_12024, t_12027, t_12030, t_12037, t_12041, ...
            setattr(tensor, attribute, value)                                  # trace_info : t_12038, t_12042, t_12046, t_12054, t_12058, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_12022, t_12025, t_12028, t_12031, t_12035, ...
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])    # trace_info : t_12023, t_12026, t_12029, t_12036, t_12040, ...


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):                                                 # trace_info : t_15245, t_15269, t_15293, t_15317, t_15341, ...
        if hasattr(source_tensor, attribute):                                  # trace_info : t_15248, t_15252, t_15256, t_15272, t_15276, ...
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))# trace_info : t_15249, t_15253, t_15257, t_15273, t_15277, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_15246, t_15250, t_15254, t_15258, t_15270, ...
        maybe_copy(attribute)                                                  # trace_info : t_15247, t_15251, t_15255, t_15271, t_15275, ...


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(                                      # trace_info : t_9849, t_9851, t_10260, t_10262, t_10402, ...
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride      # trace_info : t_9850, t_10261, t_10403, t_10732, t_10890, ...
    )

    if not expert_parallel:                                                    # trace_info : t_9862, t_10273, t_10415, t_10744, t_10902, ...
        with get_cuda_rng_tracker().fork():                                    # trace_info : t_9863, t_9885, t_10274, t_10296, t_10416, ...
            init_method(weight)                                                # trace_info : t_9883, t_10294, t_10436, t_10765, t_10923, ...
    else:
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    rank=None,
    world_size=None,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    if rank is None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        # all tensors must live on the same device
        cpu_weight = torch.cat(my_weight_list, dim=partition_dim).to_dense()
        weight.data.copy_(cpu_weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Args:
        config: A megatron.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        config: ModelParallelConfig,
    ):
        super(VocabParallelEmbedding, self).__init__()                         # trace_info : t_9807
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings                                   # trace_info : t_9808
        self.embedding_dim = embedding_dim                                     # trace_info : t_9809
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()# trace_info : t_9810
        # Divide the weight matrix along the vocaburaly dimension.
        (                                                                      # trace_info : t_9834
            self.vocab_start_index,                                            # trace_info : t_9835
            self.vocab_end_index,                                              # trace_info : t_9836
        ) = VocabUtility.vocab_range_from_global_vocab_size(                   # trace_info : t_9816, t_9823
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size# trace_info : t_9817
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index# trace_info : t_9837

        # Allocate weights and initialize.
        if config.use_cpu_initialization:                                      # trace_info : t_9838
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(                                           # trace_info : t_9839, t_9846
                torch.empty(                                                   # trace_info : t_9840, t_9845
                    self.num_embeddings_per_partition,                         # trace_info : t_9841
                    self.embedding_dim,                                        # trace_info : t_9842
                    device=torch.cuda.current_device(),                        # trace_info : t_9843
                    dtype=config.params_dtype,                                 # trace_info : t_9844
                )
            )
            if config.perform_initialization:                                  # trace_info : t_9847
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)# trace_info : t_9848

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18343, t_22073, t_25801
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)# trace_info : t_18344, t_22074, t_25802
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index             # trace_info : t_18345, t_22075, t_25803
            masked_input[input_mask] = 0                                       # trace_info : t_18346, t_22076, t_25804
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.weight[masked_input]                            # trace_info : t_18347, t_22077, t_25805
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18348, t_22078, t_25806
            output_parallel[input_mask, :] = 0.0                               # trace_info : t_18349, t_22079, t_25807
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)     # trace_info : t_18350, t_22080, t_25808
        return output                                                          # trace_info : t_18364, t_22094, t_25822

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """ Non-default implementation for embeddings due to `allow_shape_mismatch` param """
        state_dict = self.state_dict(prefix='', keep_vars=True)

        weight_prefix = f'{prefix}weight'
        return {
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(
                tensor=state_dict['weight'],
                key=weight_prefix,
                allow_shape_mismatch=True,
                prepend_offsets=sharded_offsets,
            )
        }


class LinearWithFrozenWeight(torch.autograd.Function):
    """Linear operator that does not calculate gradient for weight.
    This op and LinearWithGradAccumulationAndAsyncCommunication performs
    mathematically-identical forward and DGRAD.

    Conceptually this op is the same as torch.nn.functional.linear with
    weight.requires_grad==False, but in experiments they are not identical
    mathematically. """

    @staticmethod
    @custom_fwd
    def forward(
        ctx, input, weight, bias, allreduce_dgrad,
    ):
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)

        if ctx.allreduce_dgrad:
            # All-reduce. Note: here async and sync are effectively the same.
            torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group())

        return grad_input, None, None, None


def linear_with_frozen_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    allreduce_dgrad: bool = None,
) -> torch.Tensor:
    """Linear layer execution with weight.requires_grad == False.

    This function handles linear layers with weight frozen (untrainable).
    In the forward, it only saves weight and does not save input activations.
    In the backward, it does not perform weight gradient calculation, or
    weight gradient allreduce.

    Args:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    async_grad_allreduce (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.

    grad_output_buffer (List[torch.Tensor] optional): dummy argument, used to
    keep the API unified between all forward implementation functions.

    allreduce_dgrad (bool): Do the allreduce of input gradients.
        Here, async and sync allreduce are the same. If sequence_parallel is
        True, this must be False, as no all reduce is performed.

    """

    assert grad_output_buffer is None, (
        "grad_output_buffer kwarg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )

    if sequence_parallel:
        input = gather_from_sequence_parallel_region(input, tensor_parallel_output_grad=True)
    else:
        input = input

    if allreduce_dgrad is None:
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [
        input,
        weight,
        bias,
        allreduce_dgrad,
    ]

    return LinearWithFrozenWeight.apply(*args)


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
    ):
        ctx.save_for_backward(input, weight)                                   # trace_info : t_18475, t_18672, t_18788, t_18840, t_18970, ...
        ctx.use_bias = bias is not None                                        # trace_info : t_18476, t_18673, t_18789, t_18841, t_18971, ...
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion        # trace_info : t_18477, t_18674, t_18790, t_18842, t_18972, ...
        ctx.allreduce_dgrad = allreduce_dgrad                                  # trace_info : t_18478, t_18675, t_18791, t_18843, t_18973, ...
        ctx.sequence_parallel = sequence_parallel                              # trace_info : t_18479, t_18676, t_18792, t_18844, t_18974, ...
        ctx.grad_output_buffer = grad_output_buffer                            # trace_info : t_18480, t_18677, t_18793, t_18845, t_18975, ...

        if sequence_parallel:                                                  # trace_info : t_18481, t_18678, t_18794, t_18846, t_18976, ...
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input                                                # trace_info : t_18482, t_18679, t_18795, t_18847, t_18977, ...

        output = torch.matmul(total_input, weight.t())                         # trace_info : t_18483, t_18680, t_18796, t_18848, t_18978, ...
        if bias is not None:                                                   # trace_info : t_18484, t_18681, t_18797, t_18849, t_18979, ...
            output = output + bias                                             # trace_info : t_18485, t_18980, t_22215, t_22708, t_25943, ...
        return output                                                          # trace_info : t_18486, t_18682, t_18798, t_18850, t_18981, ...

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer

        wgrad_compute = True
        if grad_output_buffer is not None:
            grad_output_buffer.append(grad_output)
            wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    allreduce_dgrad: bool = None,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."


        async_grad_allreduce (bool required): Do the allreduce of input
            gradients asyncronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.


        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        allreduce_dgrad (bool): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.
    """
    if allreduce_dgrad is None:                                                # trace_info : t_18463, t_18660, t_18776, t_18828, t_18958, ...
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [                                                                   # trace_info : t_18471, t_18668, t_18784, t_18836, t_18966, ...
        input,                                                                 # trace_info : t_18464, t_18661, t_18777, t_18829, t_18959, ...
        weight,                                                                # trace_info : t_18465, t_18662, t_18778, t_18830, t_18960, ...
        bias,                                                                  # trace_info : t_18466, t_18663, t_18779, t_18831, t_18961, ...
        gradient_accumulation_fusion,                                          # trace_info : t_18467, t_18664, t_18780, t_18832, t_18962, ...
        allreduce_dgrad,                                                       # trace_info : t_18468, t_18665, t_18781, t_18833, t_18963, ...
        sequence_parallel,                                                     # trace_info : t_18469, t_18666, t_18782, t_18834, t_18964, ...
        grad_output_buffer,                                                    # trace_info : t_18470, t_18667, t_18783, t_18835, t_18965, ...
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:           # trace_info : t_18472, t_18669, t_18785, t_18837, t_18967, ...
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":               # trace_info : t_18473, t_18670, t_18786, t_18838, t_18968, ...
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)        # trace_info : t_18474, t_18671, t_18787, t_18839, t_18969, ...


linear_with_grad_accumulation_and_async_allreduce.warned = False


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available to all GPUs, otherwise, every GPU will have its output which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be set to False. It returns the master weights used for initialization.
        skip_bias_add: If True, do not add the bias term, instead return it to be added by the caller. This enables performance optimations where bias can be fused with other elementwise operations.
        skip_weight_param_allocation: If True, weight parameter is not allocated and must be passed as a keyword argument `weight` during the forward pass. Note that this does not affect bias, which will be allocated if bias is True. Defaults to False.
        embedding_activation_buffer: This buffer holds the input activations of the final embedding linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        grad_output_buffer: This buffer holds the gradient outputs of the final embedding linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        is_expert: If True, the layer is treated as an MoE expert layer.
        config: ModelParallelConfig object
        tp_comm_buffer_name: Communication buffer name is not used in non-Transformer-Engine modules.
        disable_grad_reduce: If True, reduction of output gradients across tensor-parallel ranks will be disabled. Defaults to False. This feature is used by Lora Adapter in Nemo to delay and fuse reduction along with other gradients for performance optimization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
    ):
        super(ColumnParallelLinear, self).__init__()                           # trace_info : t_10355, t_10684, t_11357, t_11686

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_10356, t_10685, t_11358, t_11687
        self.output_size = output_size                                         # trace_info : t_10357, t_10686, t_11359, t_11688
        self.gather_output = gather_output                                     # trace_info : t_10358, t_10687, t_11360, t_11689
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_10359, t_10688, t_11361, t_11690
        self.is_expert = is_expert                                             # trace_info : t_10360, t_10689, t_11362, t_11691
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_10361, t_10690, t_11363, t_11692
        self.embedding_activation_buffer = embedding_activation_buffer         # trace_info : t_10362, t_10691, t_11364, t_11693
        self.grad_output_buffer = grad_output_buffer                           # trace_info : t_10363, t_10692, t_11365, t_11694
        self.config = config                                                   # trace_info : t_10364, t_10693, t_11366, t_11695
        self.disable_grad_reduce = disable_grad_reduce                         # trace_info : t_10365, t_10694, t_11367, t_11696

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_10366, t_10695, t_11368, t_11697
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_10367, t_10696, t_11369, t_11698
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_10368, t_10697, t_11370, t_11699
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_10374, t_10703, t_11376, t_11705

        self.output_size_per_partition = divide(output_size, world_size)       # trace_info : t_10380, t_10709, t_11382, t_11711

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:                                   # trace_info : t_10384, t_10713, t_11386, t_11715
            if config.use_cpu_initialization:                                  # trace_info : t_10385, t_10714, t_11387, t_11716
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=world_size,
                    )
            else:
                self.weight = Parameter(                                       # trace_info : t_10386, t_10393, t_10715, t_10722, t_11388, ...
                    torch.empty(                                               # trace_info : t_10387, t_10392, t_10716, t_10721, t_11389, ...
                        self.output_size_per_partition,                        # trace_info : t_10388, t_10717, t_11390, t_11719
                        self.input_size,                                       # trace_info : t_10389, t_10718, t_11391, t_11720
                        device=torch.cuda.current_device(),                    # trace_info : t_10390, t_10719, t_11392, t_11721
                        dtype=config.params_dtype,                             # trace_info : t_10391, t_10720, t_11393, t_11722
                    )
                )
                if config.perform_initialization:                              # trace_info : t_10394, t_10723, t_11396, t_11725
                    _initialize_affine_weight_gpu(                             # trace_info : t_10395, t_10401, t_10724, t_10730, t_11397, ...
                        self.weight,                                           # trace_info : t_10396, t_10725, t_11398, t_11727
                        init_method,                                           # trace_info : t_10397, t_10726, t_11399, t_11728
                        partition_dim=0,                                       # trace_info : t_10398, t_10727, t_11400, t_11729
                        stride=stride,                                         # trace_info : t_10399, t_10728, t_11401, t_11730
                        expert_parallel=(self.is_expert and self.expert_parallel),# trace_info : t_10400, t_10729, t_11402, t_11731
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10451, t_10780, t_11453, t_11782
        else:
            self.weight = None

        if bias:                                                               # trace_info : t_10452, t_10781, t_11454, t_11783
            if config.use_cpu_initialization:                                  # trace_info : t_10453, t_10782, t_11455, t_11784
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(                                         # trace_info : t_10454, t_10460, t_10783, t_10789, t_11456, ...
                    torch.empty(                                               # trace_info : t_10455, t_10459, t_10784, t_10788, t_11457, ...
                        self.output_size_per_partition,                        # trace_info : t_10456, t_10785, t_11458, t_11787
                        device=torch.cuda.current_device(),                    # trace_info : t_10457, t_10786, t_11459, t_11788
                        dtype=config.params_dtype,                             # trace_info : t_10458, t_10787, t_11460, t_11789
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)   # trace_info : t_10461, t_10790, t_11463, t_11792
            if config.perform_initialization:                                  # trace_info : t_10472, t_10801, t_11474, t_11803
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_10473, t_10475, t_10802, t_10804, t_11475, ...
                    self.bias.zero_()                                          # trace_info : t_10474, t_10803, t_11476, t_11805
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10476, t_10805, t_11478, t_11807
        else:
            self.register_parameter('bias', None)

        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_10477, t_10806, t_11479, t_11808
        if self.sequence_parallel and world_size <= 1:                         # trace_info : t_10478, t_10807, t_11480, t_11809
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel   # trace_info : t_10479, t_10808, t_11481, t_11810

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:# trace_info : t_10480, t_10809, t_11482, t_11811
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_10481, t_10810, t_11483, t_11812

        if self.allreduce_dgrad and self.sequence_parallel:                    # trace_info : t_10482, t_10811, t_11484, t_11813
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_10483, t_10812, t_11485, t_11814

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_10484, t_10486, t_10813, t_10815, t_11486, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_10485, t_10814, t_11487, t_11816
                f'{prefix}_extra_state'
            )
        )

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            weight (optional): weight tensor to use, compulsory when
                skip_weight_param_allocation is True.

        Returns:
            - output
            - bias

        """
        if weight is None:                                                     # trace_info : t_18439, t_18752, t_18934, t_19245, t_22169, ...
            if self.weight is None:                                            # trace_info : t_18440, t_18753, t_18935, t_19246, t_22170, ...
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight                                               # trace_info : t_18441, t_18754, t_18936, t_19247, t_22171, ...
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_18442, t_18755, t_18937, t_19248, t_22172, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None                   # trace_info : t_18443, t_18756, t_18938, t_19249, t_22173, ...

        if (                                                                   # trace_info : t_18445, t_18758, t_18940, t_19251, t_22175, ...
            self.allreduce_dgrad                                               # trace_info : t_18444, t_18757, t_18939, t_19250, t_22174, ...
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_                                            # trace_info : t_18446, t_18759, t_18941, t_19252, t_22176, ...
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:                          # trace_info : t_18447, t_18760, t_18942, t_19253, t_22177, ...
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:                                           # trace_info : t_18448, t_18761, t_18943, t_19254, t_22178, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_18449, t_18762, t_18944, t_19255, t_22179, ...

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad# trace_info : t_18450, t_18763, t_18945, t_19256, t_22180, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_18451, t_18462, t_18764, t_18775, t_18946, ...
            input=input_parallel,                                              # trace_info : t_18452, t_18765, t_18947, t_19258, t_22182, ...
            weight=weight,                                                     # trace_info : t_18453, t_18766, t_18948, t_19259, t_22183, ...
            bias=bias,                                                         # trace_info : t_18454, t_18767, t_18949, t_19260, t_22184, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_18455, t_18768, t_18950, t_19261, t_22185, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_18456, t_18769, t_18951, t_19262, t_22186, ...
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,# trace_info : t_18457, t_18770, t_18952, t_19263, t_22187, ...
            grad_output_buffer=self.grad_output_buffer                         # trace_info : t_18459, t_18772, t_18954, t_19265, t_22189, ...
            if self.config.defer_embedding_wgrad_compute                       # trace_info : t_18458, t_18771, t_18953, t_19264, t_22188, ...
            else None,                                                         # trace_info : t_18460, t_18773, t_18955, t_19266, t_22190, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_18461, t_18774, t_18956, t_19267, t_22191, ...
        )
        if self.gather_output:                                                 # trace_info : t_18487, t_18799, t_18982, t_19292, t_22217, ...
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel                                           # trace_info : t_18488, t_18800, t_18983, t_19293, t_22218, ...
        output_bias = self.bias if self.skip_bias_add else None                # trace_info : t_18489, t_18801, t_18984, t_19294, t_22219, ...
        return output, output_bias                                             # trace_info : t_18490, t_18802, t_18985, t_19295, t_22220, ...

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already split across the GPUs and we do not split again.
        init_method: method to initialize weights. Note that bias is always set to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be set to False. It returns the master weights used for initialization.
        skip_bias_add: If True, do not add the bias term, instead return it to be added by the caller. This enables performance optimations where bias can be fused with other elementwise operations.
        is_expert: If True, the layer is treated as an MoE expert layer
        tp_comm_buffer_name: Communication buffer name. Not used in
                             non-Transformer-Engine modules.
        config: ModelParallelConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
    ):
        super(RowParallelLinear, self).__init__()                              # trace_info : t_10214, t_10843, t_11216, t_11845

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_10215, t_10844, t_11217, t_11846
        self.output_size = output_size                                         # trace_info : t_10216, t_10845, t_11218, t_11847
        self.input_is_parallel = input_is_parallel                             # trace_info : t_10217, t_10846, t_11219, t_11848
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_10218, t_10847, t_11220, t_11849
        self.config = config                                                   # trace_info : t_10219, t_10848, t_11221, t_11850
        self.is_expert = is_expert                                             # trace_info : t_10220, t_10849, t_11222, t_11851
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_10221, t_10850, t_11223, t_11852
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_10222, t_10851, t_11224, t_11853
        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_10223, t_10852, t_11225, t_11854
        if self.sequence_parallel and not self.input_is_parallel:              # trace_info : t_10224, t_10853, t_11226, t_11855
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_10225, t_10854, t_11227, t_11856
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_10226, t_10855, t_11228, t_11857
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_10227, t_10856, t_11229, t_11858
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_10233, t_10862, t_11235, t_11864

        self.input_size_per_partition = divide(input_size, world_size)         # trace_info : t_10239, t_10868, t_11241, t_11870

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:                                      # trace_info : t_10243, t_10872, t_11245, t_11874
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(                                           # trace_info : t_10244, t_10251, t_10873, t_10880, t_11246, ...
                torch.empty(                                                   # trace_info : t_10245, t_10250, t_10874, t_10879, t_11247, ...
                    self.output_size,                                          # trace_info : t_10246, t_10875, t_11248, t_11877
                    self.input_size_per_partition,                             # trace_info : t_10247, t_10876, t_11249, t_11878
                    device=torch.cuda.current_device(),                        # trace_info : t_10248, t_10877, t_11250, t_11879
                    dtype=config.params_dtype,                                 # trace_info : t_10249, t_10878, t_11251, t_11880
                )
            )
            if config.perform_initialization:                                  # trace_info : t_10252, t_10881, t_11254, t_11883
                _initialize_affine_weight_gpu(                                 # trace_info : t_10253, t_10259, t_10882, t_10888, t_11255, ...
                    self.weight,                                               # trace_info : t_10254, t_10883, t_11256, t_11885
                    init_method,                                               # trace_info : t_10255, t_10884, t_11257, t_11886
                    partition_dim=1,                                           # trace_info : t_10256, t_10885, t_11258, t_11887
                    stride=stride,                                             # trace_info : t_10257, t_10886, t_11259, t_11888
                    expert_parallel=(self.is_expert and self.expert_parallel), # trace_info : t_10258, t_10887, t_11260, t_11889
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10309, t_10938, t_11311, t_11940

        if bias:                                                               # trace_info : t_10310, t_10939, t_11312, t_11941
            if config.use_cpu_initialization:                                  # trace_info : t_10311, t_10940, t_11313, t_11942
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(                                         # trace_info : t_10312, t_10318, t_10941, t_10947, t_11314, ...
                    torch.empty(                                               # trace_info : t_10313, t_10317, t_10942, t_10946, t_11315, ...
                        self.output_size,                                      # trace_info : t_10314, t_10943, t_11316, t_11945
                        device=torch.cuda.current_device(),                    # trace_info : t_10315, t_10944, t_11317, t_11946
                        dtype=config.params_dtype,                             # trace_info : t_10316, t_10945, t_11318, t_11947
                    )
                )

            if config.perform_initialization:                                  # trace_info : t_10319, t_10948, t_11321, t_11950
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_10320, t_10322, t_10949, t_10951, t_11322, ...
                    self.bias.zero_()                                          # trace_info : t_10321, t_10950, t_11323, t_11952
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10323, t_10952, t_11325, t_11954
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)    # trace_info : t_10324, t_10953, t_11326, t_11955
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_10325, t_10954, t_11327, t_11956

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_10326, t_10328, t_10955, t_10957, t_11328, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_10327, t_10956, t_11329, t_11958
                f'{prefix}_extra_state'
            )
        )

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_18644, t_18812, t_19137, t_19305, t_22372, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:                                             # trace_info : t_18645, t_18813, t_19138, t_19306, t_22373, ...
            input_parallel = input_                                            # trace_info : t_18646, t_18814, t_19139, t_19307, t_22374, ...
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if not self.weight.requires_grad:                                      # trace_info : t_18647, t_18815, t_19140, t_19308, t_22375, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_18648, t_18816, t_19141, t_19309, t_22376, ...

        allreduce_dgrad = False                                                # trace_info : t_18649, t_18817, t_19142, t_19310, t_22377, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_18650, t_18659, t_18818, t_18827, t_19143, ...
            input=input_parallel,                                              # trace_info : t_18651, t_18819, t_19144, t_19312, t_22379, ...
            weight=self.weight,                                                # trace_info : t_18652, t_18820, t_19145, t_19313, t_22380, ...
            bias=None,                                                         # trace_info : t_18653, t_18821, t_19146, t_19314, t_22381, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_18654, t_18822, t_19147, t_19315, t_22382, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_18655, t_18823, t_19148, t_19316, t_22383, ...
            sequence_parallel=False,                                           # trace_info : t_18656, t_18824, t_19149, t_19317, t_22384, ...
            grad_output_buffer=None,                                           # trace_info : t_18657, t_18825, t_19150, t_19318, t_22385, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_18658, t_18826, t_19151, t_19319, t_22386, ...
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:                                          # trace_info : t_18683, t_18851, t_19176, t_19344, t_22411, ...
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:                                           # trace_info : t_18684, t_18852, t_19177, t_19345, t_22412, ...
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)# trace_info : t_18685, t_18853, t_19178, t_19346, t_22413, ...
        if not self.skip_bias_add:                                             # trace_info : t_18699, t_18867, t_19192, t_19360, t_22427, ...
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_                                                   # trace_info : t_18700, t_18868, t_19193, t_19361, t_22428, ...
            output_bias = self.bias                                            # trace_info : t_18701, t_18869, t_19194, t_19362, t_22429, ...
        return output, output_bias                                             # trace_info : t_18702, t_18870, t_19195, t_19363, t_22430, ...

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None
