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
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (# trace_info : t_17157, t_17166, t_17173, t_17182, t_17191, ...
        get_tensor_model_parallel_rank() == 0                                  # trace_info : t_17167, t_17255, t_17271, t_17287, t_17312, ...
    )


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_6486, t_6488, t_6490, t_6492, t_6897, ...
        assert not hasattr(tensor, attribute)                                  # trace_info : t_6487, t_6489, t_6491, t_6898, t_6900, ...
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)                      # trace_info : t_6493, t_6904, t_7046, t_7103, t_7375, ...
    setattr(tensor, 'partition_dim', dim)                                      # trace_info : t_6494, t_6905, t_7047, t_7104, t_7376, ...
    setattr(tensor, 'partition_stride', stride)                                # trace_info : t_6495, t_6906, t_7048, t_7105, t_7377, ...


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):                                           # trace_info : t_8701, t_8714, t_8730, t_8746, t_8762, ...
        if not hasattr(tensor, attribute):                                     # trace_info : t_8704, t_8707, t_8710, t_8717, t_8721, ...
            setattr(tensor, attribute, value)                                  # trace_info : t_8718, t_8722, t_8726, t_8734, t_8738, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_8702, t_8705, t_8708, t_8711, t_8715, ...
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])    # trace_info : t_8703, t_8706, t_8709, t_8716, t_8720, ...


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):                                                 # trace_info : t_12131, t_12155, t_12179, t_12203, t_12227, ...
        if hasattr(source_tensor, attribute):                                  # trace_info : t_12134, t_12138, t_12142, t_12158, t_12162, ...
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))# trace_info : t_12135, t_12139, t_12143, t_12159, t_12163, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_12132, t_12136, t_12140, t_12144, t_12156, ...
        maybe_copy(attribute)                                                  # trace_info : t_12133, t_12137, t_12141, t_12157, t_12161, ...


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(                                      # trace_info : t_6483, t_6485, t_6894, t_6896, t_7036, ...
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride      # trace_info : t_6484, t_6895, t_7037, t_7366, t_7524, ...
    )

    if not expert_parallel:                                                    # trace_info : t_6496, t_6907, t_7049, t_7378, t_7536, ...
        with get_cuda_rng_tracker().fork():                                    # trace_info : t_6497, t_6519, t_6908, t_6930, t_7050, ...
            init_method(weight)                                                # trace_info : t_6517, t_6928, t_7070, t_7399, t_7557, ...
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
        super(VocabParallelEmbedding, self).__init__()                         # trace_info : t_6441
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings                                   # trace_info : t_6442
        self.embedding_dim = embedding_dim                                     # trace_info : t_6443
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()# trace_info : t_6444
        # Divide the weight matrix along the vocaburaly dimension.
        (                                                                      # trace_info : t_6468
            self.vocab_start_index,                                            # trace_info : t_6469
            self.vocab_end_index,                                              # trace_info : t_6470
        ) = VocabUtility.vocab_range_from_global_vocab_size(                   # trace_info : t_6450, t_6457
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size# trace_info : t_6451
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index# trace_info : t_6471

        # Allocate weights and initialize.
        if config.use_cpu_initialization:                                      # trace_info : t_6472
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
            self.weight = Parameter(                                           # trace_info : t_6473, t_6480
                torch.empty(                                                   # trace_info : t_6474, t_6479
                    self.num_embeddings_per_partition,                         # trace_info : t_6475
                    self.embedding_dim,                                        # trace_info : t_6476
                    device=torch.cuda.current_device(),                        # trace_info : t_6477
                    dtype=config.params_dtype,                                 # trace_info : t_6478
                )
            )
            if config.perform_initialization:                                  # trace_info : t_6481
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)# trace_info : t_6482

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_15211, t_18852, t_22491
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_                                              # trace_info : t_15212, t_18853, t_22492
        # Get the embeddings.
        output_parallel = self.weight[masked_input]                            # trace_info : t_15213, t_18854, t_22493
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_15214, t_18855, t_22494
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)     # trace_info : t_15215, t_18856, t_22495
        return output                                                          # trace_info : t_15225, t_18866, t_22505

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
        ctx.save_for_backward(input, weight)                                   # trace_info : t_15344, t_15541, t_15661, t_15713, t_15847, ...
        ctx.use_bias = bias is not None                                        # trace_info : t_15345, t_15542, t_15662, t_15714, t_15848, ...
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion        # trace_info : t_15346, t_15543, t_15663, t_15715, t_15849, ...
        ctx.allreduce_dgrad = allreduce_dgrad                                  # trace_info : t_15347, t_15544, t_15664, t_15716, t_15850, ...
        ctx.sequence_parallel = sequence_parallel                              # trace_info : t_15348, t_15545, t_15665, t_15717, t_15851, ...
        ctx.grad_output_buffer = grad_output_buffer                            # trace_info : t_15349, t_15546, t_15666, t_15718, t_15852, ...

        if sequence_parallel:                                                  # trace_info : t_15350, t_15547, t_15667, t_15719, t_15853, ...
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input                                                # trace_info : t_15351, t_15548, t_15668, t_15720, t_15854, ...

        output = torch.matmul(total_input, weight.t())                         # trace_info : t_15352, t_15549, t_15669, t_15721, t_15855, ...
        if bias is not None:                                                   # trace_info : t_15353, t_15550, t_15670, t_15722, t_15856, ...
            output = output + bias                                             # trace_info : t_15354, t_15857, t_18995, t_19496, t_22634, ...
        return output                                                          # trace_info : t_15355, t_15551, t_15671, t_15723, t_15858, ...

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
    if allreduce_dgrad is None:                                                # trace_info : t_15332, t_15529, t_15649, t_15701, t_15835, ...
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [                                                                   # trace_info : t_15340, t_15537, t_15657, t_15709, t_15843, ...
        input,                                                                 # trace_info : t_15333, t_15530, t_15650, t_15702, t_15836, ...
        weight,                                                                # trace_info : t_15334, t_15531, t_15651, t_15703, t_15837, ...
        bias,                                                                  # trace_info : t_15335, t_15532, t_15652, t_15704, t_15838, ...
        gradient_accumulation_fusion,                                          # trace_info : t_15336, t_15533, t_15653, t_15705, t_15839, ...
        allreduce_dgrad,                                                       # trace_info : t_15337, t_15534, t_15654, t_15706, t_15840, ...
        sequence_parallel,                                                     # trace_info : t_15338, t_15535, t_15655, t_15707, t_15841, ...
        grad_output_buffer,                                                    # trace_info : t_15339, t_15536, t_15656, t_15708, t_15842, ...
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:           # trace_info : t_15341, t_15538, t_15658, t_15710, t_15844, ...
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":               # trace_info : t_15342, t_15539, t_15659, t_15711, t_15845, ...
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

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)        # trace_info : t_15343, t_15540, t_15660, t_15712, t_15846, ...


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
        super(ColumnParallelLinear, self).__init__()                           # trace_info : t_6989, t_7318, t_7991, t_8320, t_8640

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_6990, t_7319, t_7992, t_8321, t_8641
        self.output_size = output_size                                         # trace_info : t_6991, t_7320, t_7993, t_8322, t_8642
        self.gather_output = gather_output                                     # trace_info : t_6992, t_7321, t_7994, t_8323, t_8643
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_6993, t_7322, t_7995, t_8324, t_8644
        self.is_expert = is_expert                                             # trace_info : t_6994, t_7323, t_7996, t_8325, t_8645
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_6995, t_7324, t_7997, t_8326, t_8646
        self.embedding_activation_buffer = embedding_activation_buffer         # trace_info : t_6996, t_7325, t_7998, t_8327, t_8647
        self.grad_output_buffer = grad_output_buffer                           # trace_info : t_6997, t_7326, t_7999, t_8328, t_8648
        self.config = config                                                   # trace_info : t_6998, t_7327, t_8000, t_8329, t_8649
        self.disable_grad_reduce = disable_grad_reduce                         # trace_info : t_6999, t_7328, t_8001, t_8330, t_8650

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_7000, t_7329, t_8002, t_8331, t_8651
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_7001, t_7330, t_8003, t_8332, t_8652
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_7002, t_7331, t_8004, t_8333, t_8653
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_7008, t_7337, t_8010, t_8339, t_8659

        self.output_size_per_partition = divide(output_size, world_size)       # trace_info : t_7014, t_7343, t_8016, t_8345, t_8665

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:                                   # trace_info : t_7018, t_7347, t_8020, t_8349, t_8669
            if config.use_cpu_initialization:                                  # trace_info : t_7019, t_7348, t_8021, t_8350
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
                self.weight = Parameter(                                       # trace_info : t_7020, t_7027, t_7349, t_7356, t_8022, ...
                    torch.empty(                                               # trace_info : t_7021, t_7026, t_7350, t_7355, t_8023, ...
                        self.output_size_per_partition,                        # trace_info : t_7022, t_7351, t_8024, t_8353
                        self.input_size,                                       # trace_info : t_7023, t_7352, t_8025, t_8354
                        device=torch.cuda.current_device(),                    # trace_info : t_7024, t_7353, t_8026, t_8355
                        dtype=config.params_dtype,                             # trace_info : t_7025, t_7354, t_8027, t_8356
                    )
                )
                if config.perform_initialization:                              # trace_info : t_7028, t_7357, t_8030, t_8359
                    _initialize_affine_weight_gpu(                             # trace_info : t_7029, t_7035, t_7358, t_7364, t_8031, ...
                        self.weight,                                           # trace_info : t_7030, t_7359, t_8032, t_8361
                        init_method,                                           # trace_info : t_7031, t_7360, t_8033, t_8362
                        partition_dim=0,                                       # trace_info : t_7032, t_7361, t_8034, t_8363
                        stride=stride,                                         # trace_info : t_7033, t_7362, t_8035, t_8364
                        expert_parallel=(self.is_expert and self.expert_parallel),# trace_info : t_7034, t_7363, t_8036, t_8365
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_7085, t_7414, t_8087, t_8416
        else:
            self.weight = None                                                 # trace_info : t_8670

        if bias:                                                               # trace_info : t_7086, t_7415, t_8088, t_8417, t_8671
            if config.use_cpu_initialization:                                  # trace_info : t_7087, t_7416, t_8089, t_8418
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(                                         # trace_info : t_7088, t_7094, t_7417, t_7423, t_8090, ...
                    torch.empty(                                               # trace_info : t_7089, t_7093, t_7418, t_7422, t_8091, ...
                        self.output_size_per_partition,                        # trace_info : t_7090, t_7419, t_8092, t_8421
                        device=torch.cuda.current_device(),                    # trace_info : t_7091, t_7420, t_8093, t_8422
                        dtype=config.params_dtype,                             # trace_info : t_7092, t_7421, t_8094, t_8423
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)   # trace_info : t_7095, t_7424, t_8097, t_8426
            if config.perform_initialization:                                  # trace_info : t_7106, t_7435, t_8108, t_8437
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_7107, t_7109, t_7436, t_7438, t_8109, ...
                    self.bias.zero_()                                          # trace_info : t_7108, t_7437, t_8110, t_8439
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_7110, t_7439, t_8112, t_8441
        else:
            self.register_parameter('bias', None)                              # trace_info : t_8672

        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_7111, t_7440, t_8113, t_8442, t_8673
        if self.sequence_parallel and world_size <= 1:                         # trace_info : t_7112, t_7441, t_8114, t_8443, t_8674
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel   # trace_info : t_7113, t_7442, t_8115, t_8444, t_8675

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:# trace_info : t_7114, t_7443, t_8116, t_8445, t_8676
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_7115, t_7444, t_8117, t_8446, t_8677

        if self.allreduce_dgrad and self.sequence_parallel:                    # trace_info : t_7116, t_7445, t_8118, t_8447, t_8678
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_7117, t_7446, t_8119, t_8448, t_8679

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_7118, t_7120, t_7447, t_7449, t_8120, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_7119, t_7448, t_8121, t_8450, t_8681
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
        if weight is None:                                                     # trace_info : t_15300, t_15617, t_15803, t_16118, t_16275, ...
            if self.weight is None:                                            # trace_info : t_15301, t_15618, t_15804, t_16119, t_18942, ...
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight                                               # trace_info : t_15302, t_15619, t_15805, t_16120, t_18943, ...
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size) # trace_info : t_16276, t_19915, t_23554
            if weight.shape != expected_shape:                                 # trace_info : t_16277, t_19916, t_23555
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_15303, t_15620, t_15806, t_16121, t_16278, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None                   # trace_info : t_15304, t_15621, t_15807, t_16122, t_16279, ...

        if (                                                                   # trace_info : t_15306, t_15308, t_15310, t_15312, t_15623, ...
            self.allreduce_dgrad                                               # trace_info : t_15305, t_15622, t_15808, t_16123, t_16280, ...
            or self.sequence_parallel                                          # trace_info : t_15307, t_15624, t_15810, t_16125, t_16282, ...
            or self.explicit_expert_comm                                       # trace_info : t_15309, t_15626, t_15812, t_16127, t_16284, ...
            or self.disable_grad_reduce                                        # trace_info : t_15311, t_15628, t_15814, t_16129, t_16286, ...
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)      # trace_info : t_15313, t_15630, t_15816, t_16131, t_16288, ...

        if self.config.defer_embedding_wgrad_compute:                          # trace_info : t_15316, t_15633, t_15819, t_16134, t_16291, ...
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:                                           # trace_info : t_15317, t_15634, t_15820, t_16135, t_16292, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_15318, t_15635, t_15821, t_16136, t_16293, ...

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad# trace_info : t_15319, t_15636, t_15822, t_16137, t_16294, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_15320, t_15331, t_15637, t_15648, t_15823, ...
            input=input_parallel,                                              # trace_info : t_15321, t_15638, t_15824, t_16139, t_16296, ...
            weight=weight,                                                     # trace_info : t_15322, t_15639, t_15825, t_16140, t_16297, ...
            bias=bias,                                                         # trace_info : t_15323, t_15640, t_15826, t_16141, t_16298, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_15324, t_15641, t_15827, t_16142, t_16299, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_15325, t_15642, t_15828, t_16143, t_16300, ...
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,# trace_info : t_15326, t_15643, t_15829, t_16144, t_16301, ...
            grad_output_buffer=self.grad_output_buffer                         # trace_info : t_15328, t_15645, t_15831, t_16146, t_16303, ...
            if self.config.defer_embedding_wgrad_compute                       # trace_info : t_15327, t_15644, t_15830, t_16145, t_16302, ...
            else None,                                                         # trace_info : t_15329, t_15646, t_15832, t_16147, t_16304, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_15330, t_15647, t_15833, t_16148, t_16305, ...
        )
        if self.gather_output:                                                 # trace_info : t_15356, t_15672, t_15859, t_16173, t_16330, ...
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel                                           # trace_info : t_15357, t_15673, t_15860, t_16174, t_16331, ...
        output_bias = self.bias if self.skip_bias_add else None                # trace_info : t_15358, t_15674, t_15861, t_16175, t_16332, ...
        return output, output_bias                                             # trace_info : t_15359, t_15675, t_15862, t_16176, t_16333, ...

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
        super(RowParallelLinear, self).__init__()                              # trace_info : t_6848, t_7477, t_7850, t_8479

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_6849, t_7478, t_7851, t_8480
        self.output_size = output_size                                         # trace_info : t_6850, t_7479, t_7852, t_8481
        self.input_is_parallel = input_is_parallel                             # trace_info : t_6851, t_7480, t_7853, t_8482
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_6852, t_7481, t_7854, t_8483
        self.config = config                                                   # trace_info : t_6853, t_7482, t_7855, t_8484
        self.is_expert = is_expert                                             # trace_info : t_6854, t_7483, t_7856, t_8485
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_6855, t_7484, t_7857, t_8486
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_6856, t_7485, t_7858, t_8487
        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_6857, t_7486, t_7859, t_8488
        if self.sequence_parallel and not self.input_is_parallel:              # trace_info : t_6858, t_7487, t_7860, t_8489
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_6859, t_7488, t_7861, t_8490
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_6860, t_7489, t_7862, t_8491
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_6861, t_7490, t_7863, t_8492
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_6867, t_7496, t_7869, t_8498

        self.input_size_per_partition = divide(input_size, world_size)         # trace_info : t_6873, t_7502, t_7875, t_8504

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:                                      # trace_info : t_6877, t_7506, t_7879, t_8508
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
            self.weight = Parameter(                                           # trace_info : t_6878, t_6885, t_7507, t_7514, t_7880, ...
                torch.empty(                                                   # trace_info : t_6879, t_6884, t_7508, t_7513, t_7881, ...
                    self.output_size,                                          # trace_info : t_6880, t_7509, t_7882, t_8511
                    self.input_size_per_partition,                             # trace_info : t_6881, t_7510, t_7883, t_8512
                    device=torch.cuda.current_device(),                        # trace_info : t_6882, t_7511, t_7884, t_8513
                    dtype=config.params_dtype,                                 # trace_info : t_6883, t_7512, t_7885, t_8514
                )
            )
            if config.perform_initialization:                                  # trace_info : t_6886, t_7515, t_7888, t_8517
                _initialize_affine_weight_gpu(                                 # trace_info : t_6887, t_6893, t_7516, t_7522, t_7889, ...
                    self.weight,                                               # trace_info : t_6888, t_7517, t_7890, t_8519
                    init_method,                                               # trace_info : t_6889, t_7518, t_7891, t_8520
                    partition_dim=1,                                           # trace_info : t_6890, t_7519, t_7892, t_8521
                    stride=stride,                                             # trace_info : t_6891, t_7520, t_7893, t_8522
                    expert_parallel=(self.is_expert and self.expert_parallel), # trace_info : t_6892, t_7521, t_7894, t_8523
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_6943, t_7572, t_7945, t_8574

        if bias:                                                               # trace_info : t_6944, t_7573, t_7946, t_8575
            if config.use_cpu_initialization:                                  # trace_info : t_6945, t_7574, t_7947, t_8576
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(                                         # trace_info : t_6946, t_6952, t_7575, t_7581, t_7948, ...
                    torch.empty(                                               # trace_info : t_6947, t_6951, t_7576, t_7580, t_7949, ...
                        self.output_size,                                      # trace_info : t_6948, t_7577, t_7950, t_8579
                        device=torch.cuda.current_device(),                    # trace_info : t_6949, t_7578, t_7951, t_8580
                        dtype=config.params_dtype,                             # trace_info : t_6950, t_7579, t_7952, t_8581
                    )
                )

            if config.perform_initialization:                                  # trace_info : t_6953, t_7582, t_7955, t_8584
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_6954, t_6956, t_7583, t_7585, t_7956, ...
                    self.bias.zero_()                                          # trace_info : t_6955, t_7584, t_7957, t_8586
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_6957, t_7586, t_7959, t_8588
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)    # trace_info : t_6958, t_7587, t_7960, t_8589
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_6959, t_7588, t_7961, t_8590

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_6960, t_6962, t_7589, t_7591, t_7962, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_6961, t_7590, t_7963, t_8592
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

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_15513, t_15685, t_16014, t_16186, t_19152, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:                                             # trace_info : t_15514, t_15686, t_16015, t_16187, t_19153, ...
            input_parallel = input_                                            # trace_info : t_15515, t_15687, t_16016, t_16188, t_19154, ...
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if not self.weight.requires_grad:                                      # trace_info : t_15516, t_15688, t_16017, t_16189, t_19155, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_15517, t_15689, t_16018, t_16190, t_19156, ...

        allreduce_dgrad = False                                                # trace_info : t_15518, t_15690, t_16019, t_16191, t_19157, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_15519, t_15528, t_15691, t_15700, t_16020, ...
            input=input_parallel,                                              # trace_info : t_15520, t_15692, t_16021, t_16193, t_19159, ...
            weight=self.weight,                                                # trace_info : t_15521, t_15693, t_16022, t_16194, t_19160, ...
            bias=None,                                                         # trace_info : t_15522, t_15694, t_16023, t_16195, t_19161, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_15523, t_15695, t_16024, t_16196, t_19162, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_15524, t_15696, t_16025, t_16197, t_19163, ...
            sequence_parallel=False,                                           # trace_info : t_15525, t_15697, t_16026, t_16198, t_19164, ...
            grad_output_buffer=None,                                           # trace_info : t_15526, t_15698, t_16027, t_16199, t_19165, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_15527, t_15699, t_16028, t_16200, t_19166, ...
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:                                          # trace_info : t_15552, t_15724, t_16053, t_16225, t_19191, ...
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:                                           # trace_info : t_15553, t_15725, t_16054, t_16226, t_19192, ...
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)# trace_info : t_15554, t_15726, t_16055, t_16227, t_19193, ...
        if not self.skip_bias_add:                                             # trace_info : t_15564, t_15736, t_16065, t_16237, t_19203, ...
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_                                                   # trace_info : t_15565, t_15737, t_16066, t_16238, t_19204, ...
            output_bias = self.bias                                            # trace_info : t_15566, t_15738, t_16067, t_16239, t_19205, ...
        return output, output_bias                                             # trace_info : t_15567, t_15739, t_16068, t_16240, t_19206, ...

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
