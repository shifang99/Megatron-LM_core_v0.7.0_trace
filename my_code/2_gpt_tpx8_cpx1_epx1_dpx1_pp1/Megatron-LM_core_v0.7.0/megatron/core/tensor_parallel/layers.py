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
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (# trace_info : t_21994, t_22003, t_22010, t_22019, t_22028, ...
        get_tensor_model_parallel_rank() == 0                                  # trace_info : t_22004, t_22092, t_22108, t_22124, t_22149, ...
    )


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_11353, t_11355, t_11357, t_11359, t_11764, ...
        assert not hasattr(tensor, attribute)                                  # trace_info : t_11354, t_11356, t_11358, t_11765, t_11767, ...
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)                      # trace_info : t_11360, t_11771, t_11913, t_11970, t_12242, ...
    setattr(tensor, 'partition_dim', dim)                                      # trace_info : t_11361, t_11772, t_11914, t_11971, t_12243, ...
    setattr(tensor, 'partition_stride', stride)                                # trace_info : t_11362, t_11773, t_11915, t_11972, t_12244, ...


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):                                           # trace_info : t_13568, t_13581, t_13597, t_13613, t_13629, ...
        if not hasattr(tensor, attribute):                                     # trace_info : t_13571, t_13574, t_13577, t_13584, t_13588, ...
            setattr(tensor, attribute, value)                                  # trace_info : t_13585, t_13589, t_13593, t_13601, t_13605, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_13569, t_13572, t_13575, t_13578, t_13582, ...
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])    # trace_info : t_13570, t_13573, t_13576, t_13583, t_13587, ...


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):                                                 # trace_info : t_16998, t_17022, t_17046, t_17070, t_17094, ...
        if hasattr(source_tensor, attribute):                                  # trace_info : t_17001, t_17005, t_17009, t_17025, t_17029, ...
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))# trace_info : t_17002, t_17006, t_17010, t_17026, t_17030, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_16999, t_17003, t_17007, t_17011, t_17023, ...
        maybe_copy(attribute)                                                  # trace_info : t_17000, t_17004, t_17008, t_17024, t_17028, ...


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(                                      # trace_info : t_11350, t_11352, t_11761, t_11763, t_11903, ...
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride      # trace_info : t_11351, t_11762, t_11904, t_12233, t_12391, ...
    )

    if not expert_parallel:                                                    # trace_info : t_11363, t_11774, t_11916, t_12245, t_12403, ...
        with get_cuda_rng_tracker().fork():                                    # trace_info : t_11364, t_11386, t_11775, t_11797, t_11917, ...
            init_method(weight)                                                # trace_info : t_11384, t_11795, t_11937, t_12266, t_12424, ...
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
        super(VocabParallelEmbedding, self).__init__()                         # trace_info : t_11308
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings                                   # trace_info : t_11309
        self.embedding_dim = embedding_dim                                     # trace_info : t_11310
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()# trace_info : t_11311
        # Divide the weight matrix along the vocaburaly dimension.
        (                                                                      # trace_info : t_11335
            self.vocab_start_index,                                            # trace_info : t_11336
            self.vocab_end_index,                                              # trace_info : t_11337
        ) = VocabUtility.vocab_range_from_global_vocab_size(                   # trace_info : t_11317, t_11324
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size# trace_info : t_11318
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index# trace_info : t_11338

        # Allocate weights and initialize.
        if config.use_cpu_initialization:                                      # trace_info : t_11339
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
            self.weight = Parameter(                                           # trace_info : t_11340, t_11347
                torch.empty(                                                   # trace_info : t_11341, t_11346
                    self.num_embeddings_per_partition,                         # trace_info : t_11342
                    self.embedding_dim,                                        # trace_info : t_11343
                    device=torch.cuda.current_device(),                        # trace_info : t_11344
                    dtype=config.params_dtype,                                 # trace_info : t_11345
                )
            )
            if config.perform_initialization:                                  # trace_info : t_11348
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)# trace_info : t_11349

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_20077, t_23689, t_27299
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)# trace_info : t_20078, t_23690, t_27300
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index             # trace_info : t_20079, t_23691, t_27301
            masked_input[input_mask] = 0                                       # trace_info : t_20080, t_23692, t_27302
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.weight[masked_input]                            # trace_info : t_20081, t_23693, t_27303
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_20082, t_23694, t_27304
            output_parallel[input_mask, :] = 0.0                               # trace_info : t_20083, t_23695, t_27305
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)     # trace_info : t_20084, t_23696, t_27306
        return output                                                          # trace_info : t_20098, t_23710, t_27320

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
        ctx.save_for_backward(input, weight)                                   # trace_info : t_20209, t_20398, t_20514, t_20566, t_20696, ...
        ctx.use_bias = bias is not None                                        # trace_info : t_20210, t_20399, t_20515, t_20567, t_20697, ...
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion        # trace_info : t_20211, t_20400, t_20516, t_20568, t_20698, ...
        ctx.allreduce_dgrad = allreduce_dgrad                                  # trace_info : t_20212, t_20401, t_20517, t_20569, t_20699, ...
        ctx.sequence_parallel = sequence_parallel                              # trace_info : t_20213, t_20402, t_20518, t_20570, t_20700, ...
        ctx.grad_output_buffer = grad_output_buffer                            # trace_info : t_20214, t_20403, t_20519, t_20571, t_20701, ...

        if sequence_parallel:                                                  # trace_info : t_20215, t_20404, t_20520, t_20572, t_20702, ...
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input                                                # trace_info : t_20216, t_20405, t_20521, t_20573, t_20703, ...

        output = torch.matmul(total_input, weight.t())                         # trace_info : t_20217, t_20406, t_20522, t_20574, t_20704, ...
        if bias is not None:                                                   # trace_info : t_20218, t_20407, t_20523, t_20575, t_20705, ...
            output = output + bias                                             # trace_info : t_20219, t_20706, t_23831, t_24316, t_27441, ...
        return output                                                          # trace_info : t_20220, t_20408, t_20524, t_20576, t_20707, ...

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
    if allreduce_dgrad is None:                                                # trace_info : t_20197, t_20386, t_20502, t_20554, t_20684, ...
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [                                                                   # trace_info : t_20205, t_20394, t_20510, t_20562, t_20692, ...
        input,                                                                 # trace_info : t_20198, t_20387, t_20503, t_20555, t_20685, ...
        weight,                                                                # trace_info : t_20199, t_20388, t_20504, t_20556, t_20686, ...
        bias,                                                                  # trace_info : t_20200, t_20389, t_20505, t_20557, t_20687, ...
        gradient_accumulation_fusion,                                          # trace_info : t_20201, t_20390, t_20506, t_20558, t_20688, ...
        allreduce_dgrad,                                                       # trace_info : t_20202, t_20391, t_20507, t_20559, t_20689, ...
        sequence_parallel,                                                     # trace_info : t_20203, t_20392, t_20508, t_20560, t_20690, ...
        grad_output_buffer,                                                    # trace_info : t_20204, t_20393, t_20509, t_20561, t_20691, ...
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:           # trace_info : t_20206, t_20395, t_20511, t_20563, t_20693, ...
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":               # trace_info : t_20207, t_20396, t_20512, t_20564, t_20694, ...
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

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)        # trace_info : t_20208, t_20397, t_20513, t_20565, t_20695, ...


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
        super(ColumnParallelLinear, self).__init__()                           # trace_info : t_11856, t_12185, t_12858, t_13187, t_13507

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_11857, t_12186, t_12859, t_13188, t_13508
        self.output_size = output_size                                         # trace_info : t_11858, t_12187, t_12860, t_13189, t_13509
        self.gather_output = gather_output                                     # trace_info : t_11859, t_12188, t_12861, t_13190, t_13510
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_11860, t_12189, t_12862, t_13191, t_13511
        self.is_expert = is_expert                                             # trace_info : t_11861, t_12190, t_12863, t_13192, t_13512
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_11862, t_12191, t_12864, t_13193, t_13513
        self.embedding_activation_buffer = embedding_activation_buffer         # trace_info : t_11863, t_12192, t_12865, t_13194, t_13514
        self.grad_output_buffer = grad_output_buffer                           # trace_info : t_11864, t_12193, t_12866, t_13195, t_13515
        self.config = config                                                   # trace_info : t_11865, t_12194, t_12867, t_13196, t_13516
        self.disable_grad_reduce = disable_grad_reduce                         # trace_info : t_11866, t_12195, t_12868, t_13197, t_13517

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_11867, t_12196, t_12869, t_13198, t_13518
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_11868, t_12197, t_12870, t_13199, t_13519
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_11869, t_12198, t_12871, t_13200, t_13520
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_11875, t_12204, t_12877, t_13206, t_13526

        self.output_size_per_partition = divide(output_size, world_size)       # trace_info : t_11881, t_12210, t_12883, t_13212, t_13532

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:                                   # trace_info : t_11885, t_12214, t_12887, t_13216, t_13536
            if config.use_cpu_initialization:                                  # trace_info : t_11886, t_12215, t_12888, t_13217
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
                self.weight = Parameter(                                       # trace_info : t_11887, t_11894, t_12216, t_12223, t_12889, ...
                    torch.empty(                                               # trace_info : t_11888, t_11893, t_12217, t_12222, t_12890, ...
                        self.output_size_per_partition,                        # trace_info : t_11889, t_12218, t_12891, t_13220
                        self.input_size,                                       # trace_info : t_11890, t_12219, t_12892, t_13221
                        device=torch.cuda.current_device(),                    # trace_info : t_11891, t_12220, t_12893, t_13222
                        dtype=config.params_dtype,                             # trace_info : t_11892, t_12221, t_12894, t_13223
                    )
                )
                if config.perform_initialization:                              # trace_info : t_11895, t_12224, t_12897, t_13226
                    _initialize_affine_weight_gpu(                             # trace_info : t_11896, t_11902, t_12225, t_12231, t_12898, ...
                        self.weight,                                           # trace_info : t_11897, t_12226, t_12899, t_13228
                        init_method,                                           # trace_info : t_11898, t_12227, t_12900, t_13229
                        partition_dim=0,                                       # trace_info : t_11899, t_12228, t_12901, t_13230
                        stride=stride,                                         # trace_info : t_11900, t_12229, t_12902, t_13231
                        expert_parallel=(self.is_expert and self.expert_parallel),# trace_info : t_11901, t_12230, t_12903, t_13232
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_11952, t_12281, t_12954, t_13283
        else:
            self.weight = None                                                 # trace_info : t_13537

        if bias:                                                               # trace_info : t_11953, t_12282, t_12955, t_13284, t_13538
            if config.use_cpu_initialization:                                  # trace_info : t_11954, t_12283, t_12956, t_13285
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(                                         # trace_info : t_11955, t_11961, t_12284, t_12290, t_12957, ...
                    torch.empty(                                               # trace_info : t_11956, t_11960, t_12285, t_12289, t_12958, ...
                        self.output_size_per_partition,                        # trace_info : t_11957, t_12286, t_12959, t_13288
                        device=torch.cuda.current_device(),                    # trace_info : t_11958, t_12287, t_12960, t_13289
                        dtype=config.params_dtype,                             # trace_info : t_11959, t_12288, t_12961, t_13290
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)   # trace_info : t_11962, t_12291, t_12964, t_13293
            if config.perform_initialization:                                  # trace_info : t_11973, t_12302, t_12975, t_13304
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_11974, t_11976, t_12303, t_12305, t_12976, ...
                    self.bias.zero_()                                          # trace_info : t_11975, t_12304, t_12977, t_13306
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_11977, t_12306, t_12979, t_13308
        else:
            self.register_parameter('bias', None)                              # trace_info : t_13539

        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_11978, t_12307, t_12980, t_13309, t_13540
        if self.sequence_parallel and world_size <= 1:                         # trace_info : t_11979, t_12308, t_12981, t_13310, t_13541
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel   # trace_info : t_11980, t_12309, t_12982, t_13311, t_13542

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:# trace_info : t_11981, t_12310, t_12983, t_13312, t_13543
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_11982, t_12311, t_12984, t_13313, t_13544

        if self.allreduce_dgrad and self.sequence_parallel:                    # trace_info : t_11983, t_12312, t_12985, t_13314, t_13545
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_11984, t_12313, t_12986, t_13315, t_13546

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_11985, t_11987, t_12314, t_12316, t_12987, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_11986, t_12315, t_12988, t_13317, t_13548
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
        if weight is None:                                                     # trace_info : t_20173, t_20478, t_20660, t_20963, t_21116, ...
            if self.weight is None:                                            # trace_info : t_20174, t_20479, t_20661, t_20964, t_23786, ...
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight                                               # trace_info : t_20175, t_20480, t_20662, t_20965, t_23787, ...
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size) # trace_info : t_21117, t_24727, t_28337
            if weight.shape != expected_shape:                                 # trace_info : t_21118, t_24728, t_28338
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_20176, t_20481, t_20663, t_20966, t_21119, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None                   # trace_info : t_20177, t_20482, t_20664, t_20967, t_21120, ...

        if (                                                                   # trace_info : t_20179, t_20484, t_20666, t_20969, t_21122, ...
            self.allreduce_dgrad                                               # trace_info : t_20178, t_20483, t_20665, t_20968, t_21121, ...
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_                                            # trace_info : t_20180, t_20485, t_20667, t_20970, t_21123, ...
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:                          # trace_info : t_20181, t_20486, t_20668, t_20971, t_21124, ...
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:                                           # trace_info : t_20182, t_20487, t_20669, t_20972, t_21125, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_20183, t_20488, t_20670, t_20973, t_21126, ...

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad# trace_info : t_20184, t_20489, t_20671, t_20974, t_21127, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_20185, t_20196, t_20490, t_20501, t_20672, ...
            input=input_parallel,                                              # trace_info : t_20186, t_20491, t_20673, t_20976, t_21129, ...
            weight=weight,                                                     # trace_info : t_20187, t_20492, t_20674, t_20977, t_21130, ...
            bias=bias,                                                         # trace_info : t_20188, t_20493, t_20675, t_20978, t_21131, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_20189, t_20494, t_20676, t_20979, t_21132, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_20190, t_20495, t_20677, t_20980, t_21133, ...
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,# trace_info : t_20191, t_20496, t_20678, t_20981, t_21134, ...
            grad_output_buffer=self.grad_output_buffer                         # trace_info : t_20193, t_20498, t_20680, t_20983, t_21136, ...
            if self.config.defer_embedding_wgrad_compute                       # trace_info : t_20192, t_20497, t_20679, t_20982, t_21135, ...
            else None,                                                         # trace_info : t_20194, t_20499, t_20681, t_20984, t_21137, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_20195, t_20500, t_20682, t_20985, t_21138, ...
        )
        if self.gather_output:                                                 # trace_info : t_20221, t_20525, t_20708, t_21010, t_21163, ...
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel                                           # trace_info : t_20222, t_20526, t_20709, t_21011, t_21164, ...
        output_bias = self.bias if self.skip_bias_add else None                # trace_info : t_20223, t_20527, t_20710, t_21012, t_21165, ...
        return output, output_bias                                             # trace_info : t_20224, t_20528, t_20711, t_21013, t_21166, ...

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
        super(RowParallelLinear, self).__init__()                              # trace_info : t_11715, t_12344, t_12717, t_13346

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_11716, t_12345, t_12718, t_13347
        self.output_size = output_size                                         # trace_info : t_11717, t_12346, t_12719, t_13348
        self.input_is_parallel = input_is_parallel                             # trace_info : t_11718, t_12347, t_12720, t_13349
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_11719, t_12348, t_12721, t_13350
        self.config = config                                                   # trace_info : t_11720, t_12349, t_12722, t_13351
        self.is_expert = is_expert                                             # trace_info : t_11721, t_12350, t_12723, t_13352
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_11722, t_12351, t_12724, t_13353
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_11723, t_12352, t_12725, t_13354
        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_11724, t_12353, t_12726, t_13355
        if self.sequence_parallel and not self.input_is_parallel:              # trace_info : t_11725, t_12354, t_12727, t_13356
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_11726, t_12355, t_12728, t_13357
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_11727, t_12356, t_12729, t_13358
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_11728, t_12357, t_12730, t_13359
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_11734, t_12363, t_12736, t_13365

        self.input_size_per_partition = divide(input_size, world_size)         # trace_info : t_11740, t_12369, t_12742, t_13371

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:                                      # trace_info : t_11744, t_12373, t_12746, t_13375
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
            self.weight = Parameter(                                           # trace_info : t_11745, t_11752, t_12374, t_12381, t_12747, ...
                torch.empty(                                                   # trace_info : t_11746, t_11751, t_12375, t_12380, t_12748, ...
                    self.output_size,                                          # trace_info : t_11747, t_12376, t_12749, t_13378
                    self.input_size_per_partition,                             # trace_info : t_11748, t_12377, t_12750, t_13379
                    device=torch.cuda.current_device(),                        # trace_info : t_11749, t_12378, t_12751, t_13380
                    dtype=config.params_dtype,                                 # trace_info : t_11750, t_12379, t_12752, t_13381
                )
            )
            if config.perform_initialization:                                  # trace_info : t_11753, t_12382, t_12755, t_13384
                _initialize_affine_weight_gpu(                                 # trace_info : t_11754, t_11760, t_12383, t_12389, t_12756, ...
                    self.weight,                                               # trace_info : t_11755, t_12384, t_12757, t_13386
                    init_method,                                               # trace_info : t_11756, t_12385, t_12758, t_13387
                    partition_dim=1,                                           # trace_info : t_11757, t_12386, t_12759, t_13388
                    stride=stride,                                             # trace_info : t_11758, t_12387, t_12760, t_13389
                    expert_parallel=(self.is_expert and self.expert_parallel), # trace_info : t_11759, t_12388, t_12761, t_13390
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_11810, t_12439, t_12812, t_13441

        if bias:                                                               # trace_info : t_11811, t_12440, t_12813, t_13442
            if config.use_cpu_initialization:                                  # trace_info : t_11812, t_12441, t_12814, t_13443
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(                                         # trace_info : t_11813, t_11819, t_12442, t_12448, t_12815, ...
                    torch.empty(                                               # trace_info : t_11814, t_11818, t_12443, t_12447, t_12816, ...
                        self.output_size,                                      # trace_info : t_11815, t_12444, t_12817, t_13446
                        device=torch.cuda.current_device(),                    # trace_info : t_11816, t_12445, t_12818, t_13447
                        dtype=config.params_dtype,                             # trace_info : t_11817, t_12446, t_12819, t_13448
                    )
                )

            if config.perform_initialization:                                  # trace_info : t_11820, t_12449, t_12822, t_13451
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_11821, t_11823, t_12450, t_12452, t_12823, ...
                    self.bias.zero_()                                          # trace_info : t_11822, t_12451, t_12824, t_13453
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_11824, t_12453, t_12826, t_13455
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)    # trace_info : t_11825, t_12454, t_12827, t_13456
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_11826, t_12455, t_12828, t_13457

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_11827, t_11829, t_12456, t_12458, t_12829, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_11828, t_12457, t_12830, t_13459
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

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_20370, t_20538, t_20855, t_21023, t_23980, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:                                             # trace_info : t_20371, t_20539, t_20856, t_21024, t_23981, ...
            input_parallel = input_                                            # trace_info : t_20372, t_20540, t_20857, t_21025, t_23982, ...
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if not self.weight.requires_grad:                                      # trace_info : t_20373, t_20541, t_20858, t_21026, t_23983, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_20374, t_20542, t_20859, t_21027, t_23984, ...

        allreduce_dgrad = False                                                # trace_info : t_20375, t_20543, t_20860, t_21028, t_23985, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_20376, t_20385, t_20544, t_20553, t_20861, ...
            input=input_parallel,                                              # trace_info : t_20377, t_20545, t_20862, t_21030, t_23987, ...
            weight=self.weight,                                                # trace_info : t_20378, t_20546, t_20863, t_21031, t_23988, ...
            bias=None,                                                         # trace_info : t_20379, t_20547, t_20864, t_21032, t_23989, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_20380, t_20548, t_20865, t_21033, t_23990, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_20381, t_20549, t_20866, t_21034, t_23991, ...
            sequence_parallel=False,                                           # trace_info : t_20382, t_20550, t_20867, t_21035, t_23992, ...
            grad_output_buffer=None,                                           # trace_info : t_20383, t_20551, t_20868, t_21036, t_23993, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_20384, t_20552, t_20869, t_21037, t_23994, ...
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:                                          # trace_info : t_20409, t_20577, t_20894, t_21062, t_24019, ...
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:                                           # trace_info : t_20410, t_20578, t_20895, t_21063, t_24020, ...
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)# trace_info : t_20411, t_20579, t_20896, t_21064, t_24021, ...
        if not self.skip_bias_add:                                             # trace_info : t_20425, t_20593, t_20910, t_21078, t_24035, ...
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_                                                   # trace_info : t_20426, t_20594, t_20911, t_21079, t_24036, ...
            output_bias = self.bias                                            # trace_info : t_20427, t_20595, t_20912, t_21080, t_24037, ...
        return output, output_bias                                             # trace_info : t_20428, t_20596, t_20913, t_21081, t_24038, ...

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
