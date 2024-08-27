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
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (# trace_info : t_20797, t_20806, t_20813, t_20822, t_20831, ...
        get_tensor_model_parallel_rank() == 0                                  # trace_info : t_20807, t_20841, t_20875, t_20891, t_20907, ...
    )


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_9440, t_9442, t_9444, t_9446, t_9849, ...
        assert not hasattr(tensor, attribute)                                  # trace_info : t_9441, t_9443, t_9445, t_9850, t_9852, ...
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)                      # trace_info : t_9447, t_9856, t_9998, t_10055, t_10441, ...
    setattr(tensor, 'partition_dim', dim)                                      # trace_info : t_9448, t_9857, t_9999, t_10056, t_10442, ...
    setattr(tensor, 'partition_stride', stride)                                # trace_info : t_9449, t_9858, t_10000, t_10057, t_10443, ...


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):                                           # trace_info : t_11931, t_11944, t_11960, t_11976, t_11992, ...
        if not hasattr(tensor, attribute):                                     # trace_info : t_11934, t_11937, t_11940, t_11947, t_11951, ...
            setattr(tensor, attribute, value)                                  # trace_info : t_11948, t_11952, t_11956, t_11964, t_11968, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_11932, t_11935, t_11938, t_11941, t_11945, ...
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])    # trace_info : t_11933, t_11936, t_11939, t_11946, t_11950, ...


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(                                      # trace_info : t_9437, t_9439, t_9846, t_9848, t_9988, ...
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride      # trace_info : t_9438, t_9847, t_9989, t_10432, t_10593, ...
    )

    if not expert_parallel:                                                    # trace_info : t_9450, t_9859, t_10001, t_10444, t_10605, ...
        with get_cuda_rng_tracker().fork():                                    # trace_info : t_9451, t_9473, t_9860, t_9882, t_10002, ...
            init_method(weight)                                                # trace_info : t_9471, t_9880, t_10022, t_11020, t_11162
    else:
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):# trace_info : t_10445, t_10468, t_10606, t_10629, t_11585, ...
            init_method(weight)                                                # trace_info : t_10466, t_10627, t_11606, t_11767


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
        super(VocabParallelEmbedding, self).__init__()                         # trace_info : t_9395
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings                                   # trace_info : t_9396
        self.embedding_dim = embedding_dim                                     # trace_info : t_9397
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()# trace_info : t_9398
        # Divide the weight matrix along the vocaburaly dimension.
        (                                                                      # trace_info : t_9422
            self.vocab_start_index,                                            # trace_info : t_9423
            self.vocab_end_index,                                              # trace_info : t_9424
        ) = VocabUtility.vocab_range_from_global_vocab_size(                   # trace_info : t_9404, t_9411
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size# trace_info : t_9405
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index# trace_info : t_9425

        # Allocate weights and initialize.
        if config.use_cpu_initialization:                                      # trace_info : t_9426
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
            self.weight = Parameter(                                           # trace_info : t_9427, t_9434
                torch.empty(                                                   # trace_info : t_9428, t_9433
                    self.num_embeddings_per_partition,                         # trace_info : t_9429
                    self.embedding_dim,                                        # trace_info : t_9430
                    device=torch.cuda.current_device(),                        # trace_info : t_9431
                    dtype=config.params_dtype,                                 # trace_info : t_9432
                )
            )
            if config.perform_initialization:                                  # trace_info : t_9435
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)# trace_info : t_9436

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18215, t_22568, t_26913
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)# trace_info : t_18216, t_22569, t_26914
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index             # trace_info : t_18217, t_22570, t_26915
            masked_input[input_mask] = 0                                       # trace_info : t_18218, t_22571, t_26916
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.weight[masked_input]                            # trace_info : t_18219, t_22572, t_26917
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18220, t_22573, t_26918
            output_parallel[input_mask, :] = 0.0                               # trace_info : t_18221, t_22574, t_26919
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)     # trace_info : t_18222, t_22575, t_26920
        return output                                                          # trace_info : t_18236, t_22589, t_26934

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
        ctx.save_for_backward(input, weight)                                   # trace_info : t_18410, t_18586, t_18900, t_18952, t_19173, ...
        ctx.use_bias = bias is not None                                        # trace_info : t_18411, t_18587, t_18901, t_18953, t_19174, ...
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion        # trace_info : t_18412, t_18588, t_18902, t_18954, t_19175, ...
        ctx.allreduce_dgrad = allreduce_dgrad                                  # trace_info : t_18413, t_18589, t_18903, t_18955, t_19176, ...
        ctx.sequence_parallel = sequence_parallel                              # trace_info : t_18414, t_18590, t_18904, t_18956, t_19177, ...
        ctx.grad_output_buffer = grad_output_buffer                            # trace_info : t_18415, t_18591, t_18905, t_18957, t_19178, ...

        if sequence_parallel:                                                  # trace_info : t_18416, t_18592, t_18906, t_18958, t_19179, ...
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_18417, t_19180, t_19906, t_22770, t_23525, ...
            dim_size = list(input.size())                                      # trace_info : t_18423, t_19186, t_19912, t_22776, t_23531, ...
            dim_size[0] = dim_size[0] * world_size                             # trace_info : t_18424, t_19187, t_19913, t_22777, t_23532, ...

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")# trace_info : t_18425, t_19188, t_19914, t_22778, t_23533, ...
            torch.distributed._all_gather_base(                                # trace_info : t_18434, t_18439, t_19195, t_19200, t_19921, ...
                all_gather_buffer, input, group=get_tensor_model_parallel_group()# trace_info : t_18435, t_19196, t_19922, t_22786, t_23541, ...
            )
            total_input = all_gather_buffer                                    # trace_info : t_18440, t_19201, t_19927, t_22791, t_23546, ...
        else:
            total_input = input                                                # trace_info : t_18593, t_18907, t_18959, t_19351, t_19662, ...

        output = torch.matmul(total_input, weight.t())                         # trace_info : t_18441, t_18594, t_18908, t_18960, t_19202, ...
        if bias is not None:                                                   # trace_info : t_18442, t_18595, t_18909, t_18961, t_19203, ...
            output = output + bias                                             # trace_info : t_18443, t_19204, t_22794, t_23549, t_27139, ...
        return output                                                          # trace_info : t_18444, t_18596, t_18910, t_18962, t_19205, ...

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
    if allreduce_dgrad is None:                                                # trace_info : t_18398, t_18574, t_18888, t_18940, t_19161, ...
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [                                                                   # trace_info : t_18406, t_18582, t_18896, t_18948, t_19169, ...
        input,                                                                 # trace_info : t_18399, t_18575, t_18889, t_18941, t_19162, ...
        weight,                                                                # trace_info : t_18400, t_18576, t_18890, t_18942, t_19163, ...
        bias,                                                                  # trace_info : t_18401, t_18577, t_18891, t_18943, t_19164, ...
        gradient_accumulation_fusion,                                          # trace_info : t_18402, t_18578, t_18892, t_18944, t_19165, ...
        allreduce_dgrad,                                                       # trace_info : t_18403, t_18579, t_18893, t_18945, t_19166, ...
        sequence_parallel,                                                     # trace_info : t_18404, t_18580, t_18894, t_18946, t_19167, ...
        grad_output_buffer,                                                    # trace_info : t_18405, t_18581, t_18895, t_18947, t_19168, ...
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:           # trace_info : t_18407, t_18583, t_18897, t_18949, t_19170, ...
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":               # trace_info : t_18408, t_18584, t_18898, t_18950, t_19171, ...
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

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)        # trace_info : t_18409, t_18585, t_18899, t_18951, t_19172, ...


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
        super(ColumnParallelLinear, self).__init__()                           # trace_info : t_9941, t_10382, t_11081, t_11522, t_11870

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_9942, t_10383, t_11082, t_11523, t_11871
        self.output_size = output_size                                         # trace_info : t_9943, t_10384, t_11083, t_11524, t_11872
        self.gather_output = gather_output                                     # trace_info : t_9944, t_10385, t_11084, t_11525, t_11873
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_9945, t_10386, t_11085, t_11526, t_11874
        self.is_expert = is_expert                                             # trace_info : t_9946, t_10387, t_11086, t_11527, t_11875
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_9947, t_10388, t_11087, t_11528, t_11876
        self.embedding_activation_buffer = embedding_activation_buffer         # trace_info : t_9948, t_10389, t_11088, t_11529, t_11877
        self.grad_output_buffer = grad_output_buffer                           # trace_info : t_9949, t_10390, t_11089, t_11530, t_11878
        self.config = config                                                   # trace_info : t_9950, t_10391, t_11090, t_11531, t_11879
        self.disable_grad_reduce = disable_grad_reduce                         # trace_info : t_9951, t_10392, t_11091, t_11532, t_11880

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_9952, t_10393, t_10395, t_11092, t_11533, ...
            config.tensor_model_parallel_size > 1 or self.expert_parallel      # trace_info : t_10394, t_11534
        )
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_9953, t_10396, t_11093, t_11536, t_11882
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_9954, t_10397, t_11094, t_11537, t_11883
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_9960, t_10403, t_11100, t_11543, t_11889

        self.output_size_per_partition = divide(output_size, world_size)       # trace_info : t_9966, t_10409, t_11106, t_11549, t_11895

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:                                   # trace_info : t_9970, t_10413, t_11110, t_11553, t_11899
            if config.use_cpu_initialization:                                  # trace_info : t_9971, t_10414, t_11111, t_11554
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
                self.weight = Parameter(                                       # trace_info : t_9972, t_9979, t_10415, t_10422, t_11112, ...
                    torch.empty(                                               # trace_info : t_9973, t_9978, t_10416, t_10421, t_11113, ...
                        self.output_size_per_partition,                        # trace_info : t_9974, t_10417, t_11114, t_11557
                        self.input_size,                                       # trace_info : t_9975, t_10418, t_11115, t_11558
                        device=torch.cuda.current_device(),                    # trace_info : t_9976, t_10419, t_11116, t_11559
                        dtype=config.params_dtype,                             # trace_info : t_9977, t_10420, t_11117, t_11560
                    )
                )
                if config.perform_initialization:                              # trace_info : t_9980, t_10423, t_11120, t_11563
                    _initialize_affine_weight_gpu(                             # trace_info : t_9981, t_9987, t_10424, t_10430, t_11121, ...
                        self.weight,                                           # trace_info : t_9982, t_10425, t_11122, t_11565
                        init_method,                                           # trace_info : t_9983, t_10426, t_11123, t_11566
                        partition_dim=0,                                       # trace_info : t_9984, t_10427, t_11124, t_11567
                        stride=stride,                                         # trace_info : t_9985, t_10428, t_11125, t_11568
                        expert_parallel=(self.is_expert and self.expert_parallel),# trace_info : t_9986, t_10429, t_11126, t_11569
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10037, t_10481, t_11177, t_11621
        else:
            self.weight = None                                                 # trace_info : t_11900

        if bias:                                                               # trace_info : t_10038, t_10482, t_11178, t_11622, t_11901
            if config.use_cpu_initialization:                                  # trace_info : t_10039, t_10483, t_11179, t_11623
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(                                         # trace_info : t_10040, t_10046, t_10484, t_10490, t_11180, ...
                    torch.empty(                                               # trace_info : t_10041, t_10045, t_10485, t_10489, t_11181, ...
                        self.output_size_per_partition,                        # trace_info : t_10042, t_10486, t_11182, t_11626
                        device=torch.cuda.current_device(),                    # trace_info : t_10043, t_10487, t_11183, t_11627
                        dtype=config.params_dtype,                             # trace_info : t_10044, t_10488, t_11184, t_11628
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)   # trace_info : t_10047, t_10491, t_11187, t_11631
            if config.perform_initialization:                                  # trace_info : t_10058, t_10502, t_11198, t_11642
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_10059, t_10061, t_10503, t_10505, t_11199, ...
                    self.bias.zero_()                                          # trace_info : t_10060, t_10504, t_11200, t_11644
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10062, t_10506, t_11202, t_11646
        else:
            self.register_parameter('bias', None)                              # trace_info : t_11902

        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_10063, t_10507, t_11203, t_11647, t_11903
        if self.sequence_parallel and world_size <= 1:                         # trace_info : t_10064, t_10508, t_11204, t_11648, t_11904
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel   # trace_info : t_10065, t_10509, t_11205, t_11649, t_11905

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:# trace_info : t_10066, t_10510, t_11206, t_11650, t_11906
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_10067, t_10511, t_11207, t_11651, t_11907

        if self.allreduce_dgrad and self.sequence_parallel:                    # trace_info : t_10068, t_10512, t_11208, t_11652, t_11908
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_10069, t_10513, t_11209, t_11653, t_11909

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_10070, t_10072, t_10514, t_10516, t_11210, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_10071, t_10515, t_11211, t_11655, t_11911
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
        if weight is None:                                                     # trace_info : t_18372, t_18862, t_19135, t_19617, t_19861, ...
            if self.weight is None:                                            # trace_info : t_18373, t_18863, t_19136, t_19618, t_22726, ...
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight                                               # trace_info : t_18374, t_18864, t_19137, t_19619, t_22727, ...
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size) # trace_info : t_19862, t_24207, t_28552
            if weight.shape != expected_shape:                                 # trace_info : t_19863, t_24208, t_28553
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_18375, t_18865, t_19138, t_19620, t_19864, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None                   # trace_info : t_18376, t_18866, t_19139, t_19621, t_19865, ...

        if (                                                                   # trace_info : t_18378, t_18380, t_18868, t_18870, t_19141, ...
            self.allreduce_dgrad                                               # trace_info : t_18377, t_18867, t_19140, t_19622, t_19866, ...
            or self.sequence_parallel                                          # trace_info : t_18379, t_18869, t_19142, t_19624, t_19868, ...
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_                                            # trace_info : t_18381, t_18871, t_19144, t_19626, t_19870, ...
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:                          # trace_info : t_18382, t_18872, t_19145, t_19627, t_19871, ...
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:                                           # trace_info : t_18383, t_18873, t_19146, t_19628, t_19872, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_18384, t_18874, t_19147, t_19629, t_19873, ...

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad# trace_info : t_18385, t_18875, t_19148, t_19630, t_19874, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_18386, t_18397, t_18876, t_18887, t_19149, ...
            input=input_parallel,                                              # trace_info : t_18387, t_18877, t_19150, t_19632, t_19876, ...
            weight=weight,                                                     # trace_info : t_18388, t_18878, t_19151, t_19633, t_19877, ...
            bias=bias,                                                         # trace_info : t_18389, t_18879, t_19152, t_19634, t_19878, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_18390, t_18880, t_19153, t_19635, t_19879, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_18391, t_18881, t_19154, t_19636, t_19880, ...
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,# trace_info : t_18392, t_18882, t_19155, t_19637, t_19881, ...
            grad_output_buffer=self.grad_output_buffer                         # trace_info : t_18394, t_18884, t_19157, t_19639, t_19883, ...
            if self.config.defer_embedding_wgrad_compute                       # trace_info : t_18393, t_18883, t_19156, t_19638, t_19882, ...
            else None,                                                         # trace_info : t_18395, t_18885, t_19158, t_19640, t_19884, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_18396, t_18886, t_19159, t_19641, t_19885, ...
        )
        if self.gather_output:                                                 # trace_info : t_18445, t_18911, t_19206, t_19666, t_19931, ...
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel                                           # trace_info : t_18446, t_18912, t_19207, t_19667, t_19932, ...
        output_bias = self.bias if self.skip_bias_add else None                # trace_info : t_18447, t_18913, t_19208, t_19668, t_19933, ...
        return output, output_bias                                             # trace_info : t_18448, t_18914, t_19209, t_19669, t_19934, ...

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
        super(RowParallelLinear, self).__init__()                              # trace_info : t_9800, t_10544, t_10940, t_11684

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_9801, t_10545, t_10941, t_11685
        self.output_size = output_size                                         # trace_info : t_9802, t_10546, t_10942, t_11686
        self.input_is_parallel = input_is_parallel                             # trace_info : t_9803, t_10547, t_10943, t_11687
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_9804, t_10548, t_10944, t_11688
        self.config = config                                                   # trace_info : t_9805, t_10549, t_10945, t_11689
        self.is_expert = is_expert                                             # trace_info : t_9806, t_10550, t_10946, t_11690
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_9807, t_10551, t_10947, t_11691
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_9808, t_10552, t_10948, t_11692
        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_9809, t_10553, t_10949, t_11693
        if self.sequence_parallel and not self.input_is_parallel:              # trace_info : t_9810, t_10554, t_10950, t_11694
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_9811, t_10555, t_10557, t_10951, t_11695, ...
            config.tensor_model_parallel_size > 1 or self.expert_parallel      # trace_info : t_10556, t_11696
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_9812, t_10558, t_10952, t_11698
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_9813, t_10559, t_10953, t_11699
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_9819, t_10565, t_10959, t_11705

        self.input_size_per_partition = divide(input_size, world_size)         # trace_info : t_9825, t_10571, t_10965, t_11711

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:                                      # trace_info : t_9829, t_10575, t_10969, t_11715
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
            self.weight = Parameter(                                           # trace_info : t_9830, t_9837, t_10576, t_10583, t_10970, ...
                torch.empty(                                                   # trace_info : t_9831, t_9836, t_10577, t_10582, t_10971, ...
                    self.output_size,                                          # trace_info : t_9832, t_10578, t_10972, t_11718
                    self.input_size_per_partition,                             # trace_info : t_9833, t_10579, t_10973, t_11719
                    device=torch.cuda.current_device(),                        # trace_info : t_9834, t_10580, t_10974, t_11720
                    dtype=config.params_dtype,                                 # trace_info : t_9835, t_10581, t_10975, t_11721
                )
            )
            if config.perform_initialization:                                  # trace_info : t_9838, t_10584, t_10978, t_11724
                _initialize_affine_weight_gpu(                                 # trace_info : t_9839, t_9845, t_10585, t_10591, t_10979, ...
                    self.weight,                                               # trace_info : t_9840, t_10586, t_10980, t_11726
                    init_method,                                               # trace_info : t_9841, t_10587, t_10981, t_11727
                    partition_dim=1,                                           # trace_info : t_9842, t_10588, t_10982, t_11728
                    stride=stride,                                             # trace_info : t_9843, t_10589, t_10983, t_11729
                    expert_parallel=(self.is_expert and self.expert_parallel), # trace_info : t_9844, t_10590, t_10984, t_11730
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_9895, t_10642, t_11035, t_11782

        if bias:                                                               # trace_info : t_9896, t_10643, t_11036, t_11783
            if config.use_cpu_initialization:                                  # trace_info : t_9897, t_10644, t_11037, t_11784
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(                                         # trace_info : t_9898, t_9904, t_10645, t_10651, t_11038, ...
                    torch.empty(                                               # trace_info : t_9899, t_9903, t_10646, t_10650, t_11039, ...
                        self.output_size,                                      # trace_info : t_9900, t_10647, t_11040, t_11787
                        device=torch.cuda.current_device(),                    # trace_info : t_9901, t_10648, t_11041, t_11788
                        dtype=config.params_dtype,                             # trace_info : t_9902, t_10649, t_11042, t_11789
                    )
                )

            if config.perform_initialization:                                  # trace_info : t_9905, t_10652, t_11045, t_11792
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_9906, t_9908, t_10653, t_10655, t_11046, ...
                    self.bias.zero_()                                          # trace_info : t_9907, t_10654, t_11047, t_11794
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_9909, t_10656, t_11049, t_11796
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)    # trace_info : t_9910, t_10657, t_11050, t_11797
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_9911, t_10658, t_11051, t_11798

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_9912, t_9914, t_10659, t_10661, t_11052, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_9913, t_10660, t_11053, t_11800
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

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_18558, t_18924, t_19316, t_19679, t_22906, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:                                             # trace_info : t_18559, t_18925, t_19317, t_19680, t_22907, ...
            input_parallel = input_                                            # trace_info : t_18560, t_18926, t_19318, t_19681, t_22908, ...
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if not self.weight.requires_grad:                                      # trace_info : t_18561, t_18927, t_19319, t_19682, t_22909, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_18562, t_18928, t_19320, t_19683, t_22910, ...

        allreduce_dgrad = False                                                # trace_info : t_18563, t_18929, t_19321, t_19684, t_22911, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_18564, t_18573, t_18930, t_18939, t_19322, ...
            input=input_parallel,                                              # trace_info : t_18565, t_18931, t_19323, t_19686, t_22913, ...
            weight=self.weight,                                                # trace_info : t_18566, t_18932, t_19324, t_19687, t_22914, ...
            bias=None,                                                         # trace_info : t_18567, t_18933, t_19325, t_19688, t_22915, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_18568, t_18934, t_19326, t_19689, t_22916, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_18569, t_18935, t_19327, t_19690, t_22917, ...
            sequence_parallel=False,                                           # trace_info : t_18570, t_18936, t_19328, t_19691, t_22918, ...
            grad_output_buffer=None,                                           # trace_info : t_18571, t_18937, t_19329, t_19692, t_22919, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_18572, t_18938, t_19330, t_19693, t_22920, ...
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:                                          # trace_info : t_18597, t_18963, t_19355, t_19718, t_22945, ...
            assert self.skip_bias_add                                          # trace_info : t_18964, t_19719, t_23309, t_24064, t_27654, ...
            output_ = output_parallel                                          # trace_info : t_18965, t_19720, t_23310, t_24065, t_27655, ...
        elif self.sequence_parallel:                                           # trace_info : t_18598, t_19356, t_22946, t_23701, t_27291, ...
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)# trace_info : t_18599, t_19357, t_22947, t_23702, t_27292, ...
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:                                             # trace_info : t_18620, t_18966, t_19378, t_19721, t_22968, ...
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_                                                   # trace_info : t_18621, t_18967, t_19379, t_19722, t_22969, ...
            output_bias = self.bias                                            # trace_info : t_18622, t_18968, t_19380, t_19723, t_22970, ...
        return output, output_bias                                             # trace_info : t_18623, t_18969, t_19381, t_19724, t_22971, ...

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
