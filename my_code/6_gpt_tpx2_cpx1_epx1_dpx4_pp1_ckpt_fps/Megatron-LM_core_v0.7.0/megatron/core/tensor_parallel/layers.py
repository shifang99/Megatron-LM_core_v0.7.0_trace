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
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (# trace_info : t_20290, t_20299, t_20306, t_20315, t_20324, ...
        get_tensor_model_parallel_rank() == 0                                  # trace_info : t_20300, t_20388, t_20404, t_20420, t_20445, ...
    )


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_9515, t_9517, t_9519, t_9521, t_9926, ...
        assert not hasattr(tensor, attribute)                                  # trace_info : t_9516, t_9518, t_9520, t_9927, t_9929, ...
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)                      # trace_info : t_9522, t_9933, t_10075, t_10132, t_10404, ...
    setattr(tensor, 'partition_dim', dim)                                      # trace_info : t_9523, t_9934, t_10076, t_10133, t_10405, ...
    setattr(tensor, 'partition_stride', stride)                                # trace_info : t_9524, t_9935, t_10077, t_10134, t_10406, ...


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):                                           # trace_info : t_11730, t_11743, t_11759, t_11775, t_11791, ...
        if not hasattr(tensor, attribute):                                     # trace_info : t_11733, t_11736, t_11739, t_11746, t_11750, ...
            setattr(tensor, attribute, value)                                  # trace_info : t_11747, t_11751, t_11755, t_11763, t_11767, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_11731, t_11734, t_11737, t_11740, t_11744, ...
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])    # trace_info : t_11732, t_11735, t_11738, t_11745, t_11749, ...


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):                                                 # trace_info : t_15160, t_15184, t_15208, t_15232, t_15256, ...
        if hasattr(source_tensor, attribute):                                  # trace_info : t_15163, t_15167, t_15171, t_15187, t_15191, ...
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))# trace_info : t_15164, t_15168, t_15172, t_15188, t_15192, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_15161, t_15165, t_15169, t_15173, t_15185, ...
        maybe_copy(attribute)                                                  # trace_info : t_15162, t_15166, t_15170, t_15186, t_15190, ...


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(                                      # trace_info : t_9512, t_9514, t_9923, t_9925, t_10065, ...
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride      # trace_info : t_9513, t_9924, t_10066, t_10395, t_10553, ...
    )

    if not expert_parallel:                                                    # trace_info : t_9525, t_9936, t_10078, t_10407, t_10565, ...
        with get_cuda_rng_tracker().fork():                                    # trace_info : t_9526, t_9548, t_9937, t_9959, t_10079, ...
            init_method(weight)                                                # trace_info : t_9546, t_9957, t_10099, t_10428, t_10586, ...
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
        super(VocabParallelEmbedding, self).__init__()                         # trace_info : t_9470
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings                                   # trace_info : t_9471
        self.embedding_dim = embedding_dim                                     # trace_info : t_9472
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()# trace_info : t_9473
        # Divide the weight matrix along the vocaburaly dimension.
        (                                                                      # trace_info : t_9497
            self.vocab_start_index,                                            # trace_info : t_9498
            self.vocab_end_index,                                              # trace_info : t_9499
        ) = VocabUtility.vocab_range_from_global_vocab_size(                   # trace_info : t_9479, t_9486
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size# trace_info : t_9480
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index# trace_info : t_9500

        # Allocate weights and initialize.
        if config.use_cpu_initialization:                                      # trace_info : t_9501
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
            self.weight = Parameter(                                           # trace_info : t_9502, t_9509
                torch.empty(                                                   # trace_info : t_9503, t_9508
                    self.num_embeddings_per_partition,                         # trace_info : t_9504
                    self.embedding_dim,                                        # trace_info : t_9505
                    device=torch.cuda.current_device(),                        # trace_info : t_9506
                    dtype=config.params_dtype,                                 # trace_info : t_9507
                )
            )
            if config.perform_initialization:                                  # trace_info : t_9510
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)# trace_info : t_9511

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18356, t_21995, t_89602
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)# trace_info : t_18357, t_21996, t_89603
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index             # trace_info : t_18358, t_21997, t_89604
            masked_input[input_mask] = 0                                       # trace_info : t_18359, t_21998, t_89605
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.weight[masked_input]                            # trace_info : t_18360, t_21999, t_89606
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18361, t_22000, t_89607
            output_parallel[input_mask, :] = 0.0                               # trace_info : t_18362, t_22001, t_89608
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)     # trace_info : t_18363, t_22002, t_89609
        return output                                                          # trace_info : t_18377, t_22016, t_89623

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """ Non-default implementation for embeddings due to `allow_shape_mismatch` param """
        state_dict = self.state_dict(prefix='', keep_vars=True)                # trace_info : t_25090, t_92683

        weight_prefix = f'{prefix}weight'                                      # trace_info : t_25091, t_92684
        return {                                                               # trace_info : t_25172, t_92765
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(              # trace_info : t_25092, t_25097, t_92685, t_92690
                tensor=state_dict['weight'],                                   # trace_info : t_25093, t_92686
                key=weight_prefix,                                             # trace_info : t_25094, t_92687
                allow_shape_mismatch=True,                                     # trace_info : t_25095, t_92688
                prepend_offsets=sharded_offsets,                               # trace_info : t_25096, t_92689
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
        ctx.save_for_backward(input, weight)                                   # trace_info : t_18488, t_18685, t_18801, t_18853, t_18983, ...
        ctx.use_bias = bias is not None                                        # trace_info : t_18489, t_18686, t_18802, t_18854, t_18984, ...
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion        # trace_info : t_18490, t_18687, t_18803, t_18855, t_18985, ...
        ctx.allreduce_dgrad = allreduce_dgrad                                  # trace_info : t_18491, t_18688, t_18804, t_18856, t_18986, ...
        ctx.sequence_parallel = sequence_parallel                              # trace_info : t_18492, t_18689, t_18805, t_18857, t_18987, ...
        ctx.grad_output_buffer = grad_output_buffer                            # trace_info : t_18493, t_18690, t_18806, t_18858, t_18988, ...

        if sequence_parallel:                                                  # trace_info : t_18494, t_18691, t_18807, t_18859, t_18989, ...
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input                                                # trace_info : t_18495, t_18692, t_18808, t_18860, t_18990, ...

        output = torch.matmul(total_input, weight.t())                         # trace_info : t_18496, t_18693, t_18809, t_18861, t_18991, ...
        if bias is not None:                                                   # trace_info : t_18497, t_18694, t_18810, t_18862, t_18992, ...
            output = output + bias                                             # trace_info : t_18498, t_18993, t_22137, t_22630, t_89744, ...
        return output                                                          # trace_info : t_18499, t_18695, t_18811, t_18863, t_18994, ...

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
    if allreduce_dgrad is None:                                                # trace_info : t_18476, t_18673, t_18789, t_18841, t_18971, ...
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [                                                                   # trace_info : t_18484, t_18681, t_18797, t_18849, t_18979, ...
        input,                                                                 # trace_info : t_18477, t_18674, t_18790, t_18842, t_18972, ...
        weight,                                                                # trace_info : t_18478, t_18675, t_18791, t_18843, t_18973, ...
        bias,                                                                  # trace_info : t_18479, t_18676, t_18792, t_18844, t_18974, ...
        gradient_accumulation_fusion,                                          # trace_info : t_18480, t_18677, t_18793, t_18845, t_18975, ...
        allreduce_dgrad,                                                       # trace_info : t_18481, t_18678, t_18794, t_18846, t_18976, ...
        sequence_parallel,                                                     # trace_info : t_18482, t_18679, t_18795, t_18847, t_18977, ...
        grad_output_buffer,                                                    # trace_info : t_18483, t_18680, t_18796, t_18848, t_18978, ...
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:           # trace_info : t_18485, t_18682, t_18798, t_18850, t_18980, ...
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":               # trace_info : t_18486, t_18683, t_18799, t_18851, t_18981, ...
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

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)        # trace_info : t_18487, t_18684, t_18800, t_18852, t_18982, ...


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
        super(ColumnParallelLinear, self).__init__()                           # trace_info : t_10018, t_10347, t_11020, t_11349, t_11669

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_10019, t_10348, t_11021, t_11350, t_11670
        self.output_size = output_size                                         # trace_info : t_10020, t_10349, t_11022, t_11351, t_11671
        self.gather_output = gather_output                                     # trace_info : t_10021, t_10350, t_11023, t_11352, t_11672
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_10022, t_10351, t_11024, t_11353, t_11673
        self.is_expert = is_expert                                             # trace_info : t_10023, t_10352, t_11025, t_11354, t_11674
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_10024, t_10353, t_11026, t_11355, t_11675
        self.embedding_activation_buffer = embedding_activation_buffer         # trace_info : t_10025, t_10354, t_11027, t_11356, t_11676
        self.grad_output_buffer = grad_output_buffer                           # trace_info : t_10026, t_10355, t_11028, t_11357, t_11677
        self.config = config                                                   # trace_info : t_10027, t_10356, t_11029, t_11358, t_11678
        self.disable_grad_reduce = disable_grad_reduce                         # trace_info : t_10028, t_10357, t_11030, t_11359, t_11679

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_10029, t_10358, t_11031, t_11360, t_11680
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_10030, t_10359, t_11032, t_11361, t_11681
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_10031, t_10360, t_11033, t_11362, t_11682
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_10037, t_10366, t_11039, t_11368, t_11688

        self.output_size_per_partition = divide(output_size, world_size)       # trace_info : t_10043, t_10372, t_11045, t_11374, t_11694

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:                                   # trace_info : t_10047, t_10376, t_11049, t_11378, t_11698
            if config.use_cpu_initialization:                                  # trace_info : t_10048, t_10377, t_11050, t_11379
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
                self.weight = Parameter(                                       # trace_info : t_10049, t_10056, t_10378, t_10385, t_11051, ...
                    torch.empty(                                               # trace_info : t_10050, t_10055, t_10379, t_10384, t_11052, ...
                        self.output_size_per_partition,                        # trace_info : t_10051, t_10380, t_11053, t_11382
                        self.input_size,                                       # trace_info : t_10052, t_10381, t_11054, t_11383
                        device=torch.cuda.current_device(),                    # trace_info : t_10053, t_10382, t_11055, t_11384
                        dtype=config.params_dtype,                             # trace_info : t_10054, t_10383, t_11056, t_11385
                    )
                )
                if config.perform_initialization:                              # trace_info : t_10057, t_10386, t_11059, t_11388
                    _initialize_affine_weight_gpu(                             # trace_info : t_10058, t_10064, t_10387, t_10393, t_11060, ...
                        self.weight,                                           # trace_info : t_10059, t_10388, t_11061, t_11390
                        init_method,                                           # trace_info : t_10060, t_10389, t_11062, t_11391
                        partition_dim=0,                                       # trace_info : t_10061, t_10390, t_11063, t_11392
                        stride=stride,                                         # trace_info : t_10062, t_10391, t_11064, t_11393
                        expert_parallel=(self.is_expert and self.expert_parallel),# trace_info : t_10063, t_10392, t_11065, t_11394
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10114, t_10443, t_11116, t_11445
        else:
            self.weight = None                                                 # trace_info : t_11699

        if bias:                                                               # trace_info : t_10115, t_10444, t_11117, t_11446, t_11700
            if config.use_cpu_initialization:                                  # trace_info : t_10116, t_10445, t_11118, t_11447
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(                                         # trace_info : t_10117, t_10123, t_10446, t_10452, t_11119, ...
                    torch.empty(                                               # trace_info : t_10118, t_10122, t_10447, t_10451, t_11120, ...
                        self.output_size_per_partition,                        # trace_info : t_10119, t_10448, t_11121, t_11450
                        device=torch.cuda.current_device(),                    # trace_info : t_10120, t_10449, t_11122, t_11451
                        dtype=config.params_dtype,                             # trace_info : t_10121, t_10450, t_11123, t_11452
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)   # trace_info : t_10124, t_10453, t_11126, t_11455
            if config.perform_initialization:                                  # trace_info : t_10135, t_10464, t_11137, t_11466
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_10136, t_10138, t_10465, t_10467, t_11138, ...
                    self.bias.zero_()                                          # trace_info : t_10137, t_10466, t_11139, t_11468
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_10139, t_10468, t_11141, t_11470
        else:
            self.register_parameter('bias', None)                              # trace_info : t_11701

        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_10140, t_10469, t_11142, t_11471, t_11702
        if self.sequence_parallel and world_size <= 1:                         # trace_info : t_10141, t_10470, t_11143, t_11472, t_11703
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel   # trace_info : t_10142, t_10471, t_11144, t_11473, t_11704

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:# trace_info : t_10143, t_10472, t_11145, t_11474, t_11705
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_10144, t_10473, t_11146, t_11475, t_11706

        if self.allreduce_dgrad and self.sequence_parallel:                    # trace_info : t_10145, t_10474, t_11147, t_11476, t_11707
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_10146, t_10475, t_11148, t_11477, t_11708

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_10147, t_10149, t_10476, t_10478, t_11149, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_10148, t_10477, t_11150, t_11479, t_11710
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
        if weight is None:                                                     # trace_info : t_18452, t_18765, t_18947, t_19258, t_19411, ...
            if self.weight is None:                                            # trace_info : t_18453, t_18766, t_18948, t_19259, t_22092, ...
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight                                               # trace_info : t_18454, t_18767, t_18949, t_19260, t_22093, ...
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size) # trace_info : t_19412, t_23049, t_90656
            if weight.shape != expected_shape:                                 # trace_info : t_19413, t_23050, t_90657
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_18455, t_18768, t_18950, t_19261, t_19414, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None                   # trace_info : t_18456, t_18769, t_18951, t_19262, t_19415, ...

        if (                                                                   # trace_info : t_18458, t_18771, t_18953, t_19264, t_19417, ...
            self.allreduce_dgrad                                               # trace_info : t_18457, t_18770, t_18952, t_19263, t_19416, ...
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_                                            # trace_info : t_18459, t_18772, t_18954, t_19265, t_19418, ...
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:                          # trace_info : t_18460, t_18773, t_18955, t_19266, t_19419, ...
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:                                           # trace_info : t_18461, t_18774, t_18956, t_19267, t_19420, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_18462, t_18775, t_18957, t_19268, t_19421, ...

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad# trace_info : t_18463, t_18776, t_18958, t_19269, t_19422, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_18464, t_18475, t_18777, t_18788, t_18959, ...
            input=input_parallel,                                              # trace_info : t_18465, t_18778, t_18960, t_19271, t_19424, ...
            weight=weight,                                                     # trace_info : t_18466, t_18779, t_18961, t_19272, t_19425, ...
            bias=bias,                                                         # trace_info : t_18467, t_18780, t_18962, t_19273, t_19426, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_18468, t_18781, t_18963, t_19274, t_19427, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_18469, t_18782, t_18964, t_19275, t_19428, ...
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,# trace_info : t_18470, t_18783, t_18965, t_19276, t_19429, ...
            grad_output_buffer=self.grad_output_buffer                         # trace_info : t_18472, t_18785, t_18967, t_19278, t_19431, ...
            if self.config.defer_embedding_wgrad_compute                       # trace_info : t_18471, t_18784, t_18966, t_19277, t_19430, ...
            else None,                                                         # trace_info : t_18473, t_18786, t_18968, t_19279, t_19432, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_18474, t_18787, t_18969, t_19280, t_19433, ...
        )
        if self.gather_output:                                                 # trace_info : t_18500, t_18812, t_18995, t_19305, t_19458, ...
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel                                           # trace_info : t_18501, t_18813, t_18996, t_19306, t_19459, ...
        output_bias = self.bias if self.skip_bias_add else None                # trace_info : t_18502, t_18814, t_18997, t_19307, t_19460, ...
        return output, output_bias                                             # trace_info : t_18503, t_18815, t_18998, t_19308, t_19461, ...

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)                # trace_info : t_25786, t_26267, t_27580, t_28061, t_29036, ...
        return make_sharded_tensors_for_checkpoint(                            # trace_info : t_25788, t_25790, t_26269, t_26271, t_27582, ...
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets      # trace_info : t_25789, t_26270, t_27583, t_28064, t_29039, ...
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None                                                            # trace_info : t_25787, t_26268, t_27581, t_28062, t_29037, ...


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
        super(RowParallelLinear, self).__init__()                              # trace_info : t_9877, t_10506, t_10879, t_11508

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_9878, t_10507, t_10880, t_11509
        self.output_size = output_size                                         # trace_info : t_9879, t_10508, t_10881, t_11510
        self.input_is_parallel = input_is_parallel                             # trace_info : t_9880, t_10509, t_10882, t_11511
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_9881, t_10510, t_10883, t_11512
        self.config = config                                                   # trace_info : t_9882, t_10511, t_10884, t_11513
        self.is_expert = is_expert                                             # trace_info : t_9883, t_10512, t_10885, t_11514
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_9884, t_10513, t_10886, t_11515
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_9885, t_10514, t_10887, t_11516
        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_9886, t_10515, t_10888, t_11517
        if self.sequence_parallel and not self.input_is_parallel:              # trace_info : t_9887, t_10516, t_10889, t_11518
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_9888, t_10517, t_10890, t_11519
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_9889, t_10518, t_10891, t_11520
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_9890, t_10519, t_10892, t_11521
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_9896, t_10525, t_10898, t_11527

        self.input_size_per_partition = divide(input_size, world_size)         # trace_info : t_9902, t_10531, t_10904, t_11533

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:                                      # trace_info : t_9906, t_10535, t_10908, t_11537
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
            self.weight = Parameter(                                           # trace_info : t_9907, t_9914, t_10536, t_10543, t_10909, ...
                torch.empty(                                                   # trace_info : t_9908, t_9913, t_10537, t_10542, t_10910, ...
                    self.output_size,                                          # trace_info : t_9909, t_10538, t_10911, t_11540
                    self.input_size_per_partition,                             # trace_info : t_9910, t_10539, t_10912, t_11541
                    device=torch.cuda.current_device(),                        # trace_info : t_9911, t_10540, t_10913, t_11542
                    dtype=config.params_dtype,                                 # trace_info : t_9912, t_10541, t_10914, t_11543
                )
            )
            if config.perform_initialization:                                  # trace_info : t_9915, t_10544, t_10917, t_11546
                _initialize_affine_weight_gpu(                                 # trace_info : t_9916, t_9922, t_10545, t_10551, t_10918, ...
                    self.weight,                                               # trace_info : t_9917, t_10546, t_10919, t_11548
                    init_method,                                               # trace_info : t_9918, t_10547, t_10920, t_11549
                    partition_dim=1,                                           # trace_info : t_9919, t_10548, t_10921, t_11550
                    stride=stride,                                             # trace_info : t_9920, t_10549, t_10922, t_11551
                    expert_parallel=(self.is_expert and self.expert_parallel), # trace_info : t_9921, t_10550, t_10923, t_11552
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_9972, t_10601, t_10974, t_11603

        if bias:                                                               # trace_info : t_9973, t_10602, t_10975, t_11604
            if config.use_cpu_initialization:                                  # trace_info : t_9974, t_10603, t_10976, t_11605
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(                                         # trace_info : t_9975, t_9981, t_10604, t_10610, t_10977, ...
                    torch.empty(                                               # trace_info : t_9976, t_9980, t_10605, t_10609, t_10978, ...
                        self.output_size,                                      # trace_info : t_9977, t_10606, t_10979, t_11608
                        device=torch.cuda.current_device(),                    # trace_info : t_9978, t_10607, t_10980, t_11609
                        dtype=config.params_dtype,                             # trace_info : t_9979, t_10608, t_10981, t_11610
                    )
                )

            if config.perform_initialization:                                  # trace_info : t_9982, t_10611, t_10984, t_11613
                # Always initialize bias to zero.
                with torch.no_grad():                                          # trace_info : t_9983, t_9985, t_10612, t_10614, t_10985, ...
                    self.bias.zero_()                                          # trace_info : t_9984, t_10613, t_10986, t_11615
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))# trace_info : t_9986, t_10615, t_10988, t_11617
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)    # trace_info : t_9987, t_10616, t_10989, t_11618
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_9988, t_10617, t_10990, t_11619

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_9989, t_9991, t_10618, t_10620, t_10991, ...
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_9990, t_10619, t_10992, t_11621
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

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_18657, t_18825, t_19150, t_19318, t_22294, ...
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:                                             # trace_info : t_18658, t_18826, t_19151, t_19319, t_22295, ...
            input_parallel = input_                                            # trace_info : t_18659, t_18827, t_19152, t_19320, t_22296, ...
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if not self.weight.requires_grad:                                      # trace_info : t_18660, t_18828, t_19153, t_19321, t_22297, ...
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_18661, t_18829, t_19154, t_19322, t_22298, ...

        allreduce_dgrad = False                                                # trace_info : t_18662, t_18830, t_19155, t_19323, t_22299, ...

        output_parallel = self._forward_impl(                                  # trace_info : t_18663, t_18672, t_18831, t_18840, t_19156, ...
            input=input_parallel,                                              # trace_info : t_18664, t_18832, t_19157, t_19325, t_22301, ...
            weight=self.weight,                                                # trace_info : t_18665, t_18833, t_19158, t_19326, t_22302, ...
            bias=None,                                                         # trace_info : t_18666, t_18834, t_19159, t_19327, t_22303, ...
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_18667, t_18835, t_19160, t_19328, t_22304, ...
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_18668, t_18836, t_19161, t_19329, t_22305, ...
            sequence_parallel=False,                                           # trace_info : t_18669, t_18837, t_19162, t_19330, t_22306, ...
            grad_output_buffer=None,                                           # trace_info : t_18670, t_18838, t_19163, t_19331, t_22307, ...
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_18671, t_18839, t_19164, t_19332, t_22308, ...
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:                                          # trace_info : t_18696, t_18864, t_19189, t_19357, t_22333, ...
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:                                           # trace_info : t_18697, t_18865, t_19190, t_19358, t_22334, ...
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)# trace_info : t_18698, t_18866, t_19191, t_19359, t_22335, ...
        if not self.skip_bias_add:                                             # trace_info : t_18712, t_18880, t_19205, t_19373, t_22349, ...
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_                                                   # trace_info : t_18713, t_18881, t_19206, t_19374, t_22350, ...
            output_bias = self.bias                                            # trace_info : t_18714, t_18882, t_19207, t_19375, t_22351, ...
        return output, output_bias                                             # trace_info : t_18715, t_18883, t_19208, t_19376, t_22352, ...

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)                # trace_info : t_25564, t_26501, t_27358, t_28295, t_93157, ...
        return make_sharded_tensors_for_checkpoint(                            # trace_info : t_25566, t_25568, t_26503, t_26505, t_27360, ...
            state_dict, prefix, {'weight': 1}, sharded_offsets                 # trace_info : t_25567, t_26504, t_27361, t_28298, t_93160, ...
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None                                                            # trace_info : t_25565, t_26502, t_27359, t_28296, t_93158, ...
