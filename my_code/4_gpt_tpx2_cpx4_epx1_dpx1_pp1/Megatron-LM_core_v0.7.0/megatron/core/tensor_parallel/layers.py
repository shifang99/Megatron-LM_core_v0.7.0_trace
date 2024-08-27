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
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (# trace_info : t_19560, t_19569, t_19576, t_19585, t_19594, ...
        get_tensor_model_parallel_rank() == 0                                  # trace_info : t_19570, t_19658, t_19674, t_19690, t_19715, ...
    )


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_9792, t_9794, t_9796, t_9798
        assert not hasattr(tensor, attribute)                                  # trace_info : t_9793, t_9795, t_9797
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)                      # trace_info : t_9799
    setattr(tensor, 'partition_dim', dim)                                      # trace_info : t_9800
    setattr(tensor, 'partition_stride', stride)                                # trace_info : t_9801


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):                                           # trace_info : t_11560, t_11573, t_11589, t_11602, t_11618, ...
        if not hasattr(tensor, attribute):                                     # trace_info : t_11563, t_11566, t_11569, t_11576, t_11580, ...
            setattr(tensor, attribute, value)                                  # trace_info : t_11577, t_11581, t_11585, t_11606, t_11610, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_11561, t_11564, t_11567, t_11570, t_11574, ...
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])    # trace_info : t_11562, t_11565, t_11568, t_11575, t_11579, ...


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):                                                 # trace_info : t_14990, t_15014, t_15038, t_15062, t_15086, ...
        if hasattr(source_tensor, attribute):                                  # trace_info : t_14993, t_14997, t_15001, t_15017, t_15021, ...
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))# trace_info : t_14994, t_14998, t_15002, t_15018, t_15022, ...

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:                       # trace_info : t_14991, t_14995, t_14999, t_15003, t_15015, ...
        maybe_copy(attribute)                                                  # trace_info : t_14992, t_14996, t_15000, t_15016, t_15020, ...


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(                                      # trace_info : t_9789, t_9791
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride      # trace_info : t_9790
    )

    if not expert_parallel:                                                    # trace_info : t_9802
        with get_cuda_rng_tracker().fork():                                    # trace_info : t_9803, t_9825
            init_method(weight)                                                # trace_info : t_9823
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
        super(VocabParallelEmbedding, self).__init__()                         # trace_info : t_9747
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings                                   # trace_info : t_9748
        self.embedding_dim = embedding_dim                                     # trace_info : t_9749
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()# trace_info : t_9750
        # Divide the weight matrix along the vocaburaly dimension.
        (                                                                      # trace_info : t_9774
            self.vocab_start_index,                                            # trace_info : t_9775
            self.vocab_end_index,                                              # trace_info : t_9776
        ) = VocabUtility.vocab_range_from_global_vocab_size(                   # trace_info : t_9756, t_9763
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size# trace_info : t_9757
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index# trace_info : t_9777

        # Allocate weights and initialize.
        if config.use_cpu_initialization:                                      # trace_info : t_9778
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
            self.weight = Parameter(                                           # trace_info : t_9779, t_9786
                torch.empty(                                                   # trace_info : t_9780, t_9785
                    self.num_embeddings_per_partition,                         # trace_info : t_9781
                    self.embedding_dim,                                        # trace_info : t_9782
                    device=torch.cuda.current_device(),                        # trace_info : t_9783
                    dtype=config.params_dtype,                                 # trace_info : t_9784
                )
            )
            if config.perform_initialization:                                  # trace_info : t_9787
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)# trace_info : t_9788

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18196, t_21382, t_24568
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)# trace_info : t_18197, t_21383, t_24569
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index             # trace_info : t_18198, t_21384, t_24570
            masked_input[input_mask] = 0                                       # trace_info : t_18199, t_21385, t_24571
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.weight[masked_input]                            # trace_info : t_18200, t_21386, t_24572
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:                                # trace_info : t_18201, t_21387, t_24573
            output_parallel[input_mask, :] = 0.0                               # trace_info : t_18202, t_21388, t_24574
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)     # trace_info : t_18203, t_21389, t_24575
        return output                                                          # trace_info : t_18217, t_21403, t_24589

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
        ctx.save_for_backward(input, weight)                                   # trace_info : t_18713, t_21899, t_25085
        ctx.use_bias = bias is not None                                        # trace_info : t_18714, t_21900, t_25086
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion        # trace_info : t_18715, t_21901, t_25087
        ctx.allreduce_dgrad = allreduce_dgrad                                  # trace_info : t_18716, t_21902, t_25088
        ctx.sequence_parallel = sequence_parallel                              # trace_info : t_18717, t_21903, t_25089
        ctx.grad_output_buffer = grad_output_buffer                            # trace_info : t_18718, t_21904, t_25090

        if sequence_parallel:                                                  # trace_info : t_18719, t_21905, t_25091
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input                                                # trace_info : t_18720, t_21906, t_25092

        output = torch.matmul(total_input, weight.t())                         # trace_info : t_18721, t_21907, t_25093
        if bias is not None:                                                   # trace_info : t_18722, t_21908, t_25094
            output = output + bias
        return output                                                          # trace_info : t_18723, t_21909, t_25095

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
    if allreduce_dgrad is None:                                                # trace_info : t_18701, t_21887, t_25073
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [                                                                   # trace_info : t_18709, t_21895, t_25081
        input,                                                                 # trace_info : t_18702, t_21888, t_25074
        weight,                                                                # trace_info : t_18703, t_21889, t_25075
        bias,                                                                  # trace_info : t_18704, t_21890, t_25076
        gradient_accumulation_fusion,                                          # trace_info : t_18705, t_21891, t_25077
        allreduce_dgrad,                                                       # trace_info : t_18706, t_21892, t_25078
        sequence_parallel,                                                     # trace_info : t_18707, t_21893, t_25079
        grad_output_buffer,                                                    # trace_info : t_18708, t_21894, t_25080
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:           # trace_info : t_18710, t_21896, t_25082
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":               # trace_info : t_18711, t_21897, t_25083
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

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)        # trace_info : t_18712, t_21898, t_25084


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
        super(ColumnParallelLinear, self).__init__()                           # trace_info : t_11499

        # Keep input parameters
        self.input_size = input_size                                           # trace_info : t_11500
        self.output_size = output_size                                         # trace_info : t_11501
        self.gather_output = gather_output                                     # trace_info : t_11502
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add                                     # trace_info : t_11503
        self.is_expert = is_expert                                             # trace_info : t_11504
        self.expert_parallel = config.expert_model_parallel_size > 1           # trace_info : t_11505
        self.embedding_activation_buffer = embedding_activation_buffer         # trace_info : t_11506
        self.grad_output_buffer = grad_output_buffer                           # trace_info : t_11507
        self.config = config                                                   # trace_info : t_11508
        self.disable_grad_reduce = disable_grad_reduce                         # trace_info : t_11509

        self.explicit_expert_comm = self.is_expert and (                       # trace_info : t_11510
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if self.explicit_expert_comm and config.moe_extended_tp:               # trace_info : t_11511
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()                # trace_info : t_11512
            rank = get_tensor_model_parallel_rank()                            # trace_info : t_11518

        self.output_size_per_partition = divide(output_size, world_size)       # trace_info : t_11524

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:                                   # trace_info : t_11528
            if config.use_cpu_initialization:
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
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        expert_parallel=(self.is_expert and self.expert_parallel),
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.weight = None                                                 # trace_info : t_11529

        if bias:                                                               # trace_info : t_11530
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.register_parameter('bias', None)                              # trace_info : t_11531

        self.sequence_parallel = config.sequence_parallel                      # trace_info : t_11532
        if self.sequence_parallel and world_size <= 1:                         # trace_info : t_11533
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel   # trace_info : t_11534

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:# trace_info : t_11535
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion# trace_info : t_11536

        if self.allreduce_dgrad and self.sequence_parallel:                    # trace_info : t_11537
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce # trace_info : t_11538

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(                               # trace_info : t_11539, t_11541
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault( # trace_info : t_11540
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
        if weight is None:                                                     # trace_info : t_18677, t_21863, t_25049
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size) # trace_info : t_18678, t_21864, t_25050
            if weight.shape != expected_shape:                                 # trace_info : t_18679, t_21865, t_25051
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:                    # trace_info : t_18680, t_21866, t_25052
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None                   # trace_info : t_18681, t_21867, t_25053

        if (                                                                   # trace_info : t_18683, t_21869, t_25055
            self.allreduce_dgrad                                               # trace_info : t_18682, t_21868, t_25054
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_                                            # trace_info : t_18684, t_21870, t_25056
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:                          # trace_info : t_18685, t_21871, t_25057
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:                                           # trace_info : t_18686, t_21872, t_25058
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce# trace_info : t_18687, t_21873, t_25059

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad# trace_info : t_18688, t_21874, t_25060

        output_parallel = self._forward_impl(                                  # trace_info : t_18689, t_18700, t_21875, t_21886, t_25061, ...
            input=input_parallel,                                              # trace_info : t_18690, t_21876, t_25062
            weight=weight,                                                     # trace_info : t_18691, t_21877, t_25063
            bias=bias,                                                         # trace_info : t_18692, t_21878, t_25064
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,    # trace_info : t_18693, t_21879, t_25065
            async_grad_allreduce=allreduce_dgrad,                              # trace_info : t_18694, t_21880, t_25066
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,# trace_info : t_18695, t_21881, t_25067
            grad_output_buffer=self.grad_output_buffer                         # trace_info : t_18697, t_21883, t_25069
            if self.config.defer_embedding_wgrad_compute                       # trace_info : t_18696, t_21882, t_25068
            else None,                                                         # trace_info : t_18698, t_21884, t_25070
            allreduce_dgrad=allreduce_dgrad,                                   # trace_info : t_18699, t_21885, t_25071
        )
        if self.gather_output:                                                 # trace_info : t_18724, t_21910, t_25096
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel                                           # trace_info : t_18725, t_21911, t_25097
        output_bias = self.bias if self.skip_bias_add else None                # trace_info : t_18726, t_21912, t_25098
        return output, output_bias                                             # trace_info : t_18727, t_21913, t_25099

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
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        self.explicit_expert_comm = self.is_expert and (
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()

        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:
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
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel),
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
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

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if not self.weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        allreduce_dgrad = False

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=allreduce_dgrad,
            sequence_parallel=False,
            grad_output_buffer=None,
            allreduce_dgrad=allreduce_dgrad,
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

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
