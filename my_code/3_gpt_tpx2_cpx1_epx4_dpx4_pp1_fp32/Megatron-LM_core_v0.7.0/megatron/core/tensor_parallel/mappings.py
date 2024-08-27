# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_global_memory_buffer,
    get_tensor_and_expert_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:                            # trace_info : t_18225, t_22578, t_26923
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())# trace_info : t_18231, t_22584, t_26929

    return input_                                                              # trace_info : t_18235, t_22588, t_26933


def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()                        # trace_info : t_18248, t_22601, t_26946
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:                                                        # trace_info : t_18254, t_22607, t_26952
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]                                                # trace_info : t_18255, t_22608, t_26953
    assert (
        dim_size % world_size == 0                                             # trace_info : t_18256, t_22609, t_26954
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size                                    # trace_info : t_18257, t_22610, t_26955
    rank = get_tensor_model_parallel_rank()                                    # trace_info : t_18258, t_22611, t_26956
    dim_offset = rank * local_dim_size                                         # trace_info : t_18264, t_22617, t_26962

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()     # trace_info : t_18265, t_22618, t_26963

    return output                                                              # trace_info : t_18266, t_22619, t_26964


def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=-1).contiguous()

    return output


def _reduce_scatter_along_last_dim(input_):
    """Reduce-scatter tensors on the last dimension."""
    world_size = get_tensor_model_parallel_world_size()
    target_shape = list(input_.size())
    target_shape[-1] = target_shape[-1] // world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor).reshape(target_shape)
    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )

    return output


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()                        # trace_info : t_18602, t_19360, t_22950, t_23705, t_27295, ...
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:                                                        # trace_info : t_18608, t_19366, t_22956, t_23711, t_27301, ...
        return input_

    dim_size = list(input_.size())                                             # trace_info : t_18609, t_19367, t_22957, t_23712, t_27302, ...
    assert (
        dim_size[0] % world_size == 0                                          # trace_info : t_18610, t_19368, t_22958, t_23713, t_27303, ...
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size                                    # trace_info : t_18611, t_19369, t_22959, t_23714, t_27304, ...

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())# trace_info : t_18612, t_19370, t_22960, t_23715, t_27305, ...
    torch.distributed._reduce_scatter_base(                                    # trace_info : t_18613, t_18618, t_19371, t_19376, t_22961, ...
        output, input_.contiguous(), group=get_tensor_model_parallel_group()   # trace_info : t_18614, t_19372, t_22962, t_23717, t_27307, ...
    )
    return output                                                              # trace_info : t_18619, t_19377, t_22967, t_23722, t_27312, ...


def _gather_along_first_dim_moe(input_, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension."""
    group = get_tensor_and_expert_parallel_group()                             # trace_info : t_18770, t_18791, t_18809, t_19525, t_19546, ...
    world_size = torch.distributed.get_world_size(group=group)                 # trace_info : t_18773, t_18794, t_18812, t_19528, t_19549, ...
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:                                                        # trace_info : t_18774, t_18795, t_18813, t_19529, t_19550, ...
        return input_

    dim_size = list(input_.size())                                             # trace_info : t_18775, t_18796, t_18814, t_19530, t_19551, ...
    dim_size[0] = dim_size[0] * world_size                                     # trace_info : t_18776, t_18797, t_18815, t_19531, t_19552, ...

    if use_global_buffer:                                                      # trace_info : t_18777, t_18798, t_18816, t_19532, t_19553, ...
        output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")# trace_info : t_18817, t_19572, t_23162, t_23917, t_27507, ...
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())# trace_info : t_18778, t_18799, t_19533, t_19554, t_23123, ...
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)# trace_info : t_18779, t_18800, t_18824, t_19534, t_19555, ...

    return output                                                              # trace_info : t_18780, t_18801, t_18825, t_19535, t_19556, ...


def _reduce_scatter_along_first_dim_moe(input_, use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group."""
    group = get_tensor_and_expert_parallel_group()                             # trace_info : t_19022, t_19045, t_19777, t_19800, t_23367, ...
    world_size = torch.distributed.get_world_size(group=group)                 # trace_info : t_19025, t_19048, t_19780, t_19803, t_23370, ...
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:                                                        # trace_info : t_19026, t_19049, t_19781, t_19804, t_23371, ...
        return input_

    dim_size = list(input_.size())                                             # trace_info : t_19027, t_19050, t_19782, t_19805, t_23372, ...
    assert dim_size[0] % world_size == 0                                       # trace_info : t_19028, t_19051, t_19783, t_19806, t_23373, ...
    dim_size[0] = dim_size[0] // world_size                                    # trace_info : t_19029, t_19052, t_19784, t_19807, t_23374, ...

    if use_global_buffer:                                                      # trace_info : t_19030, t_19053, t_19785, t_19808, t_23375, ...
        output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())# trace_info : t_19031, t_19054, t_19786, t_19809, t_23376, ...
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)# trace_info : t_19032, t_19055, t_19787, t_19810, t_23377, ...
    return output                                                              # trace_info : t_19033, t_19056, t_19788, t_19811, t_23378, ...


def _gather_along_first_dim_expert_parallel(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = get_expert_model_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)                                                 # trace_info : t_18224, t_22577, t_26922

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)                                  # trace_info : t_18247, t_22600, t_26945

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)                         # trace_info : t_18601, t_19359, t_22949, t_23704, t_27294, ...

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""  # TODO

    @staticmethod
    def symbolic(graph, input_, use_global_buffer=False):
        return _gather_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, use_global_buffer=False):
        ctx.use_global_buffer = use_global_buffer                              # trace_info : t_18768, t_18789, t_18807, t_19523, t_19544, ...
        return _gather_along_first_dim_moe(input_, use_global_buffer)          # trace_info : t_18769, t_18790, t_18808, t_19524, t_19545, ...

    @staticmethod
    def backward(ctx, grad_output):
        use_global_buffer = ctx.use_global_buffer
        return _reduce_scatter_along_first_dim_moe(grad_output, use_global_buffer), None


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, use_global_buffer=False):
        return _reduce_scatter_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, use_global_buffer=False):
        ctx.use_global_buffer = use_global_buffer                              # trace_info : t_19020, t_19043, t_19775, t_19798, t_23365, ...
        return _reduce_scatter_along_first_dim_moe(input_, use_global_buffer)  # trace_info : t_19021, t_19044, t_19776, t_19799, t_23366, ...

    @staticmethod
    def backward(ctx, grad_output):
        use_global_buffer = ctx.use_global_buffer
        return _gather_along_first_dim_moe(grad_output, use_global_buffer), None


class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_last_dim(grad_output)


class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_last_dim(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)                        # trace_info : t_18223, t_22576, t_26921


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)                      # trace_info : t_18246, t_22599, t_26944


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)                # trace_info : t_18600, t_19358, t_22948, t_23703, t_27293, ...


def gather_from_sequence_parallel_region_to_moe(input_, use_global_buffer=False):
    return _GatherFromSequenceParallelRegionToMOE.apply(input_, use_global_buffer)# trace_info : t_18767, t_18788, t_18806, t_19522, t_19543, ...


def reduce_scatter_to_sequence_parallel_region_from_moe(input_, use_global_buffer=False):
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_, use_global_buffer)# trace_info : t_19019, t_19042, t_19774, t_19797, t_23364, ...


def all_gather_last_dim_from_tensor_parallel_region(input_):
    return _AllGatherFromTensorParallelRegion.apply(input_)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_):
    return _ReduceScatterToTensorParallelRegion.apply(input_)


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes_=None):
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes_)


def all_to_all_sp2hp(input_):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape [num_tokens/TP, H] to [num_tokens, H/TP].

    Args:
        input_ (torch.Tensor): The input tensor which has been distributed along the sequence dimension.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens, H/TP].

    """
    world_size = get_tensor_model_parallel_world_size()
    tp_group = get_tensor_model_parallel_group()
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(tp_group, concat_tensor)
    return output


def all_to_all_hp2sp(input_):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape [num_tokens, H/TP] to [num_tokens/TP, H].

    Args:
        input_ (torch.Tensor): The input tensor which has been distributed along the hidden dimension.
        
    Returns:
        torch.Tensor: The output tensor with shape [num_tokens/TP, H].
    """
    world_size = get_tensor_model_parallel_world_size()
    input_ = input_.reshape(-1, input_.shape[-1])
    tp_group = get_tensor_model_parallel_group()
    input_exchanged = all_to_all(tp_group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped, split_size_or_sections=input_reshaped.shape[0] // world_size, dim=0
    )
    output = torch.cat(split_tensors, dim=-1)
    return output
