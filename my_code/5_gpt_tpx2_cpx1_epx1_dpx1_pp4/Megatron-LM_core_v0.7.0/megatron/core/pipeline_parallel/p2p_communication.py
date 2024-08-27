# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch

from megatron import core
from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
)

# Types
Shape = Union[List[int], torch.Size]


def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Args:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_model_parallel_group(),
        )
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    ops = []                                                                   # trace_info : t_19533, t_19664, t_23261, t_23392, t_26989, ...
    if tensor_send_prev is not None:                                           # trace_info : t_19534, t_19665, t_23262, t_23393, t_26990, ...
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:                                           # trace_info : t_19535, t_19666, t_23263, t_23394, t_26991, ...
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:                                           # trace_info : t_19536, t_19667, t_23264, t_23395, t_26992, ...
        send_next_op = torch.distributed.P2POp(                                # trace_info : t_19537, t_19554, t_23265, t_23282, t_26993, ...
            torch.distributed.isend,                                           # trace_info : t_19538, t_23266, t_26994
            tensor_send_next,                                                  # trace_info : t_19539, t_23267, t_26995
            get_pipeline_model_parallel_next_rank(),                           # trace_info : t_19540, t_23268, t_26996
            group,                                                             # trace_info : t_19553, t_23281, t_27009
        )
        ops.append(send_next_op)                                               # trace_info : t_19555, t_23283, t_27011
    if tensor_recv_next is not None:                                           # trace_info : t_19556, t_19668, t_23284, t_23396, t_27012, ...
        recv_next_op = torch.distributed.P2POp(                                # trace_info : t_19669, t_19686, t_23397, t_23414, t_27125, ...
            torch.distributed.irecv,                                           # trace_info : t_19670, t_23398, t_27126
            tensor_recv_next,                                                  # trace_info : t_19671, t_23399, t_27127
            get_pipeline_model_parallel_next_rank(),                           # trace_info : t_19672, t_23400, t_27128
            group,                                                             # trace_info : t_19685, t_23413, t_27141
        )
        ops.append(recv_next_op)                                               # trace_info : t_19687, t_23415, t_27143
    if len(ops) > 0:                                                           # trace_info : t_19557, t_19688, t_23285, t_23416, t_27013, ...
        reqs = torch.distributed.batch_isend_irecv(ops)                        # trace_info : t_19558, t_19689, t_23286, t_23417, t_27014, ...
    else:
        reqs = []
    return reqs                                                                # trace_info : t_19559, t_19690, t_23287, t_23418, t_27015, ...


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    reqs = []
    rank = get_pipeline_model_parallel_rank()
    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)
    return reqs


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    wait_on_reqs: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Args:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None                                                    # trace_info : t_19513, t_19636, t_23241, t_23364, t_26969, ...
    tensor_recv_next = None                                                    # trace_info : t_19514, t_19637, t_23242, t_23365, t_26970, ...

    if not config.variable_seq_lengths:                                        # trace_info : t_19515, t_19638, t_23243, t_23366, t_26971, ...
        recv_prev_shape = tensor_shape                                         # trace_info : t_19516, t_19639, t_23244, t_23367, t_26972, ...
        recv_next_shape = tensor_shape                                         # trace_info : t_19517, t_19640, t_23245, t_23368, t_26973, ...
    else:
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
        )

    if recv_prev:                                                              # trace_info : t_19518, t_19641, t_23246, t_23369, t_26974, ...
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
    if recv_next:                                                              # trace_info : t_19519, t_19642, t_23247, t_23370, t_26975, ...
        if config.pipeline_dtype is None:                                      # trace_info : t_19643, t_23371, t_27099
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:                                               # trace_info : t_19644, t_23372, t_27100
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(                                        # trace_info : t_19645, t_19650, t_23373, t_23378, t_27101, ...
            recv_next_shape,                                                   # trace_info : t_19646, t_23374, t_27102
            requires_grad=True,                                                # trace_info : t_19647, t_23375, t_27103
            device=torch.cuda.current_device(),                                # trace_info : t_19648, t_23376, t_27104
            dtype=config.pipeline_dtype,                                       # trace_info : t_19649, t_23377, t_27105
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:                                           # trace_info : t_19520, t_19651, t_23248, t_23379, t_26976, ...

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:                                                # trace_info : t_19521, t_19652, t_23249, t_23380, t_26977, ...
        assert wait_on_reqs                                                    # trace_info : t_19522, t_19653, t_23250, t_23381, t_26978, ...
        p2p_func = _batched_p2p_ops                                            # trace_info : t_19523, t_19654, t_23251, t_23382, t_26979, ...
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(                                                           # trace_info : t_19524, t_19532, t_19655, t_19663, t_23252, ...
        tensor_send_prev=tensor_send_prev,                                     # trace_info : t_19525, t_19656, t_23253, t_23384, t_26981, ...
        tensor_recv_prev=tensor_recv_prev,                                     # trace_info : t_19526, t_19657, t_23254, t_23385, t_26982, ...
        tensor_send_next=tensor_send_next,                                     # trace_info : t_19527, t_19658, t_23255, t_23386, t_26983, ...
        tensor_recv_next=tensor_recv_next,                                     # trace_info : t_19528, t_19659, t_23256, t_23387, t_26984, ...
        group=get_pipeline_model_parallel_group(),                             # trace_info : t_19529, t_19660, t_23257, t_23388, t_26985, ...
    )

    if wait_on_reqs and len(reqs) > 0:                                         # trace_info : t_19560, t_19691, t_23288, t_23419, t_27016, ...
        for req in reqs:                                                       # trace_info : t_19561, t_19563, t_19692, t_19694, t_23289, ...
            req.wait()                                                         # trace_info : t_19562, t_19693, t_23290, t_23421, t_27018, ...
        reqs = None                                                            # trace_info : t_19564, t_19695, t_23292, t_23423, t_27020, ...

    if config.batch_p2p_comm and config.batch_p2p_sync:                        # trace_info : t_19565, t_19696, t_23293, t_23424, t_27021, ...
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()                                               # trace_info : t_19566, t_19697, t_23294, t_23425, t_27022, ...

    return tensor_recv_prev, tensor_recv_next, reqs                            # trace_info : t_19567, t_19698, t_23295, t_23426, t_27023, ...


def recv_forward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """

    if core.parallel_state.is_pipeline_first_stage():                          # trace_info : t_17999, t_21729, t_25457
        input_tensor = None                                                    # trace_info : t_18008, t_21738, t_25466
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensor                                                        # trace_info : t_18009, t_21739, t_25467


def recv_backward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():                           # trace_info : t_19605, t_23333, t_27061
        output_tensor_grad = None
    else:
        if config.timers is not None:                                          # trace_info : t_19620, t_23348, t_27076
            config.timers('backward-recv', log_level=2).start()                # trace_info : t_19621, t_23349, t_27077
        _, output_tensor_grad, _ = _communicate(                               # trace_info : t_19628, t_19635, t_23356, t_23363, t_27084, ...
            tensor_send_next=None,                                             # trace_info : t_19629, t_23357, t_27085
            tensor_send_prev=None,                                             # trace_info : t_19630, t_23358, t_27086
            recv_prev=False,                                                   # trace_info : t_19631, t_23359, t_27087
            recv_next=True,                                                    # trace_info : t_19632, t_23360, t_27088
            tensor_shape=tensor_shape,                                         # trace_info : t_19633, t_23361, t_27089
            config=config,                                                     # trace_info : t_19634, t_23362, t_27090
        )
        if config.timers is not None:                                          # trace_info : t_19699, t_23427, t_27155
            config.timers('backward-recv').stop()                              # trace_info : t_19700, t_23428, t_27156
    return output_tensor_grad                                                  # trace_info : t_19708, t_23436, t_27164


def send_forward(output_tensor: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not core.parallel_state.is_pipeline_last_stage():                       # trace_info : t_19482, t_23210, t_26938
        if config.timers is not None:                                          # trace_info : t_19497, t_23225, t_26953
            config.timers('forward-send', log_level=2).start()                 # trace_info : t_19498, t_23226, t_26954
        _communicate(                                                          # trace_info : t_19505, t_19512, t_23233, t_23240, t_26961, ...
            tensor_send_next=output_tensor,                                    # trace_info : t_19506, t_23234, t_26962
            tensor_send_prev=None,                                             # trace_info : t_19507, t_23235, t_26963
            recv_prev=False,                                                   # trace_info : t_19508, t_23236, t_26964
            recv_next=False,                                                   # trace_info : t_19509, t_23237, t_26965
            tensor_shape=None,                                                 # trace_info : t_19510, t_23238, t_26966
            config=config,                                                     # trace_info : t_19511, t_23239, t_26967
        )
        if config.timers is not None:                                          # trace_info : t_19568, t_23296, t_27024
            config.timers('forward-send').stop()                               # trace_info : t_19569, t_23297, t_27025


def send_backward(input_tensor_grad: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not core.parallel_state.is_pipeline_first_stage():                      # trace_info : t_19789, t_23517, t_27245
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send').stop()


def send_forward_recv_backward(
    output_tensor: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('backward-send-backward-recv', log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers('backward-send-backward-recv').stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
