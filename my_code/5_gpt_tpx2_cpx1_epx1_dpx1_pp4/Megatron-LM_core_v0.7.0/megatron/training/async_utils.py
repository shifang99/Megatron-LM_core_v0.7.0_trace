# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
This module provides a singleton instance of AsyncCallsQueue which manages
the async checkpoint save calls.
"""
import logging

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest
from megatron.training import get_args
from megatron.training.utils import print_rank_0

logger = logging.getLogger(__name__)

# Singleton manager of async calls
_async_calls_queue = AsyncCallsQueue()


def schedule_async_save(async_request: AsyncRequest):
    """ Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    _async_calls_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(blocking: bool = False):
    """ Finalizes active async save calls.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
    """
    args = get_args()                                                          # trace_info : t_17592, t_21270, t_24998, t_28734, t_28753
    if not args.async_save:                                                    # trace_info : t_17596, t_21274, t_25002, t_28738, t_28757
        return                                                                 # trace_info : t_17597, t_21275, t_25003, t_28739, t_28758

    if blocking and _async_calls_queue.get_num_unfinalized_calls() > 0:
        print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

    _async_calls_queue.maybe_finalize_async_calls(blocking)
