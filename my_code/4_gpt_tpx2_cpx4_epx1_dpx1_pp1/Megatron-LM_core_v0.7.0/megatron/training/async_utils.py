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
    args = get_args()                                                          # trace_info : t_17385, t_20515, t_23701, t_26895, t_26914
    if not args.async_save:                                                    # trace_info : t_17389, t_20519, t_23705, t_26899, t_26918
        return                                                                 # trace_info : t_17390, t_20520, t_23706, t_26900, t_26919

    if blocking and _async_calls_queue.get_num_unfinalized_calls() > 0:
        print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

    _async_calls_queue.maybe_finalize_async_calls(blocking)
