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
    args = get_args()                                                          # trace_info : t_17663, t_21246, t_88853, t_92498, t_156131
    if not args.async_save:                                                    # trace_info : t_17667, t_21250, t_88857, t_92502, t_156135
        return                                                                 # trace_info : t_17668, t_21251, t_88858, t_92503, t_156136

    if blocking and _async_calls_queue.get_num_unfinalized_calls() > 0:
        print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

    _async_calls_queue.maybe_finalize_async_calls(blocking)
