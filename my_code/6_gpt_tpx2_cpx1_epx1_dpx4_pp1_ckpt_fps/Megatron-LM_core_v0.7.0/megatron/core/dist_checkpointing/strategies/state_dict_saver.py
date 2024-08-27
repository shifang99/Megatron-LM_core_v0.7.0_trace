# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" State dict saver for PyT Distributed format allowing asynchronous save. """# trace_info : t_31501

from logging import getLogger                                                  # trace_info : t_31502
from time import time                                                          # trace_info : t_31503
from typing import TYPE_CHECKING, Optional, Tuple, cast                        # trace_info : t_31504

import torch                                                                   # trace_info : t_31505
import torch.distributed as dist                                               # trace_info : t_31506
from torch.distributed.checkpoint import CheckpointException                   # trace_info : t_31507
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner    # trace_info : t_31508
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata    # trace_info : t_31509
from torch.distributed.checkpoint.planner import SavePlanner                   # trace_info : t_31510
from torch.distributed.checkpoint.utils import _DistWrapper, _get_failure_dict # trace_info : t_31511

if TYPE_CHECKING:                                                              # trace_info : t_31512
    from .filesystem_async import FileSystemWriterAsync


logger = getLogger(__name__)                                                   # trace_info : t_31513


def save_state_dict_async_plan(                                                # trace_info : t_31517, t_31519, t_31521, t_31523, t_31525, ...
    state_dict: STATE_DICT_TYPE,                                               # trace_info : t_31518
    storage_writer: 'FileSystemWriterAsync',                                   # trace_info : t_31520
    process_group: Optional[dist.ProcessGroup] = None,                         # trace_info : t_31514, t_31522
    coordinator_rank: int = 0,                                                 # trace_info : t_31515, t_31524
    planner: Optional[SavePlanner] = None,                                     # trace_info : t_31516, t_31526
) -> Tuple['FileSystemWriterAsync', Metadata, _DistWrapper]:                   # trace_info : t_31528
    """
    First stage of saving a state dict to storage.

    This is an async adjustment of torch.distributed.checkpoint.state_dict_saver.
    In order to support async save, saving should be split into three parts:
    1. Planning
    2. Actual saving
    3. Finalization

    Out of these, step (2) *must* happen asynchronously.
    The first step is realized with this function.

    The planning part consists of several steps, described here:
    https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner

    Args:
        state_dict (STATE_DICT_TYPE): state dict to save
        storage_writer (FileSystemWriterAsync): in current version only an instance of
            FileSystemWriterAsync
        process_group (dist.ProcessGroup, optional): process group used for save planning
        coordinator_rank (int, optional): coordinator rank for planning. Defaults to 0.
        planner (SavePlanner, optional): save planner for torch.distributed.checkpoint format

    Returns: Tuple of:
        - storage writer (the one passed as input)
        - metadata from planning
        - distributed wrapper used for planning
    The return value of this function should be passed as an input to
    `save_state_dict_async_finalize`.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0# trace_info : t_88298, t_155665
    dist_wrapper = _DistWrapper(process_group, True, coordinator_rank)         # trace_info : t_88299, t_155666
    if planner is None:                                                        # trace_info : t_88300, t_155667
        planner = DefaultSavePlanner()
    assert planner is not None                                                 # trace_info : t_88301, t_155668

    global_metadata = None                                                     # trace_info : t_88302, t_155669

    def local_step():                                                          # trace_info : t_88303, t_155670
        assert planner is not None                                             # trace_info : t_88307, t_155674
        planner.set_up_planner(state_dict, dist_wrapper.is_coordinator)        # trace_info : t_88308, t_155675
        storage_writer.set_up_storage_writer(dist_wrapper.is_coordinator)      # trace_info : t_88309, t_155676
        local_plan = planner.create_local_plan()                               # trace_info : t_88310, t_155677
        local_plan = storage_writer.prepare_local_plan(local_plan)             # trace_info : t_88319, t_155686
        return local_plan                                                      # trace_info : t_88320, t_155687

    def global_step(all_local_plans):                                          # trace_info : t_88304, t_155671
        nonlocal global_metadata

        assert planner is not None                                             # trace_info : t_88321, t_155688
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)# trace_info : t_88322, t_155689
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)  # trace_info : t_88323, t_155690
        return all_local_plans                                                 # trace_info : t_88324, t_155691

    # Execute local and global planning
    start_plan = time()                                                        # trace_info : t_88305, t_155672
    central_plan = dist_wrapper.reduce_scatter("plan", local_step, global_step)# trace_info : t_88306, t_155673
    logger.debug(f"rank: {rank}, plan time: {time() - start_plan}")            # trace_info : t_88325, t_155692

    # Prepare async writing of tensors.
    # The `storage_writer` will store the information about tensors it needs to save
    start = time()                                                             # trace_info : t_88326, t_155693
    final_local_plan = planner.finish_plan(central_plan)                       # trace_info : t_88327, t_155694
    storage_writer.prepare_write_data(final_local_plan, planner)               # trace_info : t_88328, t_155695
    end = time()                                                               # trace_info : t_88663, t_156030
    logger.debug(f"{time()} rank: {rank}, write(async) time: {end - start}")   # trace_info : t_88664, t_156031
    return storage_writer, cast(Metadata, global_metadata), dist_wrapper       # trace_info : t_88665, t_156032


def save_state_dict_async_finalize(                                            # trace_info : t_31530, t_31532, t_31534, t_31536, t_31538
    storage_writer: 'FileSystemWriterAsync', global_metadata: Metadata, dist_wrapper: _DistWrapper,# trace_info : t_31531, t_31533, t_31535
) -> None:                                                                     # trace_info : t_31537
    """
    Finalization of save_state_dict_async_plan.

    The input arguments are the same as the save_state_dict_async_plan output,
    the `write_results` are retrieved from the storage_writer.

    Args:
        storage_writer (FileSystemWriterAsync): storage writer used for planning
        global_metadata (Metadata): metadata created during planning
        dist_wrapper (_DistWrapper): distributed wrapper created during planning

    Returns: None
    """
    write_results = storage_writer.retrieve_write_results()                    # trace_info : t_88700, t_156067

    # Gather the write results that will be saved to the metadata file.
    gather_start = time()                                                      # trace_info : t_88707, t_156074
    all_results = dist_wrapper.gather_object(write_results)                    # trace_info : t_88708, t_156075
    gather_end = time()                                                        # trace_info : t_88709, t_156076
    logger.debug(f"{gather_end}, {torch.distributed.get_rank()}, gather: {gather_end-gather_start}")# trace_info : t_88710, t_156077

    # Store the metadata on coordinator rank
    if dist_wrapper.is_coordinator:                                            # trace_info : t_88711, t_156078
        node_failures = _get_failure_dict(all_results)                         # trace_info : t_88712, t_156079
        if len(node_failures) == 0:                                            # trace_info : t_88713, t_156080
            assert global_metadata is not None                                 # trace_info : t_88714, t_156081
            write_start = time()                                               # trace_info : t_88715, t_156082
            storage_writer.finish(global_metadata, all_results)                # trace_info : t_88716, t_156083
            write_end = time()                                                 # trace_info : t_88717, t_156084
            logger.debug(f"{write_end}, metadata_write: {write_end - write_start}")# trace_info : t_88718, t_156085
        else:
            raise CheckpointException("write", node_failures)
