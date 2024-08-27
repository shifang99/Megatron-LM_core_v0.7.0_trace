# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Entrypoints for saving and loading the distributed checkpoints.

Functions `load` and `save` are equivalents of `torch.load` and `torch.save`
but expect torch.Tensors to be wrapped with classes from the `mapping module`.
Additionally, `load` expects the sharded state dict argument as a guidance for loading the sharded tensors.
"""

import logging
import os
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .core import CheckpointingConfig, maybe_load_config, save_config
from .dict_utils import (
    dict_list_map_inplace,
    diff,
    extract_matching_values,
    map_reduce,
    merge,
    nested_values,
)
from .mapping import (
    CheckpointingException,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
    apply_factories,
    apply_factory_merges,
    is_main_replica,
)
from .strategies.async_utils import AsyncRequest
from .strategies.base import (
    AsyncSaveShardedStrategy,
    LoadCommonStrategy,
    LoadShardedStrategy,
    SaveCommonStrategy,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)
from .utils import (
    extract_nonpersistent,
    extract_sharded_base,
    extract_sharded_tensors,
    extract_sharded_tensors_or_nonpersistent,
)

COMMON_STATE_FNAME = 'common.pt'

logger = logging.getLogger(__name__)


def load(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
) -> StateDict:
    """Loading entrypoint.

    In the steps below, the following verbs refer to corresponding objects:
    - load = load from checkpoint
    - extract = extract from sharded_state_dict
    - add = add to the final state dict
    Steps:
    1. Load common state dict and form the base of the result state dict
    2. Apply factories to sharded_state_dict
    3. Extract LocalNonPersistentObject and add
    4. (optional) Extract ShardedObjects, load and add
    5. Extract ShardedBase, load, apply factory merges and add

    Args:
        sharded_state_dict (ShardedStateDict): state dict of the existing model
            populated with ShardedTensors. Used as a mapping to determine which
            parts of global tensors stored in the checkpoint should be loaded.
        checkpoint_dir (str): directory with the checkpoint
        sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional): configures loading behavior for sharded tensors
        common_strategy (LoadCommonStrategy, Tuple[str, int], optional): configures loading behavior for common data
        validate_access_integrity (bool default = True): checks if each tensor shard is accessed
            exactly once (as main replica) by some process
    """
    if common_strategy is not None:
        raise NotImplementedError('The only supported common strategy is torch')

    sharded_strategy = _verify_checkpoint_and_load_strategy(checkpoint_dir, sharded_strategy)

    checkpoint_dir = Path(checkpoint_dir)
    common_state_dict = load_common_state_dict(checkpoint_dir)
    if not sharded_state_dict:
        return common_state_dict

    # Create a copy of sharded_state_dict as the passed in state dict may have
    # references that prevent tensors from being deallocated
    sharded_state_dict, _ = extract_matching_values(sharded_state_dict, lambda x: True)

    sh_ten_factories, _ = extract_matching_values(
        sharded_state_dict,
        lambda x: isinstance(x, ShardedTensorFactory),
        return_lists_as_dicts=True,
    )
    apply_factories(sharded_state_dict)
    # Data inside sh_ten_factories no longer needed so delete them to reduce memory usage
    def unlink_data(x):
        x.data = None
        return x

    dict_list_map_inplace(unlink_data, sh_ten_factories)
    # Non-persistent objects
    nonpersistent_state_dict, sharded_state_dict = extract_nonpersistent(sharded_state_dict)
    dict_list_map_inplace(lambda o: o.unwrap(), nonpersistent_state_dict)
    merge(common_state_dict, nonpersistent_state_dict)

    # Sharded base
    if not sharded_strategy.can_handle_sharded_objects:
        # TODO: implement is a part of common strategy
        sharded_objects, sharded_state_dict = load_sharded_objects(
            sharded_state_dict, checkpoint_dir
        )
        merge(common_state_dict, sharded_objects)
    sharded_state_dict, _ = extract_sharded_base(sharded_state_dict)

    if validate_access_integrity:
        validate_sharding_integrity(nested_values(sharded_state_dict))

    loaded_state_dict = sharded_strategy.load(sharded_state_dict, checkpoint_dir)

    loaded_state_dict = apply_factory_merges(loaded_state_dict, sh_ten_factories)

    merge(common_state_dict, loaded_state_dict)
    return common_state_dict


def _verify_checkpoint_and_load_strategy(
    checkpoint_dir: str, sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
) -> LoadShardedStrategy:
    """ Verifies if checkpoint metadata exists and matches given strategy.

    Args:
        checkpoint_dir (str): checkpoint directory
        sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional): load strategy to be verified
            if compatible with the checkpoint content. If None, the default load strategy
            for the checkpoint backend will be returned.
    """
    if not Path(checkpoint_dir).exists():
        raise CheckpointingException(f'Checkpoint directory {checkpoint_dir} does not exist')

    saved_config = maybe_load_config(checkpoint_dir)
    if saved_config is None:
        raise CheckpointingException(f'{checkpoint_dir} is not a distributed checkpoint')

    if sharded_strategy is None:
        sharded_strategy = get_default_strategy(
            StrategyAction.LOAD_SHARDED,
            saved_config.sharded_backend,
            saved_config.sharded_backend_version,
        )
    elif isinstance(sharded_strategy, tuple):
        sharded_strategy = get_default_strategy(StrategyAction.LOAD_SHARDED, *sharded_strategy)

    # TODO: implement consistency checks here
    return sharded_strategy


# TODO: implement it as common torch strategy
def load_common_state_dict(checkpoint_dir: Path) -> StateDict:
    """ Load common (non-sharded) objects state dict from the checkpoint.

    Args:
        checkpoint_dir (Path): checkpoint directory

    Returns:
        StateDict: state dict with non-sharded objects from the checkpoint
    """
    load_path = Path(checkpoint_dir) / COMMON_STATE_FNAME
    try:
        return torch.load(load_path, map_location='cpu')
    except FileNotFoundError as e:
        err_msg = f'Common file {load_path} does not exist'
        ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
        logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
        raise CheckpointingException(err_msg) from e


def load_sharded_objects(sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
    """ Replaces all ShardedObject from a given state dict with values loaded from the checkpoint.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict defining what objects should be loaded.
        checkpoint_dir (Path): checkpoint directory

    Returns:
        None: state dict is modified in place
    """
    sharded_objects, sharded_state_dict = extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, ShardedObject)
    )

    def load_sharded_object(sh_obj: ShardedObject):
        sh_obj.data = None
        load_path = (checkpoint_dir / sh_obj.unique_key).with_suffix('.pt')
        try:
            loaded_obj = torch.load(load_path)
        except FileNotFoundError as e:
            err_msg = f'Object shard {load_path} not found'
            obj_subdir = checkpoint_dir / sh_obj.key
            if obj_subdir.exists():
                obj_files = [f.name for f in obj_subdir.iterdir()]
                logger.debug(f'{err_msg}. Object {sh_obj.key} directory content: {obj_files}')
            else:
                ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
                logger.debug(
                    f'{err_msg}. Object {sh_obj.key} directory does not exist. Checkpoint directory content: {ckpt_files}'
                )
            raise CheckpointingException(err_msg) from e
        return loaded_obj

    return dict_list_map_inplace(load_sharded_object, sharded_objects), sharded_state_dict


def load_tensors_metadata(
    checkpoint_dir: str, sharded_strategy: Union[LoadShardedStrategy, None] = None
) -> ShardedStateDict:
    """Load tensors metadata from the checkpoint.

    Returns a dictionary similar to a sharded state dict, but note that
    the dictionary keys are simply ShardedTensor keys (contrary to the
    actual sharded state dicts where keys correspond to state dict keys).

    Dict values are ShardedTensors without any sharding (so, the only useful
    information is tensors global shape and dtype).

    Concrete implementation depends on the loading strategy. If no strategy is
    given, a default for a given backend is used.
    """
    sharded_strategy = _verify_checkpoint_and_load_strategy(checkpoint_dir, sharded_strategy)
    return sharded_strategy.load_tensors_metadata(Path(checkpoint_dir))


def load_plain_tensors(checkpoint_dir: str):
    """Load checkpoint tensors without any sharding.

    NOTE: common state dict is NOT included."""
    sharded_state_dict = load_tensors_metadata(checkpoint_dir)
    # Don't validate integrity because shards will be overlapped
    # if world_size > 1 (all processes load whole tensors)
    return load(sharded_state_dict, checkpoint_dir, validate_access_integrity=False)


def save(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[SaveShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[SaveCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
    async_sharded_save: bool = False,
) -> Optional[AsyncRequest]:
    """Saving entrypoint.

    Extracts ShardedTensors from the given state dict. Rank 0 saves the
    "regular" part of the checkpoint to common torch file.
    The ShardedTensors are saved according to a strategy specified by the
    config.

    Steps:
    1. Apply factories
    2. Extract and discard LocalNonPersistentObject
    3. Extract all ShardedBase object
    4. Save all other objects to common.pt
    5. (optional) Extract and save ShardedObjects
    6. Save all ShardedBase objects
    7. Write metadata.json file with backend and version metadata.

    Step (6) can be performed asynchronously (see `async_sharded_save`), in this
    case the actual save is embodied in the returned async request and can be
    scheduled by the external caller. For async request, step (7) is added as
    one of the finalization functions, so that metadata.json is written only
    if the checkpoint is complete.

    Args:
        sharded_state_dict (ShardedStateDict): state dict of the populated with
            ShardedTensors. Used as a mapping to determine how local tensors
            should be saved as global tensors in the checkpoint.
        checkpoint_dir (str): directory to save the checkpoint to
        sharded_strategy (SaveShardedStrategy, Tuple[str, int], optional): configures sharded tensors saving behavior and backend
        common_strategy (SaveCommonStrategy, Tuple[str, int], optional): configures common data saving behavior and backend
        validate_access_integrity (bool default = True): checks if each tensor shard is accessed
            exactly once (as main replica) by some process
        async_sharded_save (bool, optional): if True, for the sharded state dict part
            an async save implementation will be called, with the AsyncRequest
            being returned to the caller. Note that it is the caller responsibility to
            actually schedule the async save. Defaults to False.

    Returns:
        AsyncRequest (optional): if `async_sharded_save` is True, returns
            async request that should be scheduled by the caller of this function.
            None otherwise.
    """
    checkpoint_dir = Path(checkpoint_dir)                                      # trace_info : t_31665, t_99032

    if torch.distributed.get_rank() == 0:                                      # trace_info : t_31666, t_99033
        if not checkpoint_dir.exists():                                        # trace_info : t_31667, t_99034
            raise CheckpointingException(
                f'Checkpoint destination directory does not exist: {checkpoint_dir}'
            )

        if next(checkpoint_dir.iterdir(), None) is not None:                   # trace_info : t_31668, t_99035
            raise CheckpointingException(
                f'Checkpoint destination directory ({checkpoint_dir}) is not empty'
            )

    if common_strategy is not None:                                            # trace_info : t_31669, t_99036
        raise NotImplementedError('The only supported common strategy is torch')

    if sharded_strategy is None:                                               # trace_info : t_31670, t_99037
        sharded_strategy = get_default_save_sharded_strategy()
    if not isinstance(sharded_strategy, SaveShardedStrategy):                  # trace_info : t_31671, t_99038
        assert isinstance(sharded_strategy, tuple), type(sharded_strategy)
        sharded_strategy = get_default_strategy(StrategyAction.SAVE_SHARDED, *sharded_strategy)

    apply_factories(sharded_state_dict)                                        # trace_info : t_31672, t_99039
    _, sharded_state_dict = extract_nonpersistent(sharded_state_dict)          # trace_info : t_33002, t_100369
    sharded_state_dict, state_dict = extract_sharded_base(sharded_state_dict)  # trace_info : t_34389, t_101756
    _save_common_dict(state_dict, checkpoint_dir, True)                        # trace_info : t_35757, t_103124

    if validate_access_integrity:                                              # trace_info : t_35762, t_103129
        validate_sharding_integrity(list(nested_values(sharded_state_dict)))   # trace_info : t_35763, t_103130

    if not sharded_strategy.can_handle_sharded_objects:                        # trace_info : t_55256, t_122623
        # TODO: implement is a part of common strategy
        sharded_state_dict = _extract_and_save_sharded_objects(
            sharded_state_dict, checkpoint_dir, validate_access_integrity
        )

    def metadata_finalize_fn():                                                # trace_info : t_55258, t_122625
        if torch.distributed.get_rank() == 0:                                  # trace_info : t_88722, t_156089
            save_config(                                                       # trace_info : t_88723, t_88730, t_156090, t_156097
                CheckpointingConfig(sharded_strategy.backend, sharded_strategy.version),# trace_info : t_88724, t_156091
                checkpoint_dir,                                                # trace_info : t_88729, t_156096
            )
        torch.distributed.barrier()                                            # trace_info : t_88735, t_156102

    if not async_sharded_save:                                                 # trace_info : t_55259, t_122626
        sharded_strategy.save(sharded_state_dict, checkpoint_dir)              # trace_info : t_55260, t_122627
        metadata_finalize_fn()                                                 # trace_info : t_88721, t_156088
        return                                                                 # trace_info : t_88736, t_156103

    if not isinstance(sharded_strategy, AsyncSaveShardedStrategy):
        raise CheckpointingException(
            f'Cannot apply async_save to non-async strategy {sharded_strategy}'
        )
    async_request = sharded_strategy.async_save(sharded_state_dict, checkpoint_dir)
    async_request.finalize_fns.append(metadata_finalize_fn)
    return async_request


def get_default_save_sharded_strategy(
    backend: str = 'torch_dist', version: int = 1
) -> SaveShardedStrategy:
    return get_default_strategy(StrategyAction.SAVE_SHARDED, backend, version) # trace_info : t_31422, t_99015


def get_default_load_sharded_strategy(checkpoint_dir: str) -> LoadShardedStrategy:
    return _verify_checkpoint_and_load_strategy(checkpoint_dir)


# TODO: implement it as common torch strategy
def _save_common_dict(
    state_dict: StateDict, checkpoint_dir: Path, validate_consistency: bool = False
):
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_35758, t_103125
        torch.save(state_dict, checkpoint_dir / COMMON_STATE_FNAME)            # trace_info : t_35759, t_103126
    if validate_consistency:                                                   # trace_info : t_35760, t_103127
        # TODO: implement checking consistency with rank 0 common dict on other ranks
        pass                                                                   # trace_info : t_35761, t_103128
        # torch.distributed.barrier()
        # if not torch.distributed.get_rank() == 0:
        #     rank_0_state_dict = torch.load(checkpoint_dir / COMMON_STATE_FNAME)
        #     print(diff(common_state_dict, rank_0_state_dict))


def _extract_and_save_sharded_objects(
    state_dict: StateDict, checkpoint_dir: Path, validate_consistency: bool = False
):
    sharded_objects, state_dict = extract_matching_values(
        state_dict, lambda v: isinstance(v, ShardedObject)
    )
    sharded_objects = list(nested_values(sharded_objects))
    for sh_obj in sharded_objects:
        if is_main_replica(sh_obj.replica_id):
            save_path = (checkpoint_dir / sh_obj.unique_key).with_suffix('.pt')
            os.makedirs(save_path.parent, exist_ok=True)
            torch.save(sh_obj.data, save_path)
    return state_dict


def validate_sharding_integrity(sharded_tensors: Iterable[ShardedTensor]):
    """ Validate if the ShardedTensors from multiple processes define correct sharding of a global tensor.

    Local ShardedTensors metadata is exchanged with `torch.distributed.all_gather_object`
    and then process with global rank 0 checks if main replicas of the shards:
    - cover the whole global tensors
    - don't overlap

    Args:
        sharded_tensors (Iterable[ShardedTensor]): sharded tensors local to this process

    Returns:
        None

    Raises:
        CheckpointingException for invalid access pattern
    """
    sharding = [ten.without_data() for ten in sharded_tensors]                 # trace_info : t_36304, t_69243, t_103671, t_136610
    all_sharding = [None] * torch.distributed.get_world_size()                 # trace_info : t_37703, t_71182, t_105070, t_138549
    torch.distributed.all_gather_object(all_sharding, sharding)                # trace_info : t_37704, t_71183, t_105071, t_138550
    if torch.distributed.get_rank() != 0:                                      # trace_info : t_37705, t_71184, t_105072, t_138551
        return

    key_shardings = defaultdict(list)                                          # trace_info : t_37706, t_71185, t_105073, t_138552
    for rank, rank_shardings in enumerate(all_sharding):                       # trace_info : t_37707, t_37951, t_38195, t_38439, t_38683, ...
        for sharding in rank_shardings:                                        # trace_info : t_37708, t_37710, t_37712, t_37714, t_37716, ...
            key_shardings[sharding.key].append((rank, sharding))               # trace_info : t_37709, t_37711, t_37713, t_37715, t_37717, ...
    for key, shardings in key_shardings.items():                               # trace_info : t_39660, t_39820, t_39953, t_40195, t_40437, ...
        if isinstance(shardings[0][1], ShardedObject):                         # trace_info : t_39661, t_39821, t_39954, t_40196, t_40438, ...
            _validate_objects_for_key(shardings)                               # trace_info : t_40987, t_41668, t_42833, t_43466, t_55230, ...
        else:
            _validate_sharding_for_key(shardings)                              # trace_info : t_39662, t_39822, t_39955, t_40197, t_40439, ...


def _validate_sharding_for_key(rank_sharding: List[Tuple[int, ShardedTensor]]):
    some_rank_shard = rank_sharding[0][1]                                      # trace_info : t_39663, t_39823, t_39956, t_40198, t_40440, ...
    global_shape = some_rank_shard.global_shape                                # trace_info : t_39664, t_39824, t_39957, t_40199, t_40441, ...
    local_shape = some_rank_shard.local_shape                                  # trace_info : t_39665, t_39825, t_39958, t_40200, t_40442, ...
    dtype = some_rank_shard.dtype                                              # trace_info : t_39666, t_39826, t_39959, t_40201, t_40443, ...
    has_flattened_range = some_rank_shard.flattened_range is not None          # trace_info : t_39667, t_39827, t_39960, t_40202, t_40444, ...
    for rank, sharding in rank_sharding:                                       # trace_info : t_39668, t_39673, t_39678, t_39683, t_39688, ...
        assert sharding.dtype == dtype, (sharding.dtype, dtype, some_rank_shard)# trace_info : t_39669, t_39674, t_39679, t_39684, t_39689, ...
        assert sharding.global_shape == global_shape, (                        # trace_info : t_39670, t_39675, t_39680, t_39685, t_39690, ...
            sharding.global_shape,
            global_shape,
            some_rank_shard,
        )
        assert sharding.local_shape == local_shape, (                          # trace_info : t_39671, t_39676, t_39681, t_39686, t_39691, ...
            sharding.local_shape,
            local_shape,
            some_rank_shard,
        )
        assert (sharding.flattened_range is not None) == has_flattened_range, (# trace_info : t_39672, t_39677, t_39682, t_39687, t_39692, ...
            (sharding.flattened_range is not None),
            has_flattened_range,
            some_rank_shard,
        )

    shard_access_cnt = _compute_shards_access(rank_sharding)                   # trace_info : t_39709, t_39869, t_40042, t_40284, t_40526, ...
    if has_flattened_range:                                                    # trace_info : t_39818, t_39951, t_40193, t_40435, t_40741, ...
        map_reduce(
            rank_sharding,
            lambda x: x[1].global_offset,
            lambda x: x[1],
            _validate_sharding_for_key_flattened,
        )
    else:
        if not torch.all(shard_access_cnt == 1):                               # trace_info : t_39819, t_39952, t_40194, t_40436, t_40742, ...
            logger.error(f'Invalid access pattern for {rank_sharding[0][1]}: {shard_access_cnt}')
            raise CheckpointingException(f'Invalid access pattern for {rank_sharding[0][1]}')


def _compute_shards_access(rank_sharding):
    def chunk_offset(sharding):                                                # trace_info : t_39710, t_39870, t_40043, t_40285, t_40527, ...
        assert len(sharding.global_offset) == len(sharding.local_shape) + sharding.prepend_axis_num# trace_info : t_39723, t_39753, t_39883, t_40056, t_40083, ...
        return tuple(                                                          # trace_info : t_39724, t_39733, t_39754, t_39763, t_39884, ...
            chain(                                                             # trace_info : t_39725, t_39732, t_39755, t_39762, t_39885, ...
                (off for off in sharding.global_offset[: sharding.prepend_axis_num]),# trace_info : t_39726, t_39734, t_39756, t_39764, t_39886, ...
                (                                                              # trace_info : t_39727, t_39731, t_39735, t_39738, t_39739, ...
                    off // sh                                                  # trace_info : t_39737, t_39741, t_39767, t_39771, t_39897, ...
                    for off, sh in zip(                                        # trace_info : t_39728, t_39730, t_39736, t_39740, t_39758, ...
                        sharding.global_offset[sharding.prepend_axis_num :], sharding.local_shape# trace_info : t_39729, t_39759, t_39889, t_40062, t_40089, ...
                    )
                ),
            )
        )

    shard_access_cnt = torch.zeros(                                            # trace_info : t_39711, t_39713, t_39871, t_39873, t_40044, ...
        rank_sharding[0][1].axis_fragmentations, dtype=torch.int, device='cpu' # trace_info : t_39712, t_39872, t_40045, t_40287, t_40529, ...
    )
    for rank, sharding in rank_sharding:                                       # trace_info : t_39714, t_39744, t_39774, t_39781, t_39788, ...
        if is_main_replica(sharding.replica_id):                               # trace_info : t_39715, t_39745, t_39775, t_39782, t_39789, ...
            shard_access_cnt[chunk_offset(sharding)] += 1                      # trace_info : t_39722, t_39752, t_39882, t_40055, t_40082, ...
        # TODO: consider validating different replicas too
    return shard_access_cnt                                                    # trace_info : t_39817, t_39950, t_40192, t_40434, t_40740, ...


def _validate_sharding_for_key_flattened(tensors_by_shard):
    all_slices = []
    local_shape = tensors_by_shard[0].local_shape
    for sharding in tensors_by_shard:
        assert sharding.local_shape == local_shape
        sharding: ShardedTensor
        if not is_main_replica(sharding.replica_id):
            # TODO: this checks only saving (and loading replica_id=0) consistency
            continue

        all_slices.append((sharding.flattened_range.start, sharding.flattened_range.stop))

    starts, stops = map(np.asarray, zip(*sorted(all_slices)))
    if (
        starts[0] != 0
        or stops[-1] != np.product(local_shape)
        or not np.all(starts[1:] == stops[:-1])
    ):
        logger.error(
            f'Flattened ranges dont cover the whole shard {tensors_by_shard[0]}. Ranges: {(starts, stops)}'
        )
        raise CheckpointingException(
            f'Flattened ranges dont cover the whole shard {tensors_by_shard[0]}'
        )


def _validate_objects_for_key(sharded_objects: List[ShardedObject]):
    """ Ensure uniqueness of saved objects. """
    unique_keys = [                                                            # trace_info : t_40988, t_40990, t_41669, t_41671, t_42834, ...
        sh_obj.unique_key for _, sh_obj in sharded_objects if is_main_replica(sh_obj.replica_id)# trace_info : t_40989, t_41670, t_42835, t_43468, t_55232, ...
    ]
    if len(unique_keys) != len(set(unique_keys)):                              # trace_info : t_41067, t_41748, t_42913, t_43546, t_55252, ...
        duplicates = {k: cnt for k, cnt in Counter(unique_keys).items() if cnt > 1}
        logger.error(f'Duplicate ShardedObject keys and counts: {duplicates}')
        raise CheckpointingException(f'Duplicate ShardedObject keys: {list(duplicates.keys())}')
    expected_shard_num = np.prod(sharded_objects[0][1].global_shape)           # trace_info : t_41068, t_41749, t_42914, t_43547, t_55253, ...
    if len(unique_keys) != expected_shard_num:                                 # trace_info : t_41069, t_41750, t_42915, t_43548, t_55254, ...
        err_msg = f'Invalid access pattern: {expected_shard_num - len(unique_keys)} ShardedObject are missing.'
        logger.error(f'{err_msg} Existing shards: {unique_keys}')
        raise CheckpointingException(err_msg)
