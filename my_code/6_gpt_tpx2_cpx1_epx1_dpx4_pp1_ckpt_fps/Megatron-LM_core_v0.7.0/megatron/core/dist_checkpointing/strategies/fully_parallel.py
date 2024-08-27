import logging                                                                 # trace_info : t_66861, t_134228
from collections import defaultdict
from functools import reduce
from itertools import zip_longest
from pathlib import Path
from time import time
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, TypeVar, cast

import numpy as np
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import (
    dict_list_map_inplace,
    extract_matching_values,
    merge,
    nested_values,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, StateDict, is_main_replica
from megatron.core.dist_checkpointing.serialization import validate_sharding_integrity
from megatron.core.dist_checkpointing.strategies.base import (
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
    SaveShardedStrategy,
)

logger = logging.getLogger(__name__)


# _ShardId uniquely identifies a ShardedTensor. This is a subset of ShardedTensor
# attributes: key (str), global_offset (tuple) and flattened_range (optional tuple)
_ShardId = Tuple[str, tuple, Optional[tuple]]


class SaveLoadDistribution(NamedTuple):
    """ Represents a save or load distribution of ShardedTensors.

    Given distribution is valid only for a specific parallelization group,
    which is implicit here (not referenced by this class).

    Args:
        main_rank_for_shard (Dict[_ShardId, int]): specifies which rank should hold
            the main replica for a given shard
        shards_in_this_group (Set[_ShardId]): which shards have a main replica
            in this parallelization group
        shard_to_metadata (Dict[_ShardId, ShardedTensor]): maps ShardedTensor
            identifier to the original ShardedTensor

    """

    main_rank_for_shard: Dict[_ShardId, int]
    shards_in_this_group: Set[_ShardId]
    shard_to_metadata: Dict[_ShardId, ShardedTensor]


class FullyParallelSaveStrategyWrapper(AsyncSaveShardedStrategy):
    """ Wraps arbitrary strategy and distributes the save during `save`.

    The save distribution happens without any *data* communication.
    Only the *metadata* is exchanged and based on data replication on different
    ranks, we try to distribute the save as uniformly as possible.

    This wrapper assumes, that setting `replica_id` to 0 will make the
    underlying strategy do the saving on current rank. All the other `replica_id`s
    are set to 1.

    Currently, the save distribution is realized with a greedy algorithm
    described in `distribute_shards_to_ranks`.

    Args:
        strategy (SaveShardedStrategy): base strategy to wrap
        parallelization_group (ProcessGroup, optional): process group to use for save
            distribution. Note that this doesn't have to match exactly the
            data distribution, but should cover the replication pattern
            to maximize performance. Defaults to the whole world.
        do_cache_distribution (bool, optional): whether to cache the save distribution
            from previous calls. Should be set to True only if the state dict
            structure between the calls is always the same. Defaults to True.
    """

    def __init__(
        self,
        strategy: SaveShardedStrategy,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        do_cache_distribution: bool = False,
    ):
        super().__init__(strategy.backend, strategy.version)                   # trace_info : t_31653
        self.base_strategy = strategy                                          # trace_info : t_31656
        self.parallelization_group = parallelization_group                     # trace_info : t_31657
        self.do_cache_distribution = do_cache_distribution                     # trace_info : t_31658

        self.cached_distribution: Optional[SaveLoadDistribution] = None        # trace_info : t_31659

    def async_save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        if not isinstance(self.base_strategy, AsyncSaveShardedStrategy):
            raise CheckpointingException(
                f'Cannot apply async_save to non-async base strategy {self.base_strategy}'
            )
        self.apply_saving_parallelization(sharded_state_dict)
        return self.base_strategy.async_save(sharded_state_dict, checkpoint_dir)

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        self.apply_saving_parallelization(sharded_state_dict)                  # trace_info : t_55261, t_122628
        return self.base_strategy.save(sharded_state_dict, checkpoint_dir)     # trace_info : t_86124, t_153491

    def apply_saving_parallelization(self, sharded_state_dict: ShardedStateDict) -> None:
        """ Distributes the save across ranks by exchanging metadata.

        Exchanges metadata from the state dict and computes the uniform
        (as close as possible) distribution of saves among the ranks.

        If `self.do_cache_distribution` is True, caches the distribution between
        the calls and subsequent distributions happen without any inter-rank
        communication.

        Args:
            sharded_state_dict (ShardedStateDict): state dict to distribute the saving

        Returns: None
        """
        if self.do_cache_distribution and self.cached_distribution is not None:# trace_info : t_55262, t_122629
            logger.debug(f'Apply *cached* save parallelization')
            precomputed_distribution = self.cached_distribution
        else:
            logger.debug(f'Apply save parallelization')                        # trace_info : t_55263, t_122630
            precomputed_distribution = determine_main_replica_uniform_distribution(# trace_info : t_55264, t_55266, t_122631, t_122633
                sharded_state_dict, self.parallelization_group                 # trace_info : t_55265, t_122632
            )

        distribute_main_replicas_with_precomputed_distribution(                # trace_info : t_66862, t_66864, t_134229, t_134231
            sharded_state_dict, self.parallelization_group, precomputed_distribution# trace_info : t_66863, t_134230
        )
        if self.cached_distribution is None:                                   # trace_info : t_69241, t_136608
            # First time applying the parallelization
            validate_sharding_integrity(nested_values(sharded_state_dict))     # trace_info : t_69242, t_136609
        if self.do_cache_distribution:                                         # trace_info : t_86123, t_153490
            self.cached_distribution = precomputed_distribution

    @property
    def can_handle_sharded_objects(self):
        return self.base_strategy.can_handle_sharded_objects                   # trace_info : t_55257, t_122624


class FullyParallelLoadStrategyWrapper(LoadShardedStrategy):
    """ Wraps arbitrary load strategy and distributes the load during `load`.

    See `load` method docs for details.

    Args:
        strategy (LoadShardedStrategy): base strategy to wrap
        parallelization_group (ProcessGroup, optional): process group to use for load
            distribution. Note that this doesn't have to match exactly the
            data distribution, but should cover the replication pattern
            to maximize performance. Defaults to the whole world.
            In most cases, it's recommended to set it to the DP group.
        do_cache_distribution (bool, optional): whether to cache the load distribution
            from previous calls. Should be set to True only if the state dict
            structure between the calls is always the same. Defaults to False,
            since the loading in general happens only once during training.
            Note that the load distribution *cannot* be reused as a save distribution,
            because save/load is not fully symmetrical.
        exchange_algo (str): algorithm to use for exchanging the data.
            Options:
            - broadcast - each rank broadcasts individual tensors to others
            - gather_object (default) - ranks all_gather_object the whole loaded state dicts
            - gather_rounds (default) - ranks all gather individual tensors in rounds
            See method docs for more details.
    """

    def __init__(
        self,
        strategy: LoadShardedStrategy,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        do_cache_distribution: bool = False,
        exchange_algo: str = 'gather_rounds',
    ):
        super().__init__()
        self.base_strategy = strategy
        self.parallelization_group = parallelization_group
        self.do_cache_distribution = do_cache_distribution
        self.exchange_algo = exchange_algo

        self.cached_distribution: Optional[SaveLoadDistribution] = None

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """ Distributes the load and calls underlying strategy only for parts of the state dict.

        Steps:
        1. Load metadata is exchanged between the ranks in the parallelization group.
        2. Each rank deterministically plans the load for the whole workload
            so that the loads are as uniform as possible.
        3. Each ranks loads its planned shard of the checkpoint.
        4. All ranks exchange the loaded shards.

        Internode communication is involved in steps (1) (with metadata)
        and (4) (with actual data). Storage interaction is involved in step (3).

        Currently, the load distribution (step 2) is realized with a greedy algorithm
        described in `distribute_shards_to_ranks` (same as for saving distribution).

        Currently, the shards are all gathered between all ranks in the parallelization
        group. This might not be optimal (some ranks do not need all tensors),
        but it's a reasonable approximation for an optimal exchange in most scenarios.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to load
            checkpoint_dir (Path): checkpoint directory to load from

        Returns:
            StateDict: loaded state dict. The state dict should be equivalent to
            a state dict that would be loaded with the underlying strategy
            without this wrapper.
        """
        if torch.distributed.get_world_size(self.parallelization_group) <= 1:
            return self.base_strategy.load(sharded_state_dict, checkpoint_dir)

        # Step 1 and 2: exchange load metadata and distribute the load
        start = time()
        precomputed_distribution = self.apply_loading_parallelization(sharded_state_dict)
        assert (
            precomputed_distribution is not None
        ), 'Expecting non-trivial distribution for non-trivial parallelization group'
        end = time()
        logger.debug(f'self.apply_loading_parallelization took {end - start}s')
        start = end

        # Step 3: load part of the checkpoint.
        # Load only sharded objects first. ShardedTensors will be loaded separately
        # so that we can keep track of sharded tensors loaded by this rank
        (
            sharded_tensors,
            sharded_state_dict,
            to_load_shards,
            unloaded_shards,
        ) = self._defer_loading_sharded_tensors(sharded_state_dict)
        loaded_state_dict = self.base_strategy.load(sharded_state_dict, checkpoint_dir)

        end = time()
        logger.debug(f'Base load of ShardedObjects took {end - start}s')
        start = end

        # Load sharded tensors separately
        loaded_tensors = self.base_strategy.load(to_load_shards, checkpoint_dir)

        end = time()
        logger.debug(f'Base load of ShardedTensors took {end - start}s')
        start = end

        # Step 4: exchange data between ranks
        logger.debug(f'Applying parallel load with algo {self.exchange_algo}')
        if self.exchange_algo == 'gather_object':
            exchange_fn = self.exchange_loaded_tensors_gather_object
        elif self.exchange_algo == 'gather_rounds':
            exchange_fn = self.exchange_loaded_tensors_gather_rounds
        elif self.exchange_algo == 'broadcast':
            exchange_fn = self.exchange_loaded_tensors_broadcast
        else:
            raise NotImplementedError(f'Unrecognized gather algorithm: {self.exchange_algo}')

        all_loaded_tensors = exchange_fn(
            loaded_tensors, unloaded_shards, precomputed_distribution, self.parallelization_group,
        )
        if not set(unloaded_shards.keys()).issubset(all_loaded_tensors.keys()):
            missing_shards = set(unloaded_shards.keys()) - all_loaded_tensors.keys()
            raise CheckpointingException(
                f'Missing shards after fully parallel loading: {missing_shards}'
            )

        sync_start = time()
        torch.cuda.synchronize()
        end = time()
        logger.debug(f'torch.cuda.synchronize took {end - sync_start}s')
        logger.debug(f'self.exchange_loaded_tensors took {end - start}s')

        self.fill_in_deferred_sharded_tensors(sharded_tensors, all_loaded_tensors)
        merge(loaded_state_dict, sharded_tensors)
        return loaded_state_dict

    def _defer_loading_sharded_tensors(
        self, sharded_state_dict: ShardedStateDict
    ) -> Tuple[
        ShardedStateDict,
        ShardedStateDict,
        Dict[_ShardId, ShardedTensor],
        Dict[_ShardId, ShardedTensor],
    ]:
        """ Divides state dict into parts loaded by this vs other ranks.

        ShardedTensors with main replica_id will be loaded by this rank,
        others will be received by other ranks (after loading from storage).

        Args:
            sharded_state_dict (ShardedStateDict): state dict with ShardedTensor
                that will be divided.

        Returns: a tuple of:
            - ShardedStateDict: sub-state dict only with ShardedTensors
            - ShardedStateDict: sub-state dict with non-ShardedTensors
            - Dict[_ShardId, ShardedTensor]: ShardedTensor are uniquely identified
                by shard ids. This is a mapping from shard id to a corresponding
                ShardedTensor for tensors loaded by *this* rank
            - Dict[_ShardId, ShardedTensor]: mapping from shard id to a corresponding
                ShardedTensor for tensors loaded by *other* ranks
        """
        to_load_shards = {}
        unloaded_shards = {}

        sharded_tensors, sharded_state_dict = extract_matching_values(
            sharded_state_dict, lambda v: isinstance(v, ShardedTensor)
        )

        def wrap_non_main_replicas(x):
            if isinstance(x, ShardedTensor):
                # Assign shard to be loaded or not
                if is_main_replica(x.replica_id):
                    to_load_shards[_sharded_tensor_shard_id(x)] = x
                else:
                    unloaded_shards[_sharded_tensor_shard_id(x)] = x
            return x

        dict_list_map_inplace(wrap_non_main_replicas, sharded_tensors)
        return sharded_tensors, sharded_state_dict, to_load_shards, unloaded_shards

    def apply_loading_parallelization(
        self, sharded_state_dict: ShardedStateDict
    ) -> Optional[SaveLoadDistribution]:
        """ Distributes the load across ranks by exchanging metadata.

        Exchanges metadata from the state dict and computes the uniform
        (as close as possible) distribution of loads among the ranks.
        Marks ShardedTensors to be loaded by the current rank with replica_id 0
        (and others with non 0 values).

        If `self.do_cache_distribution` is True, caches the distribution between
        the calls and subsequent distributions happen without any inter-rank
        communication.

        Args:
            sharded_state_dict (ShardedStateDict): state dict to distribute the loading

        Returns:
            SaveLoadDistribution (optional): the computed loading distribution
        """
        if self.do_cache_distribution and self.cached_distribution is not None:
            logger.debug(f'Apply *cached* load parallelization')
            precomputed_distribution = self.cached_distribution
        else:
            logger.debug(f'Apply load parallelization')
            precomputed_distribution = determine_main_replica_uniform_distribution(
                sharded_state_dict, self.parallelization_group, True
            )

        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, precomputed_distribution
        )
        if self.do_cache_distribution:
            self.cached_distribution = precomputed_distribution

        return precomputed_distribution

    def exchange_loaded_tensors_gather_object(
        self,
        loaded_tensors: Dict[_ShardId, torch.Tensor],
        unloaded_shards: Dict[_ShardId, ShardedTensor],
        precomputed_distribution: SaveLoadDistribution,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Dict[_ShardId, torch.Tensor]:
        """ Exchange the tensors loaded by different ranks with a simple all_gather_object call.

        This version can be used for debugging purposes do to its simplistic
        implementation. Shouldn't be used if performance is important.

        Args:
            loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
                shard ids to tensors already loaded by this rank.
            unloaded_shards (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
                shard ids to ShardedTensors that aren't loaded yet.
            precomputed_distribution (SaveLoadDistribution): uniform load distribution
            parallelization_group (ProcessGroup, optional): process group used for load
                distribution. Tensors will be exchanged within this group

        Returns:
            Dict[_ShardId, torch.Tensor]: dictionary mapping shard ids to tensors
                needed by this rank to load a given state dict. Includes
                previously loaded tensors (from `loaded_tensors` input)

        """
        all_loaded_tensors_list = [None] * torch.distributed.get_world_size(
            group=parallelization_group
        )
        torch.distributed.all_gather_object(
            all_loaded_tensors_list, loaded_tensors, group=parallelization_group
        )
        all_loaded_tensors_list = cast(List[Dict[_ShardId, torch.Tensor]], all_loaded_tensors_list)
        all_loaded_tensors = reduce(lambda x, y: {**x, **y}, all_loaded_tensors_list)

        # Error checks
        if len(all_loaded_tensors) != sum(map(len, all_loaded_tensors_list)):
            err_msg = 'Duplicate shard ids loaded by different ranks'
            if torch.distributed.get_rank() == 0:
                logger.error(
                    f'{err_msg}. Shards ids by rank: {[lt.keys() for lt in all_loaded_tensors_list]}'
                )
            raise CheckpointingException(err_msg)

        return all_loaded_tensors

    @torch.no_grad()
    def exchange_loaded_tensors_gather_rounds(
        self,
        loaded_tensors: Dict[_ShardId, torch.Tensor],
        unloaded_shards: Dict[_ShardId, ShardedTensor],
        precomputed_distribution: SaveLoadDistribution = None,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Dict[_ShardId, torch.Tensor]:
        """ Exchange the tensors loaded by different ranks with several all_gather calls.

        Groups tensors by dtype, divide tensors that will be exchanged into rounds
        and execute all_gather for tensors from each round.

        Note: the loading is distributed across ranks based on total loaded size
        in bytes, so there is no guarantee that number of rounds needed for each
        rank will be similar, which might result in a lot of almost empty
        all_gathers. The solution would be to group all tensors into a one
        bytes tensor and do a single all_gather (with similarly sized messages).

        Args:
            loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
                shard ids to tensors already loaded by this rank.
            unloaded_shards (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
                shard ids to ShardedTensors that aren't loaded yet.
            precomputed_distribution (SaveLoadDistribution): uniform load distribution
            parallelization_group (ProcessGroup, optional): process group used for load
                distribution. Tensors will be exchanged within this group

        Returns:
            Dict[_ShardId, torch.Tensor]: dictionary mapping shard ids to tensors
                needed by this rank to load a given state dict. Includes
                previously loaded tensors (from `loaded_tensors` input)
        """
        shard_to_saving_rank, _, shard_to_metadata = precomputed_distribution
        local_rank = torch.distributed.get_rank(group=self.parallelization_group)

        all_loaded_tensors = dict(loaded_tensors)

        # Group by dtype so that we all_gather tensors of the same dtype
        for dtype in sorted(
            set(map(lambda sh_ten: sh_ten.dtype, shard_to_metadata.values())), key=str
        ):

            start = time()
            # shards_by_rank maps rank to tensors loaded by this rank
            shards_by_rank: List[List[torch.Tensor]] = [
                [] for _ in range(torch.distributed.get_world_size(group=parallelization_group))
            ]
            for shard_id, rank in shard_to_saving_rank.items():
                if shard_to_metadata[shard_id].dtype == dtype:
                    shards_by_rank[rank].append(shard_id)

            # Transpose `shards_by_rank` to form exchange rounds
            shards_by_round = zip_longest(*shards_by_rank, fillvalue=None)
            for round_idx, round_shard_ids in enumerate(shards_by_round):
                round_tensors = []
                for rank, shard_id in enumerate(round_shard_ids):
                    if shard_id is None:
                        # if no more useful data, the given rank will exchange empty tensor
                        local_ten = torch.empty(0, dtype=dtype, device='cuda')
                    else:
                        assert isinstance(shard_id, tuple), type(shard_id)
                        if rank == local_rank:
                            assert shard_id in all_loaded_tensors, (
                                shard_id,
                                all_loaded_tensors.keys(),
                            )
                            all_loaded_tensors[shard_id] = all_loaded_tensors[shard_id].cuda()
                            local_ten = all_loaded_tensors[shard_id]
                        else:
                            local_ten = self._get_empty_tensor_for_exchange(
                                shard_id, shard_to_metadata, unloaded_shards, all_loaded_tensors
                            )
                    round_tensors.append(local_ten)

                torch.distributed.all_gather(
                    list(round_tensors),
                    round_tensors[local_rank],
                    group=self.parallelization_group,
                    async_op=True,
                )

                del round_tensors  # remove tensor references

            end = time()
            if torch.distributed.get_rank() == 0:
                logger.debug(f'{dtype} exchange rounds all_gather schedule took {end - start}s')

        return all_loaded_tensors

    @torch.no_grad()
    def exchange_loaded_tensors_broadcast(
        self,
        loaded_tensors: Dict[_ShardId, torch.Tensor],
        unloaded_shards: Dict[_ShardId, ShardedTensor],
        precomputed_distribution: SaveLoadDistribution = None,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Dict[_ShardId, torch.Tensor]:
        """ Exchange the tensors loaded by different ranks by a series of broadcasts.

        For each rank for each loaded tensor do a broadcast to the whole group.
        A reasonable tradeoff in terms of performance and simplicity.

        Args:
            loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
                shard ids to tensors already loaded by this rank.
            unloaded_shards (Dict[_ShardId, torch.Tensor]): mapping from ShardedTensor
                shard ids to ShardedTensors that aren't loaded yet.
            precomputed_distribution (SaveLoadDistribution): uniform load distribution
            parallelization_group (ProcessGroup, optional): process group used for load
                distribution. Tensors will be exchanged within this group

        Returns:
            Dict[_ShardId, torch.Tensor]: dictionary mapping shard ids to tensors
                needed by this rank to load a given state dict. Includes
                previously loaded tensors (from `loaded_tensors` input)
        """
        shard_to_saving_rank, _, shard_to_metadata = precomputed_distribution
        local_rank = torch.distributed.get_rank(group=self.parallelization_group)

        all_loaded_tensors = dict(loaded_tensors)

        start = time()
        for shard_id, rank in shard_to_saving_rank.items():
            if rank == local_rank:
                assert shard_id in all_loaded_tensors, (shard_id, all_loaded_tensors.keys())
                all_loaded_tensors[shard_id] = all_loaded_tensors[shard_id].cuda()
                local_ten = all_loaded_tensors[shard_id]
            else:
                local_ten = self._get_empty_tensor_for_exchange(
                    shard_id, shard_to_metadata, unloaded_shards, all_loaded_tensors
                )

            global_src_rank = torch.distributed.get_global_rank(parallelization_group, rank)
            torch.distributed.broadcast(
                local_ten, src=global_src_rank, group=parallelization_group, async_op=True
            )

        end = time()
        if torch.distributed.get_rank() == 0:
            logger.debug(f'exchange broadcast schedule took {end - start}s')

        return all_loaded_tensors

    def _get_empty_tensor_for_exchange(
        self,
        shard_id: _ShardId,
        needed_shards: Dict[_ShardId, ShardedTensor],
        unneeded_shards: Dict[_ShardId, ShardedTensor],
        loaded_tensors: Dict[_ShardId, torch.Tensor],
    ) -> torch.Tensor:
        """ Determines the empty tensor to use for exchange.

        If shard_id is needed by this rank, it will be in the `unloaded_shards`.
        Otherwise, the metadata for this tensor can be found in `shard_to_metadata`

        Args:
            shard_id (_ShardId): shard_id that will be exchanged
            needed_shards (Dict[_ShardId, ShardedTensor]): mapping from shard ids
                to metadata for shards needed by this rank
            unneeded_shards (Dict[_ShardId, ShardedTensor]): mapping from shard ids
                to metadata for shards that can be discarded after exchange
            loaded_tensors (Dict[_ShardId, torch.Tensor]): mapping where useful tensors
                are placed in

        Returns:
            torch.Tensor: empty tensor to be exchanged
        """
        local_unloaded_sh_ten = needed_shards.get(shard_id)
        if local_unloaded_sh_ten is None:
            sh_ten = unneeded_shards[shard_id]
            sh_ten.init_data('cuda')
            tensor = sh_ten.data
            sh_ten.data = None  # won't be used. free memory
        else:
            local_unloaded_sh_ten.init_data('cuda')
            tensor = local_unloaded_sh_ten.data
            loaded_tensors[shard_id] = tensor
        return tensor

    def fill_in_deferred_sharded_tensors(
        self, sharded_state_dict: ShardedStateDict, loaded_tensors: Dict[_ShardId, torch.Tensor]
    ) -> None:
        """ Fill in tensors not loaded by current rank with tensors from `loaded_tensors` map.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to fill in.
                ShardedTensors are completely replaced with corresponding torch.Tensors.
            loaded_tensors (Dict[_ShardId, torch.Tensor]): dict allowing to map
                ShardedTensor from the sharded_state_dict to loaded tensors.

        Returns:

        """

        def fill_in_sharded_tensor(x):
            if isinstance(x, ShardedTensor):
                try:
                    x = loaded_tensors[_sharded_tensor_shard_id(x)]
                except KeyError as e:
                    raise CheckpointingException(
                        f'Missing loaded tensor shard: {_sharded_tensor_shard_id(x)}'
                    ) from e

            return x

        dict_list_map_inplace(fill_in_sharded_tensor, sharded_state_dict)

    @property
    def can_handle_sharded_objects(self):
        return self.base_strategy.can_handle_sharded_objects

    def load_tensors_metadata(self, checkpoint_dir: Path):
        self.base_strategy.load_tensors_metadata(checkpoint_dir)

    def check_backend_compatibility(self, loaded_version):
        self.base_strategy.check_backend_compatibility(loaded_version)

    def check_version_compatibility(self, loaded_version):
        self.base_strategy.check_version_compatibility(loaded_version)


def _sharded_tensor_shard_id(sharded_tensor: ShardedTensor) -> _ShardId:
    """ Unique id of the sharded tensor data.

    Should yield the same value for same data replicated on different ranks.

    Args:
        sharded_tensor (ShardedTensor): sharded tensor representing the data shard

    Returns (tuple): unique id of a data shard
    """
    f_range = sharded_tensor.flattened_range                                   # trace_info : t_57877, t_57899, t_57921, t_57943, t_57965, ...
    return (                                                                   # trace_info : t_57881, t_57903, t_57925, t_57947, t_57969, ...
        sharded_tensor.key,                                                    # trace_info : t_57878, t_57900, t_57922, t_57944, t_57966, ...
        sharded_tensor.global_offset,                                          # trace_info : t_57879, t_57901, t_57923, t_57945, t_57967, ...
        None if f_range is None else (f_range.start, f_range.stop),            # trace_info : t_57880, t_57902, t_57924, t_57946, t_57968, ...
    )


def _shard_size(sh_ten: ShardedTensor):
    """ Returns size in bytes of a given sharded tensor. """
    if sh_ten.flattened_range is None:                                         # trace_info : t_57885, t_57907, t_57929, t_57951, t_57973, ...
        numel = np.product(sh_ten.local_shape)                                 # trace_info : t_57886, t_57908, t_57930, t_57952, t_57974, ...
    else:
        numel = sh_ten.flattened_range.stop - sh_ten.flattened_range.start
    return numel * torch._utils._element_size(sh_ten.dtype)                    # trace_info : t_57887, t_57909, t_57931, t_57953, t_57975, ...


def determine_main_replica_uniform_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    is_loading: bool = False,
) -> Optional[SaveLoadDistribution]:
    """ Computes the save distribution.

    Should be used in conjunction with `distribute_main_replicas_with_precomputed_distribution`
    which applies the computed save distribution.

    We rely on the fact that the assignment algorithm is deterministic on all ranks,
    so there is no extra communication needed after metadata exchange.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to compute the distribution of
        parallelization_group (ProcessGroup): distribution will be computed
            within this process group
        is_loading (bool, optional): whether the distribution is for loading or saving.
            For loading, even non-main replicas must be loaded by this parallelization
            group. Defaults to False.

    Returns (SaveLoadDistribution, optional): distribution that can be used to apply the
        parallelization. Returns None if the process_group is trivial (1 rank)

    """
    group_size = torch.distributed.get_world_size(group=parallelization_group) # trace_info : t_55267, t_122634
    if group_size <= 1:                                                        # trace_info : t_55268, t_122635
        return
    local_shards = list(                                                       # trace_info : t_55269, t_55271, t_55272, t_55283, t_55285, ...
        sh_base                                                                # trace_info : t_55284, t_55293, t_55302, t_55311, t_55320, ...
        for sh_base in nested_values(sharded_state_dict)                       # trace_info : t_55270, t_55281, t_55290, t_55299, t_55308, ...
        if isinstance(sh_base, ShardedTensor)                                  # trace_info : t_55282, t_55291, t_55300, t_55309, t_55318, ...
    )
    local_shards_no_data = [ten.without_data() for ten in local_shards]        # trace_info : t_56521, t_123888

    all_shards = [None] * torch.distributed.get_world_size(group=parallelization_group)# trace_info : t_57866, t_125233
    torch.distributed.all_gather_object(                                       # trace_info : t_57867, t_57869, t_125234, t_125236
        all_shards, local_shards_no_data, group=parallelization_group          # trace_info : t_57868, t_125235
    )

    shard_to_ranks = defaultdict(list)                                         # trace_info : t_57870, t_125237
    shard_to_size = {}                                                         # trace_info : t_57871, t_125238
    shard_to_metadata = {}                                                     # trace_info : t_57872, t_125239
    shards_saved_by_this_parallelization_group: Set[_ShardId] = set()          # trace_info : t_57873, t_125240
    for rank, rank_shards in enumerate(all_shards):                            # trace_info : t_57874, t_60340, t_62022, t_63704, t_65386, ...
        for sh_ten in rank_shards:                                             # trace_info : t_57875, t_57897, t_57919, t_57941, t_57963, ...
            shard_id = _sharded_tensor_shard_id(sh_ten)                        # trace_info : t_57876, t_57898, t_57920, t_57942, t_57964, ...
            shard_to_ranks[shard_id].append(rank)                              # trace_info : t_57882, t_57904, t_57926, t_57948, t_57970, ...
            if shard_id not in shard_to_size:                                  # trace_info : t_57883, t_57905, t_57927, t_57949, t_57971, ...
                shard_to_size[shard_id] = _shard_size(sh_ten)                  # trace_info : t_57884, t_57906, t_57928, t_57950, t_57972, ...
                shard_to_metadata[shard_id] = sh_ten                           # trace_info : t_57888, t_57910, t_57932, t_57954, t_57976, ...
            if is_main_replica(sh_ten.replica_id) or is_loading:               # trace_info : t_57889, t_57911, t_57933, t_57955, t_57977, ...
                shards_saved_by_this_parallelization_group.add(shard_id)       # trace_info : t_57896, t_57918, t_57940, t_57962, t_57984, ...

    shard_to_ranks = {                                                         # trace_info : t_65387, t_65389, t_132754, t_132756
        k: v for k, v in shard_to_ranks.items() if k in shards_saved_by_this_parallelization_group# trace_info : t_65388, t_132755
    }

    shard_to_saving_rank = distribute_shards_to_ranks(                         # trace_info : t_65390, t_65392, t_132757, t_132759
        shard_to_ranks, shard_to_size, len(all_shards)                         # trace_info : t_65391, t_132758
    )

    return SaveLoadDistribution(                                               # trace_info : t_66858, t_66860, t_134225, t_134227
        shard_to_saving_rank, shards_saved_by_this_parallelization_group, shard_to_metadata# trace_info : t_66859, t_134226
    )


def distribute_main_replicas_with_precomputed_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    precomputed_distribution: Optional[SaveLoadDistribution],
):
    """ Applies the save distribution computed with `determine_main_replica_uniform_distribution`.

    Based on rank assignment, sets replica ids of the shards saved by current rank to 0
    and all the other replica ids to 1.

    Args:
        sharded_state_dict (ShardedStateDict): state dict to apply the save distribution to
        parallelization_group (ProcessGroup): distribution will be applied within this
            process group. Must match with the process group passed to
            `determine_main_replica_uniform_distribution`.
        precomputed_distribution (SaveLoadDistribution): distribution computed with
            `determine_main_replica_uniform_distribution`

    Returns: None

    Example replica ids of tensors A, B, C before distribution:
    rank0: A: (0, 0, 0), B: (0, 0, 0), C: (0, 0, 0)
    rank1: A: (0, 0, 1), B: (0, 0, 1), C: (0, 0, 1)
    rank2: A: (0, 0, 2), B: (0, 0, 2), C: (0, 0, 2)

    Replicas after distribution for the example above:
    rank0: A: 0, B: 1, C: 1
    rank1: A: 1, B: 0, C: 1
    rank2: A: 1, B: 1, C: 0
    """
    if torch.distributed.get_world_size(group=parallelization_group) <= 1:     # trace_info : t_66865, t_134232
        return
    if precomputed_distribution is None:                                       # trace_info : t_66866, t_134233
        raise ValueError(
            'precomputed_distribution must be not None for non-trivial parallelization group'
        )

    local_shards = list(                                                       # trace_info : t_66867, t_66869, t_66870, t_66881, t_66883, ...
        sh_base                                                                # trace_info : t_66882, t_66891, t_66900, t_66909, t_66918, ...
        for sh_base in nested_values(sharded_state_dict)                       # trace_info : t_66868, t_66879, t_66888, t_66897, t_66906, ...
        if isinstance(sh_base, ShardedTensor)                                  # trace_info : t_66880, t_66889, t_66898, t_66907, t_66916, ...
    )

    rank_within_dp_group = torch.distributed.get_rank(parallelization_group)   # trace_info : t_68119, t_135486
    for sh_ten in local_shards:                                                # trace_info : t_68120, t_68130, t_68140, t_68150, t_68160, ...
        shard_id = _sharded_tensor_shard_id(sh_ten)                            # trace_info : t_68121, t_68131, t_68141, t_68151, t_68161, ...
        if (
            shard_id in precomputed_distribution.shards_in_this_group          # trace_info : t_68127, t_68137, t_68147, t_68157, t_68167, ...
            and rank_within_dp_group == precomputed_distribution.main_rank_for_shard[shard_id]# trace_info : t_68128, t_68138, t_68148, t_68158, t_68168, ...
        ):
            sh_ten.replica_id = 0                                              # trace_info : t_68249, t_68409, t_68469, t_68499, t_68529, ...
        else:
            sh_ten.replica_id = 1                                              # trace_info : t_68129, t_68139, t_68149, t_68159, t_68169, ...


T = TypeVar('T')


def distribute_shards_to_ranks(
    shard_to_ranks: Dict[T, List[int]], shard_to_size: Dict[T, int], num_ranks: int
) -> Dict[T, int]:
    """ Computes uniform distribution of workload across ranks, based on sizes.

    Currently, the assignment is greedy, based on:
    1. Firstly, the coverage of each shard
        (how many ranks the shard is available on; lower coverage is assigned first)
    2. Secondly, the size of each shard (larger size is assigned first)
    3. Finally, shard id for differentiation.

    Third step is added because we rely on the fact that the assignment is deterministic on all ranks.

    Args:
        shard_to_ranks (Dict[T, List[int]]): mapping which tells which rank have access to which shards
        shard_to_size (Dict[T, int]): sizes of each shard
        num_ranks (int): number of ranks in the parallelization group

    Returns (Dict[T, int]): assignment of shard to rank (which rank should do the work
        to achieve maximal uniformity)
    """
    shard_to_ranks = {k: tuple(v) for k, v in shard_to_ranks.items()}          # trace_info : t_65393, t_132760
    shard_to_saving_rank = {}                                                  # trace_info : t_65394, t_132761
    rank_sizes = [(0, rank) for rank in range(num_ranks)]                      # trace_info : t_65395, t_132762

    # start from tensors with lowest coverage, then go by tensor size from largest (hence minus size)
    for shard_id, shard_ranks in sorted(                                       # trace_info : t_65396, t_65399, t_65856, t_65865, t_65874, ...
        shard_to_ranks.items(),                                                # trace_info : t_65397, t_132764
        key=lambda sh_id_ranks: (                                              # trace_info : t_65398, t_65403, t_65407, t_65411, t_65415, ...
            len(sh_id_ranks[1]),                                               # trace_info : t_65400, t_65404, t_65408, t_65412, t_65416, ...
            -shard_to_size[sh_id_ranks[0]],                                    # trace_info : t_65401, t_65405, t_65409, t_65413, t_65417, ...
            sh_id_ranks[0],                                                    # trace_info : t_65402, t_65406, t_65410, t_65414, t_65418, ...
        ),
    ):
        # assign greedily to the least occupied rank
        size, rank = min((size, rank) for size, rank in rank_sizes if rank in shard_ranks)# trace_info : t_65848, t_65849, t_65850, t_65851, t_65852, ...

        shard_to_saving_rank[shard_id] = rank                                  # trace_info : t_65854, t_65863, t_65872, t_65881, t_65890, ...
        rank_sizes[rank] = (size + shard_to_size[shard_id], rank)              # trace_info : t_65855, t_65864, t_65873, t_65882, t_65891, ...

    logger.debug(f'distribute_shards_to_ranks distribution: {rank_sizes}')     # trace_info : t_66856, t_134223

    return shard_to_saving_rank                                                # trace_info : t_66857, t_134224
