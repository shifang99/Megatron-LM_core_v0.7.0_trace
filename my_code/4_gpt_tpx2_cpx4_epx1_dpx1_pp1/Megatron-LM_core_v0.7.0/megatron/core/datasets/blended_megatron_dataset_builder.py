# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Type, Union

import numpy
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split, log_single_rank, normalize
from megatron.core.parallel_state import get_virtual_pipeline_model_parallel_rank

logger = logging.getLogger(__name__)

MidLevelDataset = MegatronDataset

TopLevelDataset = Union[BlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]


class BlendedMegatronDatasetBuilder(object):
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[Optional[int]]): The minimum total number of samples to draw, or None, per split

        is_built_on_rank (Callable): A callable which returns True if the dataset should be built on the current rank and False otherwise. It should be Megatron Core parallelism aware i.e. global rank, local group rank, and virtual rank may inform its return value.

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        cls: Type[MidLevelDataset],
        sizes: List[int],
        is_built_on_rank: Callable,
        config: BlendedMegatronDatasetConfig,
    ):
        self.cls = cls                                                         # trace_info : t_16008
        self.sizes = sizes                                                     # trace_info : t_16009
        self.is_built_on_rank = is_built_on_rank                               # trace_info : t_16010
        self.config = config                                                   # trace_info : t_16011

        log_single_rank(                                                       # trace_info : t_16012, t_16017
            logger,                                                            # trace_info : t_16013
            logging.INFO,                                                      # trace_info : t_16014
            f"Building dataset splits with cls={cls.__name__}, sizes={self.sizes}, and config={self.config}",# trace_info : t_16015
        )

        if not self.config.mock:                                               # trace_info : t_16025
            for split in Split:                                                # trace_info : t_16026, t_16031, t_16036, t_16041
                size_is_none = self.sizes[split.value] is None                 # trace_info : t_16027, t_16032, t_16037
                if self.config.blend_per_split is None:                        # trace_info : t_16028, t_16033, t_16038
                    weights_are_none = self.config.blend[1] is None
                else:
                    if self.config.blend_per_split[split.value] is None:       # trace_info : t_16029, t_16034, t_16039
                        continue                                               # trace_info : t_16030, t_16035, t_16040
                    weights_are_none = self.config.blend_per_split[split.value][1] is None
                if size_is_none:
                    assert (
                        weights_are_none
                    ), f"size_is_none => weights_are_none fails for {split.name} split"

        if torch.distributed.is_initialized():                                 # trace_info : t_16042
            gb_rank = torch.distributed.get_rank()                             # trace_info : t_16043
            vp_rank = get_virtual_pipeline_model_parallel_rank()               # trace_info : t_16044
            if gb_rank == 0 and (vp_rank == 0 or vp_rank is None):             # trace_info : t_16046
                assert (                                                       # trace_info : t_16065
                    self.is_built_on_rank()                                    # trace_info : t_16047
                ), "is_built_on_rank must return True when global rank = 0 and vp rank = 0"

    def build(self) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)

        This method is distributed-aware and must be called on all ranks.

        The dataset splits returned can vary according to the config. Supply config.blend and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.blend_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions. In either case, for each split, handle the following
        cases:

        (1) The split is None
            - do nothing

        (2) The split has one contributing dataset, and...

            (a) 'size' is not None
                - Build a mid-level dataset with low-level dataset sampling in proportion to the size

            (b) 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling

        (3) The split has multiple contributing datasets, and...

            (a) 'weights' is not None and 'size' is not None
                - Build mid-level datasets with low-level dataset sampling in proportion to their weights and the size
                - Build a top-level dataset of length marginally greater than 'size' with mid-level dataset sampling in proportion to their weights and the size

            (b) 'weights' is not None and 'size' is None
                - Error

            (c) 'weights' is None and 'size' is not None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset of length 'size' with mid-level dataset sampling in proportion to their lengths and the size

                  - The 'size' of the top-level dataset is capped at the sum of the mid-level dataset lengths

            (d) 'weights' is None and 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset with no excess mid-level dataset sampling

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per split
        """
        datasets = self._build_blended_dataset_splits()                        # trace_info : t_16067

        for dataset in datasets:                                               # trace_info : t_16876, t_16880, t_16884, t_16888
            if dataset is not None and len(dataset) > 0:                       # trace_info : t_16877, t_16881, t_16885
                if isinstance(dataset, BlendedDataset):                        # trace_info : t_16879, t_16883, t_16887
                    # Check blend size
                    assert dataset.size is None or dataset.size == dataset.dataset_index.shape[0]
                    # Check blend access of mid-level datasets
                    _, sizes = numpy.unique(dataset.dataset_index, return_counts=True)
                    for i, dataset_and_size in enumerate(zip(dataset.datasets, sizes)):
                        if len(dataset_and_size[0]) < dataset_and_size[1]:
                            raise IndexError(
                                f"{type(dataset).__name__} blend goes out of bounds for {type([dataset_and_size[0]]).__name__} {i} for {dataset.split.name} split"
                            )

        return datasets                                                        # trace_info : t_16889

    def _build_blended_dataset_splits(self,) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)

        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per split
        """
        ##
        # Return fake "mock" datasets
        ##
        if self.config.mock:                                                   # trace_info : t_16068
            split = self.config.split_matrix
            try:
                return self._build_megatron_dataset_splits(None, split, self.sizes)
            except Exception as error:
                raise Exception(
                    f"{self.cls.__name__} failed to build as a mock data generator"
                ) from error

        ##
        # All splits come from the same distribution
        ##
        elif self.config.blend:                                                # trace_info : t_16069
            prefixes, weights = self.config.blend                              # trace_info : t_16070
            if weights is not None:                                            # trace_info : t_16071
                weights = normalize(weights)

            split = self.config.split_matrix                                   # trace_info : t_16072

            # Blend consists of a single prefix
            if len(prefixes) == 1 and weights is None:                         # trace_info : t_16073
                return self._build_megatron_dataset_splits(prefixes[0], split, self.sizes)# trace_info : t_16074

            # Build the mid-level datasets
            if weights is None:
                sizes_per_dataset = [[None for split in Split] for prefix in prefixes]
            else:
                sizes_per_dataset = _get_size_per_split_per_dataset(weights, self.sizes)

            # build each dataset in parallel
            megatron_datasets = self._build_megatron_datasets_parallel(
                prefixes, split, sizes_per_dataset
            )

            # Build the top-level datasets
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                if split[i] is not None:
                    weights_i = weights
                    if weights_i is not None and self.sizes[i] is not None:
                        size_i = sum(list(zip(*sizes_per_dataset))[i])
                    elif weights_i is None:
                        try:
                            weights_i = [
                                len(megatron_dataset) for megatron_dataset in megatron_datasets[i]
                            ]
                        except TypeError:
                            weights_i = [0 for _ in prefixes]
                        if self.sizes[i] is not None:
                            size_i = min(self.sizes[i], sum(weights_i))
                        else:
                            size_i = None  # => the size will be sum(weights_i)
                    else:
                        raise RuntimeError
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,
                        self.is_built_on_rank,
                        True,  # synchronize_ranks, default behavior to build on rank-0 first
                        megatron_datasets[i],
                        weights_i,
                        size_i,
                        self.config,
                    )

            return blended_datasets

        ##
        # Each split comes from a separate distribution
        ##
        else:
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                split_spoof = [None] * len(Split)
                split_spoof[i] = (0.0, 1.0)
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend is provided for the split
                blend = self.config.blend_per_split[i]
                if blend is not None:
                    prefixes, weights = blend
                    if weights is not None:
                        weights = normalize(weights)

                    # Blend consists of a sigle prefix
                    if len(prefixes) == 1:
                        blended_datasets[i] = self._build_megatron_dataset_splits(
                            prefixes[0], split_spoof, sizes_spoof
                        )[i]
                        continue

                    # Build mid-level datasets
                    if weights is None:
                        sizes_per_dataset = [[None for split in Split] for prefix in prefixes]
                    else:
                        sizes_per_dataset = _get_size_per_split_per_dataset(weights, sizes_spoof)

                    # build each dataset in parallel
                    megatron_datasets = self._build_megatron_datasets_parallel(
                        prefixes, split_spoof, sizes_per_dataset
                    )[i]

                    # Build top-level dataset
                    if weights is not None and self.sizes[i] is not None:
                        size = list(map(sum, zip(*sizes_per_dataset)))[i]
                    elif weights is None:
                        try:
                            weights = [
                                len(megatron_dataset) for megatron_dataset in megatron_datasets
                            ]
                        except TypeError:
                            weights = [0 for _ in prefixes]
                        if self.sizes[i] is not None:
                            size = min(self.sizes[i], sum(weights))
                        else:
                            size = None  # => the size will be sum(weights)
                    else:
                        raise RuntimeError
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,
                        self.is_built_on_rank,
                        True,  # synchronize_ranks, default behavior to build on rank-0 first
                        megatron_datasets,
                        weights,
                        size,
                        self.config,
                    )

            return blended_datasets

    def _build_megatron_datasets_parallel(
        self, prefixes: List[str], split: List[float], sizes_per_dataset: List[List[int]],
    ) -> List[List[Optional[MegatronDataset]]]:
        """Build the megatron datasets for a list of prefixes in parallel

        Args:
            prefixes (List[str]): The list of prefix strings

            split (List[float]): The dataset split ratios (must sum to 1.00)

            sizes_per_dataset (List[List[int]]): The number of samples to request
            per MegatronDataset per spilt

        Returns:
            List[List[Optional[MegatronDataset]]]: For each split, have a list of
            MegatronDataset per prefix
        """
        # Helper function to wrap the threading logic
        def _threading_helper(
            megatron_datasets: List[List[Optional[MegatronDataset]]],
            num_workers: int,
            prefixes: List[str],
            split: List[float],
            sizes_per_dataset: List[List[int]],
        ) -> None:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                all_futures = []
                for i in range(len(prefixes)):
                    all_futures.append(
                        executor.submit(
                            self._build_megatron_dataset_splits,
                            prefixes[i],
                            split,
                            sizes_per_dataset[i],
                            False,  # synchronize_ranks, barrier is called in this function
                        )
                    )
                for future in all_futures:
                    try:
                        megatron_datasets_split = future.result()
                        for j in range(len(megatron_datasets_split)):
                            megatron_datasets[j].append(megatron_datasets_split[j])
                    except Exception as err:
                        raise err
            return megatron_datasets

        megatron_datasets = [[] for _ in range(len(Split))]
        num_dataset_builder_threads = self.config.num_dataset_builder_threads

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            # First, build on rank 0
            if rank == 0:
                num_workers = num_dataset_builder_threads
                if num_workers > 1:
                    # since only rank 0 is running, scale up the thread count
                    # but not too much to avoid overloading storage on miss path.
                    # if user set num_dataset_builder_threads to 1,
                    # i.e. meant for serial build, do not scale up.
                    num_workers *= min(2, max(1, torch.cuda.device_count()))
                _threading_helper(
                    megatron_datasets, num_workers, prefixes, split, sizes_per_dataset,
                )

            torch.distributed.barrier()

            # Then, build on other ranks; guaranteed to be data_cache hit
            if rank != 0:
                _threading_helper(
                    megatron_datasets,
                    num_dataset_builder_threads,
                    prefixes,
                    split,
                    sizes_per_dataset,
                )
        else:
            _threading_helper(
                megatron_datasets, num_dataset_builder_threads, prefixes, split, sizes_per_dataset,
            )

        return megatron_datasets

    def _build_megatron_dataset_splits(
        self,
        dataset_path: Optional[str],
        split: List[float],
        sizes: List[int],
        synchronize_ranks: bool = True,
    ) -> List[Optional[MidLevelDataset]]:
        """Build each MidLevelDataset split from a single LowLevelDataset

        Args:
            dataset_path (Optional[str]): The path on disk which defines the underlying LowLevelDataset, or None for mock dataset classes

            split (List[Tuple[float, float]]): The dataset split matrix

            sizes (List[int]): The number of total samples to draw from each split

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks behavior. Set to False when we enforce this behavior at higher level.

        Returns:
            List[Optional[MidLevelDataset]]: The MidLevelDataset (or None) per split
        """
        # Build the low level dataset
        low_level_dataset = self.cls.build_low_level_dataset(dataset_path, self.config)# trace_info : t_16075

        # Build the split indices for the low level dataset
        num_elements = self.cls.numel_low_level_dataset(low_level_dataset)     # trace_info : t_16208
        split_indices = []                                                     # trace_info : t_16211
        for i, _ in enumerate(Split):                                          # trace_info : t_16212, t_16217, t_16222, t_16227
            if split[i] is not None:                                           # trace_info : t_16213, t_16218, t_16223
                beg = int(round(split[i][0] * float(num_elements)))            # trace_info : t_16214, t_16219, t_16224
                end = int(round(split[i][1] * float(num_elements)))            # trace_info : t_16215, t_16220, t_16225
                split_indices.append(numpy.arange(start=beg, stop=end, step=1, dtype=numpy.int32))# trace_info : t_16216, t_16221, t_16226
            else:
                split_indices.append(None)

        # Build the mid level dataset
        mid_level_datasets = []                                                # trace_info : t_16228
        for i, _split in enumerate(Split):                                     # trace_info : t_16229, t_16444, t_16659, t_16874
            if split[i] is None:                                               # trace_info : t_16230, t_16445, t_16660
                mid_level_datasets.append(None)
            else:
                mid_level_datasets.append(                                     # trace_info : t_16231, t_16443, t_16446, t_16658, t_16661, ...
                    self.build_generic_dataset(                                # trace_info : t_16232, t_16242, t_16447, t_16457, t_16662, ...
                        self.cls,                                              # trace_info : t_16233, t_16448, t_16663
                        self.is_built_on_rank,                                 # trace_info : t_16234, t_16449, t_16664
                        synchronize_ranks,                                     # trace_info : t_16235, t_16450, t_16665
                        low_level_dataset,                                     # trace_info : t_16236, t_16451, t_16666
                        dataset_path,                                          # trace_info : t_16237, t_16452, t_16667
                        split_indices[i],                                      # trace_info : t_16238, t_16453, t_16668
                        sizes[i],                                              # trace_info : t_16239, t_16454, t_16669
                        _split,                                                # trace_info : t_16240, t_16455, t_16670
                        self.config,                                           # trace_info : t_16241, t_16456, t_16671
                    )
                )

        return mid_level_datasets                                              # trace_info : t_16875

    @staticmethod
    def build_generic_dataset(
        cls: Union[Type[DistributedDataset], Callable],
        is_built_on_rank: Callable,
        synchronize_ranks: bool,
        *args: Any,
    ) -> Optional[Union[DistributedDataset, Iterable]]:
        """Build the DistributedDataset

        Return None if and only if the underlying dataset class is not built on the current rank
        and torch.distributed is initialized.

        Args:
            cls (Union[Type[DistributedDataset], Callable]): The DistributedDataset class to be built. In special cases, e.g. when we are building the low level dataset for a RawMegatronDataset instance, we can accept a Callable which returns an Iterable.

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks behavior. Set to False when we enforce this behavior at higher level.

            args (Tuple[Any]): The positional arguments used to build the provided DistributedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            Optional[Union[DistributedDataset, Iterable]]: The DistributedDataset instantion, the Iterable instantiation, or None
        """
        if torch.distributed.is_initialized():                                 # trace_info : t_16243, t_16458, t_16673
            rank = torch.distributed.get_rank()                                # trace_info : t_16244, t_16459, t_16674

            dataset = None                                                     # trace_info : t_16245, t_16460, t_16675

            # First, build on rank 0
            if rank == 0 and is_built_on_rank():                               # trace_info : t_16246, t_16461, t_16676
                try:                                                           # trace_info : t_16264, t_16479, t_16694
                    dataset = cls(*args)                                       # trace_info : t_16265, t_16480, t_16695
                except OSError as err:
                    log = (
                        f"Failed to write dataset materials to the data cache directory. "
                        + f"Please supply a directory to which you have write access via "
                        + f"the path_to_cache attribute in BlendedMegatronDatasetConfig and "
                        + f"retry. Refer to the preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            if synchronize_ranks:                                              # trace_info : t_16439, t_16654, t_16869
                torch.distributed.barrier()                                    # trace_info : t_16440, t_16655, t_16870

            # After, build on other ranks
            if rank != 0 and is_built_on_rank():                               # trace_info : t_16441, t_16656, t_16871
                dataset = cls(*args)

            return dataset                                                     # trace_info : t_16442, t_16657, t_16872

        return cls(*args)


def _get_size_per_split_per_dataset(
    normalized_weights: List[float], target_size_per_split: List[int]
) -> List[List[int]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits

    Args:
        normalized_weights (List[float]): e.g. [0.3, 0.7]

        target_size_per_split (List[int]): The number of samples to target for each BlendedDataset split

    Returns:
        List[List[int]]: The number of samples to request per MegatronDataset per split
    """
    assert numpy.isclose(sum(normalized_weights), 1.0)

    # Use 0.5% target margin to ensure we satiate the request
    sizes_per_dataset = [
        [int(math.ceil(target_size * weight * 1.005)) for target_size in target_size_per_split]
        for weight in normalized_weights
    ]

    return sizes_per_dataset
