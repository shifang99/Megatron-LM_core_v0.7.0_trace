# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split

LowLevelDataset = Union[IndexedDataset, Iterable]


class MegatronDataset(ABC, torch.utils.data.Dataset):
    """The highest level wrapper class from which all dataset classes should inherit

    Args:
        dataset (LowLevelDataset): The dataset around which to build the MegatronDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The minimum number of samples to build from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indices Split

        config (BlendedMegatronDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        self.dataset = dataset                                                 # trace_info : t_18277, t_18492, t_18707
        self.dataset_path = dataset_path                                       # trace_info : t_18278, t_18493, t_18708
        self.indices = indices                                                 # trace_info : t_18279, t_18494, t_18709
        self.num_samples = num_samples                                         # trace_info : t_18280, t_18495, t_18710
        self.index_split = index_split                                         # trace_info : t_18281, t_18496, t_18711
        self.config = config                                                   # trace_info : t_18282, t_18497, t_18712

        self.unique_identifiers = OrderedDict()                                # trace_info : t_18283, t_18498, t_18713

        self.unique_identifiers["class"] = type(self).__name__                 # trace_info : t_18284, t_18499, t_18714
        self.unique_identifiers["dataset_path"] = self.dataset_path            # trace_info : t_18285, t_18500, t_18715
        self.unique_identifiers["num_samples"] = self.num_samples              # trace_info : t_18286, t_18501, t_18716
        self.unique_identifiers["index_split"] = self.index_split.name         # trace_info : t_18287, t_18502, t_18717
        for attr in self._key_config_attributes():                             # trace_info : t_18288, t_18291, t_18293, t_18295, t_18297, ...
            self.unique_identifiers[attr] = getattr(self.config, attr)         # trace_info : t_18290, t_18292, t_18294, t_18296, t_18298, ...

        self.unique_description = json.dumps(                                  # trace_info : t_18300, t_18302, t_18515, t_18517, t_18730, ...
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers# trace_info : t_18301, t_18303, t_18516, t_18518, t_18731, ...
        )
        self.unique_description_hash = hashlib.md5(                            # trace_info : t_18304, t_18306, t_18308, t_18519, t_18521, ...
            self.unique_description.encode("utf-8")                            # trace_info : t_18305, t_18520, t_18735
        ).hexdigest()                                                          # trace_info : t_18307, t_18522, t_18737

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        """Return the number of elements in the underlying low level dataset for the purpose of
        segregating the train/valid/test split indices

        It may be that the low level dataset can be split any number of ways, depending on the mid
        level dataset it supports, which is why we define the "number of elements" function
        separately from the __len__ function here in the mid level dataset class

        Args:
            low_level_dataset (LowLevelDataset): The underlying low level dataset

        Returns:
            int: The number of elements in the underlying low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: BlendedMegatronDatasetConfig
    ) -> LowLevelDataset:
        """Build the low level dataset via a function to be called from within
        BlendedMegatronDatasetBuilder.build_generic_dataset

        It may be that the low level dataset spans any subset of train/valid/test splits, which is
        why we define a static "build" function separately from the constructor in the mid level
        dataset class

        Args:
            dataset_path (str): The real path on disk to the dataset

            config (BlendedMegatronDatasetConfig): The dataset config

        Returns:
            LowLevelDataset: The low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Return all config attributes which contribute to uniquely identifying the dataset.

        These attributes will be used to build a uniquely identifying string and MD5 hash which
        will be used to cache/load dataset resources from run to run.

        Returns:
            List[str]: The key config attributes
        """
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]# trace_info : t_18289, t_18504, t_18719

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: See abstract implementation
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, Union[torch.Tensor, numpy.ndarray]]: See abstract implementation
        """
        pass
