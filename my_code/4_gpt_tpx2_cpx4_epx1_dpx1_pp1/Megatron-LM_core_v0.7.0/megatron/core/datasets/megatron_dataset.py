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
        self.dataset = dataset                                                 # trace_info : t_16269, t_16484, t_16699
        self.dataset_path = dataset_path                                       # trace_info : t_16270, t_16485, t_16700
        self.indices = indices                                                 # trace_info : t_16271, t_16486, t_16701
        self.num_samples = num_samples                                         # trace_info : t_16272, t_16487, t_16702
        self.index_split = index_split                                         # trace_info : t_16273, t_16488, t_16703
        self.config = config                                                   # trace_info : t_16274, t_16489, t_16704

        self.unique_identifiers = OrderedDict()                                # trace_info : t_16275, t_16490, t_16705

        self.unique_identifiers["class"] = type(self).__name__                 # trace_info : t_16276, t_16491, t_16706
        self.unique_identifiers["dataset_path"] = self.dataset_path            # trace_info : t_16277, t_16492, t_16707
        self.unique_identifiers["num_samples"] = self.num_samples              # trace_info : t_16278, t_16493, t_16708
        self.unique_identifiers["index_split"] = self.index_split.name         # trace_info : t_16279, t_16494, t_16709
        for attr in self._key_config_attributes():                             # trace_info : t_16280, t_16283, t_16285, t_16287, t_16289, ...
            self.unique_identifiers[attr] = getattr(self.config, attr)         # trace_info : t_16282, t_16284, t_16286, t_16288, t_16290, ...

        self.unique_description = json.dumps(                                  # trace_info : t_16292, t_16294, t_16507, t_16509, t_16722, ...
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers# trace_info : t_16293, t_16295, t_16508, t_16510, t_16723, ...
        )
        self.unique_description_hash = hashlib.md5(                            # trace_info : t_16296, t_16298, t_16300, t_16511, t_16513, ...
            self.unique_description.encode("utf-8")                            # trace_info : t_16297, t_16512, t_16727
        ).hexdigest()                                                          # trace_info : t_16299, t_16514, t_16729

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
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]# trace_info : t_16281, t_16496, t_16711

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
