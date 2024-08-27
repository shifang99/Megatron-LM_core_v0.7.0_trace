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
        self.dataset = dataset                                                 # trace_info : t_16370, t_16585, t_16800
        self.dataset_path = dataset_path                                       # trace_info : t_16371, t_16586, t_16801
        self.indices = indices                                                 # trace_info : t_16372, t_16587, t_16802
        self.num_samples = num_samples                                         # trace_info : t_16373, t_16588, t_16803
        self.index_split = index_split                                         # trace_info : t_16374, t_16589, t_16804
        self.config = config                                                   # trace_info : t_16375, t_16590, t_16805

        self.unique_identifiers = OrderedDict()                                # trace_info : t_16376, t_16591, t_16806

        self.unique_identifiers["class"] = type(self).__name__                 # trace_info : t_16377, t_16592, t_16807
        self.unique_identifiers["dataset_path"] = self.dataset_path            # trace_info : t_16378, t_16593, t_16808
        self.unique_identifiers["num_samples"] = self.num_samples              # trace_info : t_16379, t_16594, t_16809
        self.unique_identifiers["index_split"] = self.index_split.name         # trace_info : t_16380, t_16595, t_16810
        for attr in self._key_config_attributes():                             # trace_info : t_16381, t_16384, t_16386, t_16388, t_16390, ...
            self.unique_identifiers[attr] = getattr(self.config, attr)         # trace_info : t_16383, t_16385, t_16387, t_16389, t_16391, ...

        self.unique_description = json.dumps(                                  # trace_info : t_16393, t_16395, t_16608, t_16610, t_16823, ...
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers# trace_info : t_16394, t_16396, t_16609, t_16611, t_16824, ...
        )
        self.unique_description_hash = hashlib.md5(                            # trace_info : t_16397, t_16399, t_16401, t_16612, t_16614, ...
            self.unique_description.encode("utf-8")                            # trace_info : t_16398, t_16613, t_16828
        ).hexdigest()                                                          # trace_info : t_16400, t_16615, t_16830

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
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]# trace_info : t_16382, t_16597, t_16812

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
