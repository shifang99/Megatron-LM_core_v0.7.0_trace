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
        self.dataset = dataset                                                 # trace_info : t_13410, t_13625, t_13840
        self.dataset_path = dataset_path                                       # trace_info : t_13411, t_13626, t_13841
        self.indices = indices                                                 # trace_info : t_13412, t_13627, t_13842
        self.num_samples = num_samples                                         # trace_info : t_13413, t_13628, t_13843
        self.index_split = index_split                                         # trace_info : t_13414, t_13629, t_13844
        self.config = config                                                   # trace_info : t_13415, t_13630, t_13845

        self.unique_identifiers = OrderedDict()                                # trace_info : t_13416, t_13631, t_13846

        self.unique_identifiers["class"] = type(self).__name__                 # trace_info : t_13417, t_13632, t_13847
        self.unique_identifiers["dataset_path"] = self.dataset_path            # trace_info : t_13418, t_13633, t_13848
        self.unique_identifiers["num_samples"] = self.num_samples              # trace_info : t_13419, t_13634, t_13849
        self.unique_identifiers["index_split"] = self.index_split.name         # trace_info : t_13420, t_13635, t_13850
        for attr in self._key_config_attributes():                             # trace_info : t_13421, t_13424, t_13426, t_13428, t_13430, ...
            self.unique_identifiers[attr] = getattr(self.config, attr)         # trace_info : t_13423, t_13425, t_13427, t_13429, t_13431, ...

        self.unique_description = json.dumps(                                  # trace_info : t_13433, t_13435, t_13648, t_13650, t_13863, ...
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers# trace_info : t_13434, t_13436, t_13649, t_13651, t_13864, ...
        )
        self.unique_description_hash = hashlib.md5(                            # trace_info : t_13437, t_13439, t_13441, t_13652, t_13654, ...
            self.unique_description.encode("utf-8")                            # trace_info : t_13438, t_13653, t_13868
        ).hexdigest()                                                          # trace_info : t_13440, t_13655, t_13870

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
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]# trace_info : t_13422, t_13637, t_13852

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
