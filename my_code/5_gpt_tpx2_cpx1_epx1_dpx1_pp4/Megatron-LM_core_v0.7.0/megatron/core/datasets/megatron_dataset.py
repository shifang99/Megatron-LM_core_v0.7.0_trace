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
        self.dataset = dataset                                                 # trace_info : t_16476, t_16691, t_16906
        self.dataset_path = dataset_path                                       # trace_info : t_16477, t_16692, t_16907
        self.indices = indices                                                 # trace_info : t_16478, t_16693, t_16908
        self.num_samples = num_samples                                         # trace_info : t_16479, t_16694, t_16909
        self.index_split = index_split                                         # trace_info : t_16480, t_16695, t_16910
        self.config = config                                                   # trace_info : t_16481, t_16696, t_16911

        self.unique_identifiers = OrderedDict()                                # trace_info : t_16482, t_16697, t_16912

        self.unique_identifiers["class"] = type(self).__name__                 # trace_info : t_16483, t_16698, t_16913
        self.unique_identifiers["dataset_path"] = self.dataset_path            # trace_info : t_16484, t_16699, t_16914
        self.unique_identifiers["num_samples"] = self.num_samples              # trace_info : t_16485, t_16700, t_16915
        self.unique_identifiers["index_split"] = self.index_split.name         # trace_info : t_16486, t_16701, t_16916
        for attr in self._key_config_attributes():                             # trace_info : t_16487, t_16490, t_16492, t_16494, t_16496, ...
            self.unique_identifiers[attr] = getattr(self.config, attr)         # trace_info : t_16489, t_16491, t_16493, t_16495, t_16497, ...

        self.unique_description = json.dumps(                                  # trace_info : t_16499, t_16501, t_16714, t_16716, t_16929, ...
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers# trace_info : t_16500, t_16502, t_16715, t_16717, t_16930, ...
        )
        self.unique_description_hash = hashlib.md5(                            # trace_info : t_16503, t_16505, t_16507, t_16718, t_16720, ...
            self.unique_description.encode("utf-8")                            # trace_info : t_16504, t_16719, t_16934
        ).hexdigest()                                                          # trace_info : t_16506, t_16721, t_16936

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
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]# trace_info : t_16488, t_16703, t_16918

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
