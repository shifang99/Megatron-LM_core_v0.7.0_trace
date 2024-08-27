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
        self.dataset = dataset                                                 # trace_info : t_16439, t_16654, t_16869
        self.dataset_path = dataset_path                                       # trace_info : t_16440, t_16655, t_16870
        self.indices = indices                                                 # trace_info : t_16441, t_16656, t_16871
        self.num_samples = num_samples                                         # trace_info : t_16442, t_16657, t_16872
        self.index_split = index_split                                         # trace_info : t_16443, t_16658, t_16873
        self.config = config                                                   # trace_info : t_16444, t_16659, t_16874

        self.unique_identifiers = OrderedDict()                                # trace_info : t_16445, t_16660, t_16875

        self.unique_identifiers["class"] = type(self).__name__                 # trace_info : t_16446, t_16661, t_16876
        self.unique_identifiers["dataset_path"] = self.dataset_path            # trace_info : t_16447, t_16662, t_16877
        self.unique_identifiers["num_samples"] = self.num_samples              # trace_info : t_16448, t_16663, t_16878
        self.unique_identifiers["index_split"] = self.index_split.name         # trace_info : t_16449, t_16664, t_16879
        for attr in self._key_config_attributes():                             # trace_info : t_16450, t_16453, t_16455, t_16457, t_16459, ...
            self.unique_identifiers[attr] = getattr(self.config, attr)         # trace_info : t_16452, t_16454, t_16456, t_16458, t_16460, ...

        self.unique_description = json.dumps(                                  # trace_info : t_16462, t_16464, t_16677, t_16679, t_16892, ...
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers# trace_info : t_16463, t_16465, t_16678, t_16680, t_16893, ...
        )
        self.unique_description_hash = hashlib.md5(                            # trace_info : t_16466, t_16468, t_16470, t_16681, t_16683, ...
            self.unique_description.encode("utf-8")                            # trace_info : t_16467, t_16682, t_16897
        ).hexdigest()                                                          # trace_info : t_16469, t_16684, t_16899

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
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]# trace_info : t_16451, t_16666, t_16881

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
