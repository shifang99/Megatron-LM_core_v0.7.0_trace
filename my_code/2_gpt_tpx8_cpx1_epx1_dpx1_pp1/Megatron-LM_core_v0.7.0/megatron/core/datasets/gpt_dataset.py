# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging                                                                 # trace_info : t_17929, t_18024
import os                                                                      # trace_info : t_17930
import time                                                                    # trace_info : t_17931
from dataclasses import dataclass                                              # trace_info : t_17932
from typing import Dict, Optional, Tuple                                       # trace_info : t_17933
                                                                               # trace_info : t_17934
import numpy                                                                   # trace_info : t_17935
import torch                                                                   # trace_info : t_17936
                                                                               # trace_info : t_17937
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig# trace_info : t_17938
from megatron.core.datasets.indexed_dataset import IndexedDataset              # trace_info : t_17939
from megatron.core.datasets.megatron_dataset import MegatronDataset            # trace_info : t_17940
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer        # trace_info : t_17941
from megatron.core.datasets.utils import Split, log_single_rank                # trace_info : t_17942
                                                                               # trace_info : t_17943
logger = logging.getLogger(__name__)                                           # trace_info : t_17944

_PAD_TOKEN_ID = -1


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets"""

    reset_position_ids: bool = None
    """Option to reset the position IDs in the dataset at an interval"""

    reset_attention_mask: bool = None
    """Option to reset the attention mask from the dataset"""

    eod_mask_loss: bool = None
    """Option to enable the EOD mask loss"""

    create_attention_mask: bool = True
    """Option to enable the attention masks generation. Can be disabled if attention kernel
       generates masks by itself.
    """

    drop_last_partial_validation_sequence: bool = True
    """Option to drop the last partial validation sequence"""

    add_extra_token_to_sequence: bool = True
    """Option to draw sequences with one extra token to ensure the sample input tokens and sample
       output tokens are both of the desired sequence length
    """

    def __post_init__(self) -> None:
        """Do asserts and set fields post init
        """
        super().__post_init__()                                                # trace_info : t_17945

        assert self.tokenizer is not None                                      # trace_info : t_18000

        assert self.reset_position_ids is not None                             # trace_info : t_18001
        assert self.reset_attention_mask is not None                           # trace_info : t_18002
        assert self.eod_mask_loss is not None                                  # trace_info : t_18003


class GPTDataset(MegatronDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(                                                      # trace_info : t_18274, t_18276, t_18489, t_18491, t_18704, ...
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config# trace_info : t_18275, t_18490, t_18705
        )
        self.masks_and_position_ids_are_cacheable = not any(                   # trace_info : t_18309, t_18314, t_18524, t_18529, t_18739, ...
            [                                                                  # trace_info : t_18313, t_18528, t_18743
                self.config.reset_position_ids,                                # trace_info : t_18310, t_18525, t_18740
                self.config.reset_attention_mask,                              # trace_info : t_18311, t_18526, t_18741
                self.config.eod_mask_loss,                                     # trace_info : t_18312, t_18527, t_18742
            ]
        )
        self.masks_and_position_ids_are_cached = False                         # trace_info : t_18315, t_18530, t_18745
        self.cached_attention_mask = None                                      # trace_info : t_18316, t_18531, t_18746
        self.cached_loss_mask = None                                           # trace_info : t_18317, t_18532, t_18747
        self.cached_position_ids = None                                        # trace_info : t_18318, t_18533, t_18748

        try:                                                                   # trace_info : t_18319, t_18534, t_18749
            self._pad_token_id = self.config.tokenizer.pad                     # trace_info : t_18320, t_18535, t_18750
        except:                                                                # trace_info : t_18322, t_18537, t_18752
            self._pad_token_id = _PAD_TOKEN_ID                                 # trace_info : t_18323, t_18538, t_18753

        (                                                                      # trace_info : t_18443, t_18658, t_18873
            self.document_index,                                               # trace_info : t_18444, t_18659, t_18874
            self.sample_index,                                                 # trace_info : t_18445, t_18660, t_18875
            self.shuffle_index,                                                # trace_info : t_18446, t_18661, t_18876
        ) = self._build_document_sample_shuffle_indices()                      # trace_info : t_18324, t_18539, t_18754

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]                     # trace_info : t_18217

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)# trace_info : t_18084

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1                                  # trace_info : t_18886, t_18890, t_18894, t_18915, t_18965, ...

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )
        assert len(document_ids) == len(
            sample_parts
        ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        length = sum(map(len, sample_parts))

        # Pad the sample if necessary
        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id]
                * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index
        
        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the shuffle index
        """
        path_to_cache = self.config.path_to_cache                              # trace_info : t_18325, t_18540, t_18755
        if path_to_cache is None and not self.config.mock:                     # trace_info : t_18326, t_18541, t_18756
            path_to_cache = os.path.join(                                      # trace_info : t_18327, t_18329, t_18542, t_18544, t_18757, ...
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"# trace_info : t_18328, t_18543, t_18758
            )

        if path_to_cache:                                                      # trace_info : t_18330, t_18545, t_18760
            get_path_to = lambda suffix: os.path.join(                         # trace_info : t_18331, t_18333, t_18336, t_18338, t_18341, ...
                path_to_cache,                                                 # trace_info : t_18334, t_18339, t_18344, t_18349, t_18549, ...
                f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}-{suffix}",# trace_info : t_18335, t_18340, t_18345, t_18350, t_18550, ...
            )
            path_to_description = get_path_to("description.txt")               # trace_info : t_18332, t_18547, t_18762
            path_to_document_index = get_path_to("document_index.npy")         # trace_info : t_18337, t_18552, t_18767
            path_to_sample_index = get_path_to("sample_index.npy")             # trace_info : t_18342, t_18557, t_18772
            path_to_shuffle_index = get_path_to("shuffle_index.npy")           # trace_info : t_18347, t_18562, t_18777
            cache_hit = all(                                                   # trace_info : t_18352, t_18361, t_18567, t_18576, t_18782, ...
                map(                                                           # trace_info : t_18353, t_18360, t_18568, t_18575, t_18783, ...
                    os.path.isfile,                                            # trace_info : t_18354, t_18569, t_18784
                    [                                                          # trace_info : t_18359, t_18574, t_18789
                        path_to_description,                                   # trace_info : t_18355, t_18570, t_18785
                        path_to_document_index,                                # trace_info : t_18356, t_18571, t_18786
                        path_to_sample_index,                                  # trace_info : t_18357, t_18572, t_18787
                        path_to_shuffle_index,                                 # trace_info : t_18358, t_18573, t_18788
                    ],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (                                              # trace_info : t_18362, t_18364, t_18577, t_18579, t_18792, ...
            not cache_hit                                                      # trace_info : t_18363, t_18578, t_18793
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):

            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch
                    - self.config.add_extra_token_to_sequence
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (
                    num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
                log_single_rank(
                    logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_single_rank(
                logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )

            drop_last_partial_sequence = True
            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

            # Build the sample index
            from megatron.core.datasets import helpers

            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
            else:
                drop_last_partial_sequence = True

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            sample_index = helpers.build_sample_idx(
                self.dataset.sequence_lengths,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last_partial_sequence,
                self.config.add_extra_token_to_sequence,
            )

            # Build the shuffle index
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save the {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index

        log_single_rank(                                                       # trace_info : t_18365, t_18367, t_18580, t_18582, t_18795, ...
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"# trace_info : t_18366, t_18581, t_18796
        )

        log_single_rank(                                                       # trace_info : t_18375, t_18379, t_18590, t_18594, t_18805, ...
            logger,                                                            # trace_info : t_18376, t_18591, t_18806
            logging.INFO,                                                      # trace_info : t_18377, t_18592, t_18807
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",# trace_info : t_18378, t_18593, t_18808
        )
        t_beg = time.time()                                                    # trace_info : t_18387, t_18602, t_18817
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')# trace_info : t_18388, t_18603, t_18818
        t_end = time.time()                                                    # trace_info : t_18389, t_18604, t_18819
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")# trace_info : t_18390, t_18605, t_18820

        log_single_rank(                                                       # trace_info : t_18394, t_18398, t_18609, t_18613, t_18824, ...
            logger,                                                            # trace_info : t_18395, t_18610, t_18825
            logging.INFO,                                                      # trace_info : t_18396, t_18611, t_18826
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",# trace_info : t_18397, t_18612, t_18827
        )
        t_beg = time.time()                                                    # trace_info : t_18406, t_18621, t_18836
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')# trace_info : t_18407, t_18622, t_18837
        t_end = time.time()                                                    # trace_info : t_18408, t_18623, t_18838
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")# trace_info : t_18409, t_18624, t_18839

        log_single_rank(                                                       # trace_info : t_18413, t_18417, t_18628, t_18632, t_18843, ...
            logger,                                                            # trace_info : t_18414, t_18629, t_18844
            logging.INFO,                                                      # trace_info : t_18415, t_18630, t_18845
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",# trace_info : t_18416, t_18631, t_18846
        )
        t_beg = time.time()                                                    # trace_info : t_18425, t_18640, t_18855
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')# trace_info : t_18426, t_18641, t_18856
        t_end = time.time()                                                    # trace_info : t_18427, t_18642, t_18857
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")# trace_info : t_18428, t_18643, t_18858

        log_single_rank(                                                       # trace_info : t_18432, t_18434, t_18647, t_18649, t_18862, ...
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"# trace_info : t_18433, t_18648, t_18863
        )

        return document_index, sample_index, shuffle_index                     # trace_info : t_18442, t_18657, t_18872

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index
    """
    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle
    
    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index
    """
    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


class MockGPTLowLevelDataset:

    seed: int = 0
    size: int = 100000
    max_sequence_length: int = 4096

    def __init__(self, tokenizer: MegatronTokenizer) -> None:
        self.tokenizer = tokenizer
        rng = numpy.random.default_rng(seed=self.seed)
        self.sequence_lengths = rng.integers(
            low=1, high=self.max_sequence_length, size=self.size, dtype=numpy.int32
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        sample = numpy.int64(
            numpy.concatenate([numpy.arange(length - 1) + 1, [self.tokenizer.eod]])
        )
        return sample

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return self[idx][offset : offset + length]


class MockGPTDataset(GPTDataset):
    """The mock GPT dataset

    Args:
        indexed_dataset (MockGPTLowLevelDataset): The MockGPTLowLevelDataset around which to build the MockGPTDataset

        dataset_path (Optional[str]): This argument is of no consequence for the MockGPTDataset

        indices (numpy.ndarray): The set of the dataset indices to expose

        num_samples (int): The number of samples to draw from the dataset

        index_split (Split): The indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: MockGPTLowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        assert config.mock

        if num_samples is None:
            num_samples = len(indices)

        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: MockGPTLowLevelDataset) -> int:
        """Abstract method implementation

        Args:
            low_level_dataset (MockGPTLowLevelDataset): The underlying MockGPTLowLevelDataset

        Returns:
            int: The number of unique elements in the underlying MockGPTLowLevelDataset
        """
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: Optional[str], config: GPTDatasetConfig
    ) -> MockGPTLowLevelDataset:
        """Abstract method implementation

        Args:
            dataset_path (Optional[str]): This argument is of no consequence for the MockGPTLowLevelDataset

            config (GPTDatasetConfig): The config

        Returns:
            MockGPTLowLevelDataset: The underlying MockGPTLowLevelDataset
        """
        return MockGPTLowLevelDataset(config.tokenizer)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (int): The integer seed for mock data generation

        Returns:
            Dict[str, numpy.ndarray]: The mock sample information wrapped in a dictionary
        """
        if idx is not None and idx >= self.num_samples:
            raise IndexError(
                f"The index {idx} exceeds the available number of samples ({self.num_samples})"
            )

        return super().__getitem__(idx)
