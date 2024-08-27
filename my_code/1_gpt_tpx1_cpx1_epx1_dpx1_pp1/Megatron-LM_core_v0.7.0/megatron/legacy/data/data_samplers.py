# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:                                                        # trace_info : t_14040, t_14090, t_14137
        return None
    args = get_args()                                                          # trace_info : t_14041, t_14091, t_14138

    # Megatron sampler
    if args.dataloader_type == 'single':                                       # trace_info : t_14045, t_14095, t_14142
        batch_sampler = MegatronPretrainingSampler(                            # trace_info : t_14046, t_14067, t_14096, t_14117, t_14143, ...
            total_samples=len(dataset),                                        # trace_info : t_14047, t_14097, t_14144
            consumed_samples=consumed_samples,                                 # trace_info : t_14049, t_14099, t_14146
            micro_batch_size=args.micro_batch_size,                            # trace_info : t_14050, t_14100, t_14147
            data_parallel_rank=mpu.get_data_parallel_rank(),                   # trace_info : t_14051, t_14101, t_14148
            data_parallel_size=mpu.get_data_parallel_world_size())             # trace_info : t_14059, t_14109, t_14156
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,                                # trace_info : t_14080, t_14085, t_14130, t_14135, t_14177, ...
                                       batch_sampler=batch_sampler,            # trace_info : t_14081, t_14131, t_14178
                                       num_workers=args.num_workers,           # trace_info : t_14082, t_14132, t_14179
                                       pin_memory=True,                        # trace_info : t_14083, t_14133, t_14180
                                       persistent_workers=True if args.num_workers > 0 else False,# trace_info : t_14084, t_14134, t_14181
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples                                     # trace_info : t_14068, t_14118, t_14165
        self.consumed_samples = consumed_samples                               # trace_info : t_14069, t_14119, t_14166
        self.micro_batch_size = micro_batch_size                               # trace_info : t_14070, t_14120, t_14167
        self.data_parallel_rank = data_parallel_rank                           # trace_info : t_14071, t_14121, t_14168
        self.micro_batch_times_data_parallel_size = \                          # trace_info : t_14073, t_14123, t_14170
            self.micro_batch_size * data_parallel_size                         # trace_info : t_14072, t_14122, t_14169
        self.drop_last = drop_last                                             # trace_info : t_14074, t_14124, t_14171

        # Sanity checks.
        assert self.total_samples > 0, \                                       # trace_info : t_14075, t_14125, t_14172
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \                   # trace_info : t_14076, t_14126, t_14173
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0                                       # trace_info : t_14077, t_14127, t_14174
        assert data_parallel_size > 0                                          # trace_info : t_14078, t_14128, t_14175
        assert self.data_parallel_rank < data_parallel_size, \                 # trace_info : t_14079, t_14129, t_14176
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size            # trace_info : t_14210, t_14219, t_14228, t_14237, t_14252, ...
        end_idx = start_idx + self.micro_batch_size                            # trace_info : t_14211, t_14220, t_14229, t_14238, t_14253, ...
        return start_idx, end_idx                                              # trace_info : t_14212, t_14221, t_14230, t_14239, t_14254, ...

    def __iter__(self):
        batch = []                                                             # trace_info : t_14205, t_14247, t_14289
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):           # trace_info : t_14206, t_14215, t_14224, t_14233, t_14248, ...
            batch.append(idx)                                                  # trace_info : t_14207, t_14216, t_14225, t_14234, t_14249, ...
            if len(batch) == self.micro_batch_times_data_parallel_size:        # trace_info : t_14208, t_14217, t_14226, t_14235, t_14250, ...
                start_idx, end_idx = self.get_start_end_idx()                  # trace_info : t_14209, t_14218, t_14227, t_14236, t_14251, ...
                yield batch[start_idx:end_idx]                                 # trace_info : t_14213, t_14222, t_14231, t_14240, t_14255, ...
                batch = []                                                     # trace_info : t_14214, t_14223, t_14232, t_14256, t_14265, ...

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
