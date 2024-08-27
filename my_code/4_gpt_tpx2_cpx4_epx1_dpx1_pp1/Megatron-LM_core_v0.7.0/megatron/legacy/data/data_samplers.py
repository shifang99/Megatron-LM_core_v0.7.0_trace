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

    if dataset is None:                                                        # trace_info : t_16899, t_16949, t_16996
        return None
    args = get_args()                                                          # trace_info : t_16900, t_16950, t_16997

    # Megatron sampler
    if args.dataloader_type == 'single':                                       # trace_info : t_16904, t_16954, t_17001
        batch_sampler = MegatronPretrainingSampler(                            # trace_info : t_16905, t_16926, t_16955, t_16976, t_17002, ...
            total_samples=len(dataset),                                        # trace_info : t_16906, t_16956, t_17003
            consumed_samples=consumed_samples,                                 # trace_info : t_16908, t_16958, t_17005
            micro_batch_size=args.micro_batch_size,                            # trace_info : t_16909, t_16959, t_17006
            data_parallel_rank=mpu.get_data_parallel_rank(),                   # trace_info : t_16910, t_16960, t_17007
            data_parallel_size=mpu.get_data_parallel_world_size())             # trace_info : t_16918, t_16968, t_17015
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
    return torch.utils.data.DataLoader(dataset,                                # trace_info : t_16939, t_16944, t_16989, t_16994, t_17036, ...
                                       batch_sampler=batch_sampler,            # trace_info : t_16940, t_16990, t_17037
                                       num_workers=args.num_workers,           # trace_info : t_16941, t_16991, t_17038
                                       pin_memory=True,                        # trace_info : t_16942, t_16992, t_17039
                                       persistent_workers=True if args.num_workers > 0 else False,# trace_info : t_16943, t_16993, t_17040
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples                                     # trace_info : t_16927, t_16977, t_17024
        self.consumed_samples = consumed_samples                               # trace_info : t_16928, t_16978, t_17025
        self.micro_batch_size = micro_batch_size                               # trace_info : t_16929, t_16979, t_17026
        self.data_parallel_rank = data_parallel_rank                           # trace_info : t_16930, t_16980, t_17027
        self.micro_batch_times_data_parallel_size = \                          # trace_info : t_16932, t_16982, t_17029
            self.micro_batch_size * data_parallel_size                         # trace_info : t_16931, t_16981, t_17028
        self.drop_last = drop_last                                             # trace_info : t_16933, t_16983, t_17030

        # Sanity checks.
        assert self.total_samples > 0, \                                       # trace_info : t_16934, t_16984, t_17031
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \                   # trace_info : t_16935, t_16985, t_17032
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0                                       # trace_info : t_16936, t_16986, t_17033
        assert data_parallel_size > 0                                          # trace_info : t_16937, t_16987, t_17034
        assert self.data_parallel_rank < data_parallel_size, \                 # trace_info : t_16938, t_16988, t_17035
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size            # trace_info : t_17069, t_17078, t_17087, t_17096, t_17111, ...
        end_idx = start_idx + self.micro_batch_size                            # trace_info : t_17070, t_17079, t_17088, t_17097, t_17112, ...
        return start_idx, end_idx                                              # trace_info : t_17071, t_17080, t_17089, t_17098, t_17113, ...

    def __iter__(self):
        batch = []                                                             # trace_info : t_17064, t_17106, t_17148
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):           # trace_info : t_17065, t_17074, t_17083, t_17092, t_17107, ...
            batch.append(idx)                                                  # trace_info : t_17066, t_17075, t_17084, t_17093, t_17108, ...
            if len(batch) == self.micro_batch_times_data_parallel_size:        # trace_info : t_17067, t_17076, t_17085, t_17094, t_17109, ...
                start_idx, end_idx = self.get_start_end_idx()                  # trace_info : t_17068, t_17077, t_17086, t_17095, t_17110, ...
                yield batch[start_idx:end_idx]                                 # trace_info : t_17072, t_17081, t_17090, t_17099, t_17114, ...
                batch = []                                                     # trace_info : t_17073, t_17082, t_17091, t_17115, t_17124, ...

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
