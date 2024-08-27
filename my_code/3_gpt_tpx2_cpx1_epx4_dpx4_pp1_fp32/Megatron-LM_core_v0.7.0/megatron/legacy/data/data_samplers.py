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

    if dataset is None:                                                        # trace_info : t_17000, t_17050, t_17097
        return None
    args = get_args()                                                          # trace_info : t_17001, t_17051, t_17098

    # Megatron sampler
    if args.dataloader_type == 'single':                                       # trace_info : t_17005, t_17055, t_17102
        batch_sampler = MegatronPretrainingSampler(                            # trace_info : t_17006, t_17027, t_17056, t_17077, t_17103, ...
            total_samples=len(dataset),                                        # trace_info : t_17007, t_17057, t_17104
            consumed_samples=consumed_samples,                                 # trace_info : t_17009, t_17059, t_17106
            micro_batch_size=args.micro_batch_size,                            # trace_info : t_17010, t_17060, t_17107
            data_parallel_rank=mpu.get_data_parallel_rank(),                   # trace_info : t_17011, t_17061, t_17108
            data_parallel_size=mpu.get_data_parallel_world_size())             # trace_info : t_17019, t_17069, t_17116
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
    return torch.utils.data.DataLoader(dataset,                                # trace_info : t_17040, t_17045, t_17090, t_17095, t_17137, ...
                                       batch_sampler=batch_sampler,            # trace_info : t_17041, t_17091, t_17138
                                       num_workers=args.num_workers,           # trace_info : t_17042, t_17092, t_17139
                                       pin_memory=True,                        # trace_info : t_17043, t_17093, t_17140
                                       persistent_workers=True if args.num_workers > 0 else False,# trace_info : t_17044, t_17094, t_17141
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples                                     # trace_info : t_17028, t_17078, t_17125
        self.consumed_samples = consumed_samples                               # trace_info : t_17029, t_17079, t_17126
        self.micro_batch_size = micro_batch_size                               # trace_info : t_17030, t_17080, t_17127
        self.data_parallel_rank = data_parallel_rank                           # trace_info : t_17031, t_17081, t_17128
        self.micro_batch_times_data_parallel_size = \                          # trace_info : t_17033, t_17083, t_17130
            self.micro_batch_size * data_parallel_size                         # trace_info : t_17032, t_17082, t_17129
        self.drop_last = drop_last                                             # trace_info : t_17034, t_17084, t_17131

        # Sanity checks.
        assert self.total_samples > 0, \                                       # trace_info : t_17035, t_17085, t_17132
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \                   # trace_info : t_17036, t_17086, t_17133
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0                                       # trace_info : t_17037, t_17087, t_17134
        assert data_parallel_size > 0                                          # trace_info : t_17038, t_17088, t_17135
        assert self.data_parallel_rank < data_parallel_size, \                 # trace_info : t_17039, t_17089, t_17136
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size            # trace_info : t_17179, t_17197, t_17215, t_17233, t_17257, ...
        end_idx = start_idx + self.micro_batch_size                            # trace_info : t_17180, t_17198, t_17216, t_17234, t_17258, ...
        return start_idx, end_idx                                              # trace_info : t_17181, t_17199, t_17217, t_17235, t_17259, ...

    def __iter__(self):
        batch = []                                                             # trace_info : t_17165, t_17243, t_17321
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):           # trace_info : t_17166, t_17169, t_17172, t_17175, t_17184, ...
            batch.append(idx)                                                  # trace_info : t_17167, t_17170, t_17173, t_17176, t_17185, ...
            if len(batch) == self.micro_batch_times_data_parallel_size:        # trace_info : t_17168, t_17171, t_17174, t_17177, t_17186, ...
                start_idx, end_idx = self.get_start_end_idx()                  # trace_info : t_17178, t_17196, t_17214, t_17232, t_17256, ...
                yield batch[start_idx:end_idx]                                 # trace_info : t_17182, t_17200, t_17218, t_17236, t_17260, ...
                batch = []                                                     # trace_info : t_17183, t_17201, t_17219, t_17261, t_17279, ...

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
