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

    if dataset is None:                                                        # trace_info : t_17106, t_17156, t_17203
        return None
    args = get_args()                                                          # trace_info : t_17107, t_17157, t_17204

    # Megatron sampler
    if args.dataloader_type == 'single':                                       # trace_info : t_17111, t_17161, t_17208
        batch_sampler = MegatronPretrainingSampler(                            # trace_info : t_17112, t_17133, t_17162, t_17183, t_17209, ...
            total_samples=len(dataset),                                        # trace_info : t_17113, t_17163, t_17210
            consumed_samples=consumed_samples,                                 # trace_info : t_17115, t_17165, t_17212
            micro_batch_size=args.micro_batch_size,                            # trace_info : t_17116, t_17166, t_17213
            data_parallel_rank=mpu.get_data_parallel_rank(),                   # trace_info : t_17117, t_17167, t_17214
            data_parallel_size=mpu.get_data_parallel_world_size())             # trace_info : t_17125, t_17175, t_17222
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
    return torch.utils.data.DataLoader(dataset,                                # trace_info : t_17146, t_17151, t_17196, t_17201, t_17243, ...
                                       batch_sampler=batch_sampler,            # trace_info : t_17147, t_17197, t_17244
                                       num_workers=args.num_workers,           # trace_info : t_17148, t_17198, t_17245
                                       pin_memory=True,                        # trace_info : t_17149, t_17199, t_17246
                                       persistent_workers=True if args.num_workers > 0 else False,# trace_info : t_17150, t_17200, t_17247
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples                                     # trace_info : t_17134, t_17184, t_17231
        self.consumed_samples = consumed_samples                               # trace_info : t_17135, t_17185, t_17232
        self.micro_batch_size = micro_batch_size                               # trace_info : t_17136, t_17186, t_17233
        self.data_parallel_rank = data_parallel_rank                           # trace_info : t_17137, t_17187, t_17234
        self.micro_batch_times_data_parallel_size = \                          # trace_info : t_17139, t_17189, t_17236
            self.micro_batch_size * data_parallel_size                         # trace_info : t_17138, t_17188, t_17235
        self.drop_last = drop_last                                             # trace_info : t_17140, t_17190, t_17237

        # Sanity checks.
        assert self.total_samples > 0, \                                       # trace_info : t_17141, t_17191, t_17238
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \                   # trace_info : t_17142, t_17192, t_17239
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0                                       # trace_info : t_17143, t_17193, t_17240
        assert data_parallel_size > 0                                          # trace_info : t_17144, t_17194, t_17241
        assert self.data_parallel_rank < data_parallel_size, \                 # trace_info : t_17145, t_17195, t_17242
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size            # trace_info : t_17276, t_17285, t_17294, t_17303, t_17318, ...
        end_idx = start_idx + self.micro_batch_size                            # trace_info : t_17277, t_17286, t_17295, t_17304, t_17319, ...
        return start_idx, end_idx                                              # trace_info : t_17278, t_17287, t_17296, t_17305, t_17320, ...

    def __iter__(self):
        batch = []                                                             # trace_info : t_17271, t_17313, t_17355
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):           # trace_info : t_17272, t_17281, t_17290, t_17299, t_17314, ...
            batch.append(idx)                                                  # trace_info : t_17273, t_17282, t_17291, t_17300, t_17315, ...
            if len(batch) == self.micro_batch_times_data_parallel_size:        # trace_info : t_17274, t_17283, t_17292, t_17301, t_17316, ...
                start_idx, end_idx = self.get_start_end_idx()                  # trace_info : t_17275, t_17284, t_17293, t_17302, t_17317, ...
                yield batch[start_idx:end_idx]                                 # trace_info : t_17279, t_17288, t_17297, t_17306, t_17321, ...
                batch = []                                                     # trace_info : t_17280, t_17289, t_17298, t_17322, t_17331, ...

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
