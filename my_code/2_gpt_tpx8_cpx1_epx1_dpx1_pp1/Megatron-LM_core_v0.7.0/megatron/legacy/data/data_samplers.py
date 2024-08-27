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

    if dataset is None:                                                        # trace_info : t_18907, t_18957, t_19004
        return None
    args = get_args()                                                          # trace_info : t_18908, t_18958, t_19005

    # Megatron sampler
    if args.dataloader_type == 'single':                                       # trace_info : t_18912, t_18962, t_19009
        batch_sampler = MegatronPretrainingSampler(                            # trace_info : t_18913, t_18934, t_18963, t_18984, t_19010, ...
            total_samples=len(dataset),                                        # trace_info : t_18914, t_18964, t_19011
            consumed_samples=consumed_samples,                                 # trace_info : t_18916, t_18966, t_19013
            micro_batch_size=args.micro_batch_size,                            # trace_info : t_18917, t_18967, t_19014
            data_parallel_rank=mpu.get_data_parallel_rank(),                   # trace_info : t_18918, t_18968, t_19015
            data_parallel_size=mpu.get_data_parallel_world_size())             # trace_info : t_18926, t_18976, t_19023
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
    return torch.utils.data.DataLoader(dataset,                                # trace_info : t_18947, t_18952, t_18997, t_19002, t_19044, ...
                                       batch_sampler=batch_sampler,            # trace_info : t_18948, t_18998, t_19045
                                       num_workers=args.num_workers,           # trace_info : t_18949, t_18999, t_19046
                                       pin_memory=True,                        # trace_info : t_18950, t_19000, t_19047
                                       persistent_workers=True if args.num_workers > 0 else False,# trace_info : t_18951, t_19001, t_19048
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples                                     # trace_info : t_18935, t_18985, t_19032
        self.consumed_samples = consumed_samples                               # trace_info : t_18936, t_18986, t_19033
        self.micro_batch_size = micro_batch_size                               # trace_info : t_18937, t_18987, t_19034
        self.data_parallel_rank = data_parallel_rank                           # trace_info : t_18938, t_18988, t_19035
        self.micro_batch_times_data_parallel_size = \                          # trace_info : t_18940, t_18990, t_19037
            self.micro_batch_size * data_parallel_size                         # trace_info : t_18939, t_18989, t_19036
        self.drop_last = drop_last                                             # trace_info : t_18941, t_18991, t_19038

        # Sanity checks.
        assert self.total_samples > 0, \                                       # trace_info : t_18942, t_18992, t_19039
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \                   # trace_info : t_18943, t_18993, t_19040
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0                                       # trace_info : t_18944, t_18994, t_19041
        assert data_parallel_size > 0                                          # trace_info : t_18945, t_18995, t_19042
        assert self.data_parallel_rank < data_parallel_size, \                 # trace_info : t_18946, t_18996, t_19043
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size            # trace_info : t_19077, t_19086, t_19095, t_19104, t_19119, ...
        end_idx = start_idx + self.micro_batch_size                            # trace_info : t_19078, t_19087, t_19096, t_19105, t_19120, ...
        return start_idx, end_idx                                              # trace_info : t_19079, t_19088, t_19097, t_19106, t_19121, ...

    def __iter__(self):
        batch = []                                                             # trace_info : t_19072, t_19114, t_19156
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):           # trace_info : t_19073, t_19082, t_19091, t_19100, t_19115, ...
            batch.append(idx)                                                  # trace_info : t_19074, t_19083, t_19092, t_19101, t_19116, ...
            if len(batch) == self.micro_batch_times_data_parallel_size:        # trace_info : t_19075, t_19084, t_19093, t_19102, t_19117, ...
                start_idx, end_idx = self.get_start_end_idx()                  # trace_info : t_19076, t_19085, t_19094, t_19103, t_19118, ...
                yield batch[start_idx:end_idx]                                 # trace_info : t_19080, t_19089, t_19098, t_19107, t_19122, ...
                batch = []                                                     # trace_info : t_19081, t_19090, t_19099, t_19123, t_19132, ...

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
