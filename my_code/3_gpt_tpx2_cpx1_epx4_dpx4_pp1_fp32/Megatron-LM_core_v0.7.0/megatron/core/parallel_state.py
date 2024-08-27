# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
import warnings
from datetime import timedelta
from typing import List, Optional

import torch

from .utils import GlobalMemoryBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = None


_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# MOE logging
_MOE_AUX_LOSSES_LOGGING_TRACKER = {}


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:                                              # trace_info : t_4702, t_4713, t_4996, t_5009, t_5405, ...
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None                                                            # trace_info : t_4703, t_4714, t_4997, t_5010, t_5406, ...


def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool],
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example, 
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then 
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the 
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the 
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).
        
        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:                     # trace_info : t_4461, t_4734, t_5032, t_5477, t_5818, ...
        r = [init]                                                             # trace_info : t_4467, t_4485, t_4497, t_4517, t_4537, ...
        for v in a:                                                            # trace_info : t_4468, t_4471, t_4474, t_4477, t_4480, ...
            init = init * v                                                    # trace_info : t_4469, t_4472, t_4475, t_4478, t_4487, ...
            r.append(init)                                                     # trace_info : t_4470, t_4473, t_4476, t_4479, t_4488, ...
        return r                                                               # trace_info : t_4481, t_4490, t_4508, t_4522, t_4542, ...

    def inner_product(a: List[int], b: List[int]) -> int:                      # trace_info : t_4462, t_4735, t_5033, t_5478, t_5819, ...
        return sum([x * y for x, y in zip(a, b)])                              # trace_info : t_4528, t_4530, t_4548, t_4550, t_4568, ...

    def decompose(index, shape, stride=None):                                  # trace_info : t_4463, t_4736, t_5034, t_5479, t_5820, ...
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:                                                     # trace_info : t_4495, t_4515, t_4535, t_4555, t_4575, ...
            stride = prefix_product(shape)                                     # trace_info : t_4496, t_4516, t_4536, t_4556, t_4576, ...
        idx = [(index // d) % s for s, d in zip(shape, stride)]                # trace_info : t_4509, t_4523, t_4543, t_4563, t_4583, ...
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index            # trace_info : t_4510, t_4524, t_4544, t_4564, t_4584, ...
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx                                                             # trace_info : t_4511, t_4525, t_4545, t_4565, t_4585, ...

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]               # trace_info : t_4464, t_4737, t_5035, t_5480, t_5821, ...
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]         # trace_info : t_4465, t_4738, t_5036, t_5481, t_5822, ...

    global_stride = prefix_product(parallel_size)                              # trace_info : t_4466, t_4739, t_5037, t_5482, t_5823, ...
    masked_stride = [d for d, m in zip(global_stride, mask) if m]              # trace_info : t_4482, t_4755, t_5053, t_5498, t_5839, ...
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]        # trace_info : t_4483, t_4756, t_5054, t_5499, t_5840, ...

    group_size = prefix_product(masked_shape)[-1]                              # trace_info : t_4484, t_4757, t_5055, t_5500, t_5841, ...
    num_of_group = world_size // group_size                                    # trace_info : t_4491, t_4767, t_5062, t_5510, t_5848, ...

    ranks = []                                                                 # trace_info : t_4492, t_4768, t_5063, t_5511, t_5849, ...
    for group_index in range(num_of_group):                                    # trace_info : t_4493, t_4595, t_4697, t_4769, t_4880, ...
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)          # trace_info : t_4494, t_4596, t_4770, t_4881, t_5065, ...
        rank = []                                                              # trace_info : t_4512, t_4614, t_4785, t_4896, t_5083, ...
        for rank_in_group in range(group_size):                                # trace_info : t_4513, t_4533, t_4553, t_4573, t_4593, ...
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)       # trace_info : t_4514, t_4534, t_4554, t_4574, t_4616, ...
            rank.append(                                                       # trace_info : t_4526, t_4532, t_4546, t_4552, t_4566, ...
                inner_product(decomposed_rank_idx, masked_stride)              # trace_info : t_4527, t_4531, t_4547, t_4551, t_4567, ...
                + inner_product(decomposed_group_idx, unmasked_stride)         # trace_info : t_4529, t_4549, t_4569, t_4589, t_4631, ...
            )
        ranks.append(rank)                                                     # trace_info : t_4594, t_4696, t_4879, t_4990, t_5105, ...
    return ranks                                                               # trace_info : t_4698, t_4992, t_5401, t_5773, t_6099, ...


class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp                                                           # trace_info : t_4386
        self.ep = ep                                                           # trace_info : t_4387
        self.dp = dp                                                           # trace_info : t_4388
        self.pp = pp                                                           # trace_info : t_4389
        self.cp = cp                                                           # trace_info : t_4390
        self.world_size = tp * dp * pp * cp                                    # trace_info : t_4391

        self.name_to_size = {                                                  # trace_info : t_4397
            "tp": self.tp,                                                     # trace_info : t_4392
            "pp": self.pp,                                                     # trace_info : t_4393
            "dp": self.dp,                                                     # trace_info : t_4394
            "ep": self.ep,                                                     # trace_info : t_4395
            "cp": self.cp,                                                     # trace_info : t_4396
        }
        self.order = order                                                     # trace_info : t_4398
        order = order.lower()                                                  # trace_info : t_4399

        if 'ep' in order:                                                      # trace_info : t_4400
            if 'ep-dp' not in order and 'dp-ep' not in order:                  # trace_info : t_4401
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():                                  # trace_info : t_4402, t_4405, t_4408, t_4411, t_4414, ...
            if name not in order and self.name_to_size[name] != 1:             # trace_info : t_4403, t_4406, t_4409, t_4412, t_4415
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:                                            # trace_info : t_4404, t_4407, t_4410, t_4413, t_4416
                order = order + '-' + name

        self.order_w_ep = order                                                # trace_info : t_4418
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])# trace_info : t_4419
        self.ordered_size_wo_ep = []                                           # trace_info : t_4420
        self.ordered_size_w_ep = []                                            # trace_info : t_4421

        for token in order.split('-'):                                         # trace_info : t_4422, t_4427, t_4432, t_4436, t_4440, ...
            if token == 'dp':                                                  # trace_info : t_4423, t_4428, t_4433, t_4437, t_4441
                self.ordered_size_w_ep.append(self.dp // self.ep)              # trace_info : t_4438
                self.ordered_size_wo_ep.append(self.dp)                        # trace_info : t_4439
            elif token == 'ep':                                                # trace_info : t_4424, t_4429, t_4434, t_4442
                self.ordered_size_w_ep.append(self.ep)                         # trace_info : t_4435
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])        # trace_info : t_4425, t_4430, t_4443
                self.ordered_size_wo_ep.append(self.name_to_size[token])       # trace_info : t_4426, t_4431, t_4444

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')                                       # trace_info : t_4453, t_4724, t_5024, t_5467, t_5810, ...
        token = token.split('-')                                               # trace_info : t_4454, t_4725, t_5025, t_5468, t_5811, ...
        mask = [False] * len(ordered_token)                                    # trace_info : t_4455, t_4726, t_5026, t_5469, t_5812, ...
        for t in token:                                                        # trace_info : t_4456, t_4458, t_4727, t_4729, t_4731, ...
            mask[ordered_token.index(t)] = True                                # trace_info : t_4457, t_4728, t_4730, t_5028, t_5471, ...
        return mask                                                            # trace_info : t_4459, t_4732, t_5030, t_5475, t_5816, ...

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        '''
        if independent_ep:                                                     # trace_info : t_4449, t_4720, t_5020, t_5463, t_5806, ...
            parallel_size = self.ordered_size_w_ep                             # trace_info : t_7292, t_7562, t_7838
            order = self.order_w_ep                                            # trace_info : t_7293, t_7563, t_7839
        else:
            parallel_size = self.ordered_size_wo_ep                            # trace_info : t_4450, t_4721, t_5021, t_5464, t_5807, ...
            order = self.order_wo_ep                                           # trace_info : t_4451, t_4722, t_5022, t_5465, t_5808, ...
        mask = self.get_mask(order, token)                                     # trace_info : t_4452, t_4723, t_5023, t_5466, t_5809, ...
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)# trace_info : t_4460, t_4733, t_5031, t_5476, t_5817, ...
        return ranks                                                           # trace_info : t_4699, t_4993, t_5402, t_5774, t_6100, ...


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.
        
        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()                                  # trace_info : t_4359
    world_size: int = torch.distributed.get_world_size()                       # trace_info : t_4360

    if (
        world_size                                                             # trace_info : t_4361, t_4363, t_4365
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)# trace_info : t_4362
        != 0                                                                   # trace_info : t_4364
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (                                  # trace_info : t_4366, t_4368
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size# trace_info : t_4367
    )

    if data_parallel_size % expert_model_parallel_size != 0:                   # trace_info : t_4369
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:           # trace_info : t_4370
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size# trace_info : t_4371
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size# trace_info : t_4372

    if virtual_pipeline_model_parallel_size is not None:                       # trace_info : t_4373
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:                         # trace_info : t_4374
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()                                        # trace_info : t_4375

    nccl_comm_cfgs = {}                                                        # trace_info : t_4376
    if nccl_communicator_config_path is not None:                              # trace_info : t_4377
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    rank_generator = RankGenerator(                                            # trace_info : t_4378, t_4385
        tp=tensor_model_parallel_size,                                         # trace_info : t_4379
        ep=expert_model_parallel_size,                                         # trace_info : t_4380
        dp=data_parallel_size,                                                 # trace_info : t_4381
        pp=pipeline_model_parallel_size,                                       # trace_info : t_4382
        cp=context_parallel_size,                                              # trace_info : t_4383
        order=order,                                                           # trace_info : t_4384
    )
    timeout = timedelta(minutes=distributed_timeout_minutes)                   # trace_info : t_4446

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'# trace_info : t_4447

    for ranks in rank_generator.get_ranks('dp'):                               # trace_info : t_4448, t_4710, t_4718
        group = torch.distributed.new_group(                                   # trace_info : t_4700, t_4704, t_4711, t_4715
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)# trace_info : t_4701, t_4712
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")# trace_info : t_4705, t_4716
        if rank in ranks:                                                      # trace_info : t_4706, t_4717
            _DATA_PARALLEL_GROUP = group                                       # trace_info : t_4707
            _DATA_PARALLEL_GROUP_GLOO = group_gloo                             # trace_info : t_4708
            _DATA_PARALLEL_GLOBAL_RANKS = ranks                                # trace_info : t_4709
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):                    # trace_info : t_4719, t_5006, t_5016
        group_with_cp = torch.distributed.new_group(                           # trace_info : t_4994, t_4998, t_5007, t_5011
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)# trace_info : t_4995, t_5008
        )
        group_with_cp_gloo = torch.distributed.new_group(                      # trace_info : t_4999, t_5001, t_5012, t_5014
            ranks_with_cp, timeout=timeout, backend="gloo"                     # trace_info : t_5000, t_5013
        )
        if rank in ranks_with_cp:                                              # trace_info : t_5002, t_5015
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp                       # trace_info : t_5003
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo             # trace_info : t_5004
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp                # trace_info : t_5005

    # Apply SHARP to DP process groups
    if use_sharp:                                                              # trace_info : t_5017
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'# trace_info : t_5018
    for ranks in rank_generator.get_ranks('cp'):                               # trace_info : t_5019, t_5411, t_5418, t_5425, t_5432, ...
        group = torch.distributed.new_group(                                   # trace_info : t_5403, t_5407, t_5412, t_5416, t_5419, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)# trace_info : t_5404, t_5413, t_5420, t_5427, t_5434, ...
        )
        if rank in ranks:                                                      # trace_info : t_5408, t_5417, t_5424, t_5431, t_5438, ...
            _CONTEXT_PARALLEL_GROUP = group                                    # trace_info : t_5409
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks                             # trace_info : t_5410

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'# trace_info : t_5461
    for ranks in rank_generator.get_ranks('tp-pp'):                            # trace_info : t_5462, t_5782, t_5789, t_5796, t_5803
        group = torch.distributed.new_group(                                   # trace_info : t_5775, t_5779, t_5783, t_5787, t_5790, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)# trace_info : t_5776, t_5784, t_5791, t_5798
        )
        if rank in ranks:                                                      # trace_info : t_5780, t_5788, t_5795, t_5802
            _MODEL_PARALLEL_GROUP = group                                      # trace_info : t_5781

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_5804
    ), 'tensor model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp'):                               # trace_info : t_5805, t_6109, t_6116, t_6123, t_6130
        group = torch.distributed.new_group(                                   # trace_info : t_6101, t_6105, t_6110, t_6114, t_6117, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)# trace_info : t_6102, t_6111, t_6118, t_6125
        )
        if rank in ranks:                                                      # trace_info : t_6106, t_6115, t_6122, t_6129
            _TENSOR_MODEL_PARALLEL_GROUP = group                               # trace_info : t_6107
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks                        # trace_info : t_6108

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None                                 # trace_info : t_6131
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'  # trace_info : t_6132
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'# trace_info : t_6133
    for ranks in rank_generator.get_ranks('pp'):                               # trace_info : t_6134, t_6549, t_6575, t_6601, t_6627, ...
        group = torch.distributed.new_group(                                   # trace_info : t_6518, t_6522, t_6550, t_6554, t_6576, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)# trace_info : t_6519, t_6551, t_6577, t_6603, t_6629, ...
        )
        if rank in ranks:                                                      # trace_info : t_6523, t_6555, t_6581, t_6607, t_6633, ...
            _PIPELINE_MODEL_PARALLEL_GROUP = group                             # trace_info : t_6524
            _PIPELINE_GLOBAL_RANKS = ranks                                     # trace_info : t_6525
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:                                                     # trace_info : t_6526, t_6556, t_6582, t_6608, t_6634, ...
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks                                            # trace_info : t_6527, t_6557, t_6583, t_6609, t_6635, ...
            position_embedding_ranks = ranks                                   # trace_info : t_6528, t_6558, t_6584, t_6610, t_6636, ...

        group = torch.distributed.new_group(                                   # trace_info : t_6529, t_6533, t_6559, t_6563, t_6585, ...
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)# trace_info : t_6530, t_6560, t_6586, t_6612, t_6638, ...
        )
        if rank in embedding_ranks:                                            # trace_info : t_6534, t_6564, t_6590, t_6616, t_6642, ...
            _EMBEDDING_GROUP = group                                           # trace_info : t_6535
        if rank in ranks:                                                      # trace_info : t_6536, t_6565, t_6591, t_6617, t_6643, ...
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks                          # trace_info : t_6537

        group = torch.distributed.new_group(                                   # trace_info : t_6538, t_6544, t_6566, t_6572, t_6592, ...
            position_embedding_ranks,                                          # trace_info : t_6539, t_6567, t_6593, t_6619, t_6645, ...
            timeout=timeout,                                                   # trace_info : t_6540, t_6568, t_6594, t_6620, t_6646, ...
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),               # trace_info : t_6541, t_6569, t_6595, t_6621, t_6647, ...
        )
        if rank in position_embedding_ranks:                                   # trace_info : t_6545, t_6573, t_6599, t_6625, t_6651, ...
            _POSITION_EMBEDDING_GROUP = group                                  # trace_info : t_6546
        if rank in ranks:                                                      # trace_info : t_6547, t_6574, t_6600, t_6626, t_6652, ...
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks        # trace_info : t_6548

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None                                # trace_info : t_6732
    ), 'Tensor + data parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-dp-cp'):                         # trace_info : t_6733, t_7022
        group = torch.distributed.new_group(                                   # trace_info : t_7015, t_7019
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)# trace_info : t_7016
        )
        if rank in ranks:                                                      # trace_info : t_7020
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group                    # trace_info : t_7021
    for ranks in rank_generator.get_ranks('tp-dp'):                            # trace_info : t_7023, t_7286
        group = torch.distributed.new_group(                                   # trace_info : t_7279, t_7283
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)# trace_info : t_7280
        )
        if rank in ranks:                                                      # trace_info : t_7284
            _TENSOR_AND_DATA_PARALLEL_GROUP = group                            # trace_info : t_7285

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'# trace_info : t_7287
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None                              # trace_info : t_7288
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None                             # trace_info : t_7289
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO

    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):       # trace_info : t_7290, t_7559
        group = torch.distributed.new_group(                                   # trace_info : t_7552, t_7556
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)# trace_info : t_7553
        )
        if rank in ranks:                                                      # trace_info : t_7557
            _TENSOR_AND_EXPERT_PARALLEL_GROUP = group                          # trace_info : t_7558

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):          # trace_info : t_7560, t_7828, t_7835
        group = torch.distributed.new_group(                                   # trace_info : t_7821, t_7825, t_7829, t_7833
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)          # trace_info : t_7822, t_7830
        )
        if rank in ranks:                                                      # trace_info : t_7826, t_7834
            _EXPERT_MODEL_PARALLEL_GROUP = group                               # trace_info : t_7827

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):          # trace_info : t_7836, t_8256, t_8264, t_8272, t_8280, ...
        group = torch.distributed.new_group(                                   # trace_info : t_8247, t_8251, t_8257, t_8261, t_8265, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)# trace_info : t_8248, t_8258, t_8266, t_8274, t_8282, ...
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")        # trace_info : t_8252, t_8262, t_8270, t_8278, t_8286, ...
        if rank in ranks:                                                      # trace_info : t_8253, t_8263, t_8271, t_8279, t_8287, ...
            _DATA_MODULO_EXPERT_PARALLEL_GROUP = group                         # trace_info : t_8254
            _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo               # trace_info : t_8255

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()                                                # trace_info : t_8313


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def is_unitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.

    """
    warnings.warn(
        "is_unitialized is deprecated, use is_initialized instead", DeprecationWarning,
    )
    return not is_initialized()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_4346
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False                                                           # trace_info : t_4347
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'# trace_info : t_15645, t_15683, t_21099, t_21469, t_25444, ...
    return _MODEL_PARALLEL_GROUP                                               # trace_info : t_15646, t_15684, t_21100, t_21470, t_25445, ...


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_8323, t_8356, t_8424, t_8429, t_8617, ...
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None                           # trace_info : t_8324, t_8357, t_8425, t_8430, t_8618, ...
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP                                        # trace_info : t_8325, t_8358, t_8426, t_8431, t_8619, ...


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None                             # trace_info : t_8333, t_8344, t_8752, t_8761, t_8772, ...
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP                                      # trace_info : t_8334, t_8345, t_8753, t_8762, t_8773, ...


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_12376, t_12428, t_12442, t_13839, t_14605, ...
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None                           # trace_info : t_12429, t_13840, t_14606, t_15653
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP                                    # trace_info : t_12430, t_13841, t_14607, t_15654
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'# trace_info : t_12377, t_12443, t_17016, t_17024, t_17066, ...
        return _DATA_PARALLEL_GROUP                                            # trace_info : t_12378, t_12444, t_17017, t_17025, t_17067, ...


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_15656
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None                      # trace_info : t_15657
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO                               # trace_info : t_15658
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_expert_model_parallel_group():
    assert (
        _EXPERT_MODEL_PARALLEL_GROUP is not None
    ), 'expert model parallel group is not initialized'
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None                          # trace_info : t_8418, t_10248, t_10266, t_11388, t_11406, ...
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP                                   # trace_info : t_8419, t_10249, t_10267, t_11389, t_11407, ...


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None                         # trace_info : t_12431, t_15705
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP                                  # trace_info : t_12432, t_15706


def get_data_modulo_expert_parallel_group_gloo():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is not None                    # trace_info : t_15708
    ), 'data modulo expert parallel group-gloo is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO                             # trace_info : t_15709


def set_expert_model_parallel_world_size(world_size):
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:                      # trace_info : t_8321, t_8422, t_8615, t_9399, t_9674, ...
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())# trace_info : t_8322, t_8423, t_8616, t_9400, t_9675, ...


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:                    # trace_info : t_8331, t_8750, t_8774, t_9513, t_9579, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())# trace_info : t_8332, t_8751, t_8775, t_9514, t_9580, ...


def set_expert_model_parallel_rank(rank):
    """Set expert model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:                            # trace_info : t_8354, t_8427, t_9406, t_9820, t_9961, ...
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group()) # trace_info : t_8355, t_8428, t_9407, t_9821, t_9962, ...


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:                          # trace_info : t_8342, t_8759, t_8770, t_9574, t_10714, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())# trace_info : t_8343, t_8760, t_8771, t_9575, t_10715, ...


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_8755, t_16150, t_16349, t_16564, t_16779, ...
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None       # trace_info : t_8756, t_16151, t_16350, t_16565, t_16780, ...
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0                             # trace_info : t_8758, t_16153, t_16352, t_16567, t_16782, ...


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_8764, t_20015, t_21686, t_24360, t_26031, ...
        virtual_pipeline_model_parallel_world_size = (                         # trace_info : t_8767, t_20018, t_24363, t_28708
            get_virtual_pipeline_model_parallel_world_size()                   # trace_info : t_8765, t_20016, t_24361, t_28706
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (# trace_info : t_8768, t_20019, t_24364, t_28709
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)# trace_info : t_8769, t_20020, t_21687, t_24365, t_26032, ...


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()                                        # trace_info : t_20550, t_24895, t_29240
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:                                                         # trace_info : t_20551, t_24896, t_29241
        return rank in _EMBEDDING_GLOBAL_RANKS                                 # trace_info : t_20552, t_24897, t_29242
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()                                        # trace_info : t_20561, t_24906, t_29251
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS                            # trace_info : t_20562, t_24907, t_29252


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:                          # trace_info : t_20091, t_24436, t_28781
        return True                                                            # trace_info : t_20096, t_24441, t_28786
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK                               # trace_info : t_16146


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE                         # trace_info : t_8757, t_8766, t_9519, t_9585, t_10725, ...


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None                        # trace_info : t_18142, t_18150, t_18158, t_18166, t_18174, ...
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]                              # trace_info : t_18143, t_18151, t_18159, t_18167, t_18175, ...


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_12439, t_17020, t_17070, t_17117, t_21712, ...
        return torch.distributed.get_world_size(                               # trace_info : t_12440, t_12445, t_17021, t_17026, t_17071, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_12441, t_17022, t_17072, t_17119, t_21714, ...
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_12373, t_13836, t_14602, t_17012, t_17062, ...
        return torch.distributed.get_rank(                                     # trace_info : t_12374, t_12379, t_13837, t_13842, t_14603, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_12375, t_13838, t_14604, t_17014, t_17064, ...
        )
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    else:
        return 0


def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE:                                  # trace_info : t_10244, t_11384
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_10245, t_11385
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(# trace_info : t_10246, t_10250, t_11386, t_11390
            group=get_tensor_and_expert_parallel_group()                       # trace_info : t_10247, t_11387
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()# trace_info : t_10251, t_11391
    else:
        return 0


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_18996, t_19751, t_23341, t_24096, t_27686, ...
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(# trace_info : t_18997, t_19001, t_19752, t_19756, t_23342, ...
            group=get_tensor_and_expert_parallel_group()                       # trace_info : t_18998, t_19753, t_23343, t_24098, t_27688, ...
        )
        return tensor_and_expert_parallel_world_size                           # trace_info : t_19002, t_19757, t_23347, t_24102, t_27692, ...
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:                                        # trace_info : t_8414, t_10262, t_11402, t_15686
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_8415, t_10263, t_11403, t_15687
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(          # trace_info : t_8416, t_8420, t_10264, t_10268, t_11404, ...
            group=get_tensor_and_expert_parallel_group()                       # trace_info : t_8417, t_10265, t_11405, t_15689
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()# trace_info : t_8421, t_10269, t_11409, t_15693
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def get_tensor_and_expert_parallel_rank():
    """Return my rank for the tensor and expert parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_tensor_and_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'# trace_info : t_8314
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()                               # trace_info : t_8315


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'# trace_info : t_18426, t_18502, t_18818, t_19189, t_19263, ...
    return _GLOBAL_MEMORY_BUFFER                                               # trace_info : t_18427, t_18503, t_18819, t_19190, t_19264, ...


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None
