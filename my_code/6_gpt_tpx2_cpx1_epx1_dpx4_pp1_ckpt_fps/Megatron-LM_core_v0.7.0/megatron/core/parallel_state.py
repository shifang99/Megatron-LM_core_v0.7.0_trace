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
    if pg_name in nccl_comm_cfgs:                                              # trace_info : t_4704, t_4715, t_4998, t_5011, t_5407, ...
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None                                                            # trace_info : t_4705, t_4716, t_4999, t_5012, t_5408, ...


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

    def prefix_product(a: List[int], init=1) -> List[int]:                     # trace_info : t_4463, t_4736, t_5034, t_5479, t_5820, ...
        r = [init]                                                             # trace_info : t_4469, t_4487, t_4499, t_4519, t_4539, ...
        for v in a:                                                            # trace_info : t_4470, t_4473, t_4476, t_4479, t_4482, ...
            init = init * v                                                    # trace_info : t_4471, t_4474, t_4477, t_4480, t_4489, ...
            r.append(init)                                                     # trace_info : t_4472, t_4475, t_4478, t_4481, t_4490, ...
        return r                                                               # trace_info : t_4483, t_4492, t_4510, t_4524, t_4544, ...

    def inner_product(a: List[int], b: List[int]) -> int:                      # trace_info : t_4464, t_4737, t_5035, t_5480, t_5821, ...
        return sum([x * y for x, y in zip(a, b)])                              # trace_info : t_4530, t_4532, t_4550, t_4552, t_4570, ...

    def decompose(index, shape, stride=None):                                  # trace_info : t_4465, t_4738, t_5036, t_5481, t_5822, ...
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:                                                     # trace_info : t_4497, t_4517, t_4537, t_4557, t_4577, ...
            stride = prefix_product(shape)                                     # trace_info : t_4498, t_4518, t_4538, t_4558, t_4578, ...
        idx = [(index // d) % s for s, d in zip(shape, stride)]                # trace_info : t_4511, t_4525, t_4545, t_4565, t_4585, ...
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index            # trace_info : t_4512, t_4526, t_4546, t_4566, t_4586, ...
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx                                                             # trace_info : t_4513, t_4527, t_4547, t_4567, t_4587, ...

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]               # trace_info : t_4466, t_4739, t_5037, t_5482, t_5823, ...
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]         # trace_info : t_4467, t_4740, t_5038, t_5483, t_5824, ...

    global_stride = prefix_product(parallel_size)                              # trace_info : t_4468, t_4741, t_5039, t_5484, t_5825, ...
    masked_stride = [d for d, m in zip(global_stride, mask) if m]              # trace_info : t_4484, t_4757, t_5055, t_5500, t_5841, ...
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]        # trace_info : t_4485, t_4758, t_5056, t_5501, t_5842, ...

    group_size = prefix_product(masked_shape)[-1]                              # trace_info : t_4486, t_4759, t_5057, t_5502, t_5843, ...
    num_of_group = world_size // group_size                                    # trace_info : t_4493, t_4769, t_5064, t_5512, t_5850, ...

    ranks = []                                                                 # trace_info : t_4494, t_4770, t_5065, t_5513, t_5851, ...
    for group_index in range(num_of_group):                                    # trace_info : t_4495, t_4597, t_4699, t_4771, t_4882, ...
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)          # trace_info : t_4496, t_4598, t_4772, t_4883, t_5067, ...
        rank = []                                                              # trace_info : t_4514, t_4616, t_4787, t_4898, t_5085, ...
        for rank_in_group in range(group_size):                                # trace_info : t_4515, t_4535, t_4555, t_4575, t_4595, ...
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)       # trace_info : t_4516, t_4536, t_4556, t_4576, t_4618, ...
            rank.append(                                                       # trace_info : t_4528, t_4534, t_4548, t_4554, t_4568, ...
                inner_product(decomposed_rank_idx, masked_stride)              # trace_info : t_4529, t_4533, t_4549, t_4553, t_4569, ...
                + inner_product(decomposed_group_idx, unmasked_stride)         # trace_info : t_4531, t_4551, t_4571, t_4591, t_4633, ...
            )
        ranks.append(rank)                                                     # trace_info : t_4596, t_4698, t_4881, t_4992, t_5107, ...
    return ranks                                                               # trace_info : t_4700, t_4994, t_5403, t_5775, t_6101, ...


class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp                                                           # trace_info : t_4388
        self.ep = ep                                                           # trace_info : t_4389
        self.dp = dp                                                           # trace_info : t_4390
        self.pp = pp                                                           # trace_info : t_4391
        self.cp = cp                                                           # trace_info : t_4392
        self.world_size = tp * dp * pp * cp                                    # trace_info : t_4393

        self.name_to_size = {                                                  # trace_info : t_4399
            "tp": self.tp,                                                     # trace_info : t_4394
            "pp": self.pp,                                                     # trace_info : t_4395
            "dp": self.dp,                                                     # trace_info : t_4396
            "ep": self.ep,                                                     # trace_info : t_4397
            "cp": self.cp,                                                     # trace_info : t_4398
        }
        self.order = order                                                     # trace_info : t_4400
        order = order.lower()                                                  # trace_info : t_4401

        if 'ep' in order:                                                      # trace_info : t_4402
            if 'ep-dp' not in order and 'dp-ep' not in order:                  # trace_info : t_4403
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():                                  # trace_info : t_4404, t_4407, t_4410, t_4413, t_4416, ...
            if name not in order and self.name_to_size[name] != 1:             # trace_info : t_4405, t_4408, t_4411, t_4414, t_4417
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:                                            # trace_info : t_4406, t_4409, t_4412, t_4415, t_4418
                order = order + '-' + name

        self.order_w_ep = order                                                # trace_info : t_4420
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])# trace_info : t_4421
        self.ordered_size_wo_ep = []                                           # trace_info : t_4422
        self.ordered_size_w_ep = []                                            # trace_info : t_4423

        for token in order.split('-'):                                         # trace_info : t_4424, t_4429, t_4434, t_4438, t_4442, ...
            if token == 'dp':                                                  # trace_info : t_4425, t_4430, t_4435, t_4439, t_4443
                self.ordered_size_w_ep.append(self.dp // self.ep)              # trace_info : t_4440
                self.ordered_size_wo_ep.append(self.dp)                        # trace_info : t_4441
            elif token == 'ep':                                                # trace_info : t_4426, t_4431, t_4436, t_4444
                self.ordered_size_w_ep.append(self.ep)                         # trace_info : t_4437
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])        # trace_info : t_4427, t_4432, t_4445
                self.ordered_size_wo_ep.append(self.name_to_size[token])       # trace_info : t_4428, t_4433, t_4446

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')                                       # trace_info : t_4455, t_4726, t_5026, t_5469, t_5812, ...
        token = token.split('-')                                               # trace_info : t_4456, t_4727, t_5027, t_5470, t_5813, ...
        mask = [False] * len(ordered_token)                                    # trace_info : t_4457, t_4728, t_5028, t_5471, t_5814, ...
        for t in token:                                                        # trace_info : t_4458, t_4460, t_4729, t_4731, t_4733, ...
            mask[ordered_token.index(t)] = True                                # trace_info : t_4459, t_4730, t_4732, t_5030, t_5473, ...
        return mask                                                            # trace_info : t_4461, t_4734, t_5032, t_5477, t_5818, ...

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
        if independent_ep:                                                     # trace_info : t_4451, t_4722, t_5022, t_5465, t_5808, ...
            parallel_size = self.ordered_size_w_ep                             # trace_info : t_7294, t_7651, t_8119
            order = self.order_w_ep                                            # trace_info : t_7295, t_7652, t_8120
        else:
            parallel_size = self.ordered_size_wo_ep                            # trace_info : t_4452, t_4723, t_5023, t_5466, t_5809, ...
            order = self.order_wo_ep                                           # trace_info : t_4453, t_4724, t_5024, t_5467, t_5810, ...
        mask = self.get_mask(order, token)                                     # trace_info : t_4454, t_4725, t_5025, t_5468, t_5811, ...
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)# trace_info : t_4462, t_4735, t_5033, t_5478, t_5819, ...
        return ranks                                                           # trace_info : t_4701, t_4995, t_5404, t_5776, t_6102, ...


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
    assert torch.distributed.is_initialized()                                  # trace_info : t_4361
    world_size: int = torch.distributed.get_world_size()                       # trace_info : t_4362

    if (
        world_size                                                             # trace_info : t_4363, t_4365, t_4367
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)# trace_info : t_4364
        != 0                                                                   # trace_info : t_4366
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (                                  # trace_info : t_4368, t_4370
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size# trace_info : t_4369
    )

    if data_parallel_size % expert_model_parallel_size != 0:                   # trace_info : t_4371
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:           # trace_info : t_4372
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size# trace_info : t_4373
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size# trace_info : t_4374

    if virtual_pipeline_model_parallel_size is not None:                       # trace_info : t_4375
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:                         # trace_info : t_4376
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()                                        # trace_info : t_4377

    nccl_comm_cfgs = {}                                                        # trace_info : t_4378
    if nccl_communicator_config_path is not None:                              # trace_info : t_4379
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    rank_generator = RankGenerator(                                            # trace_info : t_4380, t_4387
        tp=tensor_model_parallel_size,                                         # trace_info : t_4381
        ep=expert_model_parallel_size,                                         # trace_info : t_4382
        dp=data_parallel_size,                                                 # trace_info : t_4383
        pp=pipeline_model_parallel_size,                                       # trace_info : t_4384
        cp=context_parallel_size,                                              # trace_info : t_4385
        order=order,                                                           # trace_info : t_4386
    )
    timeout = timedelta(minutes=distributed_timeout_minutes)                   # trace_info : t_4448

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'# trace_info : t_4449

    for ranks in rank_generator.get_ranks('dp'):                               # trace_info : t_4450, t_4712, t_4720
        group = torch.distributed.new_group(                                   # trace_info : t_4702, t_4706, t_4713, t_4717
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)# trace_info : t_4703, t_4714
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")# trace_info : t_4707, t_4718
        if rank in ranks:                                                      # trace_info : t_4708, t_4719
            _DATA_PARALLEL_GROUP = group                                       # trace_info : t_4709
            _DATA_PARALLEL_GROUP_GLOO = group_gloo                             # trace_info : t_4710
            _DATA_PARALLEL_GLOBAL_RANKS = ranks                                # trace_info : t_4711
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):                    # trace_info : t_4721, t_5008, t_5018
        group_with_cp = torch.distributed.new_group(                           # trace_info : t_4996, t_5000, t_5009, t_5013
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)# trace_info : t_4997, t_5010
        )
        group_with_cp_gloo = torch.distributed.new_group(                      # trace_info : t_5001, t_5003, t_5014, t_5016
            ranks_with_cp, timeout=timeout, backend="gloo"                     # trace_info : t_5002, t_5015
        )
        if rank in ranks_with_cp:                                              # trace_info : t_5004, t_5017
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp                       # trace_info : t_5005
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo             # trace_info : t_5006
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp                # trace_info : t_5007

    # Apply SHARP to DP process groups
    if use_sharp:                                                              # trace_info : t_5019
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
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'# trace_info : t_5020
    for ranks in rank_generator.get_ranks('cp'):                               # trace_info : t_5021, t_5413, t_5420, t_5427, t_5434, ...
        group = torch.distributed.new_group(                                   # trace_info : t_5405, t_5409, t_5414, t_5418, t_5421, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)# trace_info : t_5406, t_5415, t_5422, t_5429, t_5436, ...
        )
        if rank in ranks:                                                      # trace_info : t_5410, t_5419, t_5426, t_5433, t_5440, ...
            _CONTEXT_PARALLEL_GROUP = group                                    # trace_info : t_5411
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks                             # trace_info : t_5412

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'# trace_info : t_5463
    for ranks in rank_generator.get_ranks('tp-pp'):                            # trace_info : t_5464, t_5784, t_5791, t_5798, t_5805
        group = torch.distributed.new_group(                                   # trace_info : t_5777, t_5781, t_5785, t_5789, t_5792, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)# trace_info : t_5778, t_5786, t_5793, t_5800
        )
        if rank in ranks:                                                      # trace_info : t_5782, t_5790, t_5797, t_5804
            _MODEL_PARALLEL_GROUP = group                                      # trace_info : t_5783

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_5806
    ), 'tensor model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp'):                               # trace_info : t_5807, t_6111, t_6118, t_6125, t_6132
        group = torch.distributed.new_group(                                   # trace_info : t_6103, t_6107, t_6112, t_6116, t_6119, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)# trace_info : t_6104, t_6113, t_6120, t_6127
        )
        if rank in ranks:                                                      # trace_info : t_6108, t_6117, t_6124, t_6131
            _TENSOR_MODEL_PARALLEL_GROUP = group                               # trace_info : t_6109
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks                        # trace_info : t_6110

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None                                 # trace_info : t_6133
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'  # trace_info : t_6134
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'# trace_info : t_6135
    for ranks in rank_generator.get_ranks('pp'):                               # trace_info : t_6136, t_6551, t_6577, t_6603, t_6629, ...
        group = torch.distributed.new_group(                                   # trace_info : t_6520, t_6524, t_6552, t_6556, t_6578, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)# trace_info : t_6521, t_6553, t_6579, t_6605, t_6631, ...
        )
        if rank in ranks:                                                      # trace_info : t_6525, t_6557, t_6583, t_6609, t_6635, ...
            _PIPELINE_MODEL_PARALLEL_GROUP = group                             # trace_info : t_6526
            _PIPELINE_GLOBAL_RANKS = ranks                                     # trace_info : t_6527
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:                                                     # trace_info : t_6528, t_6558, t_6584, t_6610, t_6636, ...
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
            embedding_ranks = ranks                                            # trace_info : t_6529, t_6559, t_6585, t_6611, t_6637, ...
            position_embedding_ranks = ranks                                   # trace_info : t_6530, t_6560, t_6586, t_6612, t_6638, ...

        group = torch.distributed.new_group(                                   # trace_info : t_6531, t_6535, t_6561, t_6565, t_6587, ...
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)# trace_info : t_6532, t_6562, t_6588, t_6614, t_6640, ...
        )
        if rank in embedding_ranks:                                            # trace_info : t_6536, t_6566, t_6592, t_6618, t_6644, ...
            _EMBEDDING_GROUP = group                                           # trace_info : t_6537
        if rank in ranks:                                                      # trace_info : t_6538, t_6567, t_6593, t_6619, t_6645, ...
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks                          # trace_info : t_6539

        group = torch.distributed.new_group(                                   # trace_info : t_6540, t_6546, t_6568, t_6574, t_6594, ...
            position_embedding_ranks,                                          # trace_info : t_6541, t_6569, t_6595, t_6621, t_6647, ...
            timeout=timeout,                                                   # trace_info : t_6542, t_6570, t_6596, t_6622, t_6648, ...
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),               # trace_info : t_6543, t_6571, t_6597, t_6623, t_6649, ...
        )
        if rank in position_embedding_ranks:                                   # trace_info : t_6547, t_6575, t_6601, t_6627, t_6653, ...
            _POSITION_EMBEDDING_GROUP = group                                  # trace_info : t_6548
        if rank in ranks:                                                      # trace_info : t_6549, t_6576, t_6602, t_6628, t_6654, ...
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks        # trace_info : t_6550

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None                                # trace_info : t_6734
    ), 'Tensor + data parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-dp-cp'):                         # trace_info : t_6735, t_7024
        group = torch.distributed.new_group(                                   # trace_info : t_7017, t_7021
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)# trace_info : t_7018
        )
        if rank in ranks:                                                      # trace_info : t_7022
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group                    # trace_info : t_7023
    for ranks in rank_generator.get_ranks('tp-dp'):                            # trace_info : t_7025, t_7288
        group = torch.distributed.new_group(                                   # trace_info : t_7281, t_7285
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)# trace_info : t_7282
        )
        if rank in ranks:                                                      # trace_info : t_7286
            _TENSOR_AND_DATA_PARALLEL_GROUP = group                            # trace_info : t_7287

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'# trace_info : t_7289
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None                              # trace_info : t_7290
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None                             # trace_info : t_7291
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO

    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):       # trace_info : t_7292, t_7627, t_7634, t_7641, t_7648
        group = torch.distributed.new_group(                                   # trace_info : t_7620, t_7624, t_7628, t_7632, t_7635, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)# trace_info : t_7621, t_7629, t_7636, t_7643
        )
        if rank in ranks:                                                      # trace_info : t_7625, t_7633, t_7640, t_7647
            _TENSOR_AND_EXPERT_PARALLEL_GROUP = group                          # trace_info : t_7626

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):          # trace_info : t_7649, t_8067, t_8074, t_8081, t_8088, ...
        group = torch.distributed.new_group(                                   # trace_info : t_8060, t_8064, t_8068, t_8072, t_8075, ...
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)          # trace_info : t_8061, t_8069, t_8076, t_8083, t_8090, ...
        )
        if rank in ranks:                                                      # trace_info : t_8065, t_8073, t_8080, t_8087, t_8094, ...
            _EXPERT_MODEL_PARALLEL_GROUP = group                               # trace_info : t_8066

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):          # trace_info : t_8117, t_8387, t_8395
        group = torch.distributed.new_group(                                   # trace_info : t_8378, t_8382, t_8388, t_8392
            ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)# trace_info : t_8379, t_8389
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")        # trace_info : t_8383, t_8393
        if rank in ranks:                                                      # trace_info : t_8384, t_8394
            _DATA_MODULO_EXPERT_PARALLEL_GROUP = group                         # trace_info : t_8385
            _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo               # trace_info : t_8386

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()                                                # trace_info : t_8396


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
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_4348
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False                                                           # trace_info : t_4349
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'# trace_info : t_15069, t_20122, t_20646, t_23759, t_24283, ...
    return _MODEL_PARALLEL_GROUP                                               # trace_info : t_15070, t_20123, t_20647, t_23760, t_24284, ...


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_8406, t_8439, t_8507, t_8512, t_9476, ...
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None                           # trace_info : t_8407, t_8440, t_8508, t_8513, t_9477, ...
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP                                        # trace_info : t_8408, t_8441, t_8509, t_8514, t_9478, ...


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None                             # trace_info : t_8416, t_8427, t_8827, t_8836, t_8847, ...
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP                                      # trace_info : t_8417, t_8428, t_8828, t_8837, t_8848, ...


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_12143, t_12208, t_12222, t_13895, t_15076, ...
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None                           # trace_info : t_12209, t_13896, t_15077, t_25015, t_25105, ...
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP                                    # trace_info : t_12210, t_13897, t_15078, t_25016, t_25106, ...
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'# trace_info : t_12144, t_12223, t_17085, t_17093, t_17135, ...
        return _DATA_PARALLEL_GROUP                                            # trace_info : t_12145, t_12224, t_17086, t_17094, t_17136, ...


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_15080
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None                      # trace_info : t_15081
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO                               # trace_info : t_15082
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
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None                          # trace_info : t_8501
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP                                   # trace_info : t_8502


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None                         # trace_info : t_12211, t_25038, t_92631
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP                                  # trace_info : t_12212, t_25039, t_92632


def get_data_modulo_expert_parallel_group_gloo():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is not None
    ), 'data modulo expert parallel group-gloo is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO


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
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:                      # trace_info : t_8404, t_8505, t_9474, t_9749, t_9800, ...
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())# trace_info : t_8405, t_8506, t_9475, t_9750, t_9801, ...


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:                    # trace_info : t_8414, t_8825, t_8849, t_9588, t_9654, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())# trace_info : t_8415, t_8826, t_8850, t_9589, t_9655, ...


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
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:                            # trace_info : t_8437, t_8510, t_9481, t_9897, t_10038, ...
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group()) # trace_info : t_8438, t_8511, t_9482, t_9898, t_10039, ...


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:                          # trace_info : t_8425, t_8834, t_8845, t_9649, t_10651, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())# trace_info : t_8426, t_8835, t_8846, t_9650, t_10652, ...


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_8830, t_16219, t_16418, t_16633, t_16848, ...
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None       # trace_info : t_8831, t_16220, t_16419, t_16634, t_16849, ...
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0                             # trace_info : t_8833, t_16222, t_16421, t_16636, t_16851, ...


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_8839, t_19533, t_19568, t_21073, t_23170, ...
        virtual_pipeline_model_parallel_world_size = (                         # trace_info : t_8842, t_19536, t_19571, t_23173, t_23208, ...
            get_virtual_pipeline_model_parallel_world_size()                   # trace_info : t_8840, t_19534, t_19569, t_23171, t_23206, ...
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (# trace_info : t_8843, t_19537, t_19572, t_23174, t_23209, ...
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)# trace_info : t_8844, t_19538, t_19573, t_21074, t_23175, ...


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()                                        # trace_info : t_19813, t_23450, t_91057
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:                                                         # trace_info : t_19814, t_23451, t_91058
        return rank in _EMBEDDING_GLOBAL_RANKS                                 # trace_info : t_19815, t_23452, t_91059
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
    rank = torch.distributed.get_rank()                                        # trace_info : t_19824, t_23461, t_91068
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS                            # trace_info : t_19825, t_23462, t_91069


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
    if get_pipeline_model_parallel_world_size() == 1:                          # trace_info : t_19638, t_23275, t_90882
        return True                                                            # trace_info : t_19643, t_23280, t_90887
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
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK                               # trace_info : t_16215


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE                         # trace_info : t_8832, t_8841, t_9594, t_9660, t_10662, ...


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None                        # trace_info : t_18244, t_18252, t_18260, t_18268, t_18276, ...
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]                              # trace_info : t_18245, t_18253, t_18261, t_18269, t_18277, ...


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
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_12219, t_17089, t_17139, t_17186, t_21099, ...
        return torch.distributed.get_world_size(                               # trace_info : t_12220, t_12225, t_17090, t_17095, t_17140, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_12221, t_17091, t_17141, t_17188, t_21101, ...
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_12140, t_13892, t_17081, t_17131, t_17178, ...
        return torch.distributed.get_rank(                                     # trace_info : t_12141, t_12146, t_13893, t_13898, t_17082, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_12142, t_13894, t_17083, t_17133, t_17180, ...
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
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:                                        # trace_info : t_8497
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_8498
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(          # trace_info : t_8499, t_8503
            group=get_tensor_and_expert_parallel_group()                       # trace_info : t_8500
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()# trace_info : t_8504
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_25036, t_92629
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())# trace_info : t_25037, t_92630
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
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'# trace_info : t_8397
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()                               # trace_info : t_8398


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'# trace_info : t_18557, t_19052, t_22196, t_22689, t_89803, ...
    return _GLOBAL_MEMORY_BUFFER                                               # trace_info : t_18558, t_19053, t_22197, t_22690, t_89804, ...


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
