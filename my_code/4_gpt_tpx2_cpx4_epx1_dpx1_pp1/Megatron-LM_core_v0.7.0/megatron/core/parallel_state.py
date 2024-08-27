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
    if pg_name in nccl_comm_cfgs:                                              # trace_info : t_4835, t_4846, t_4854, t_4862, t_4870, ...
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None                                                            # trace_info : t_4836, t_4847, t_4855, t_4863, t_4871, ...


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

    def prefix_product(a: List[int], init=1) -> List[int]:                     # trace_info : t_4462, t_4915, t_5213, t_5484, t_5825, ...
        r = [init]                                                             # trace_info : t_4468, t_4486, t_4498, t_4518, t_4540, ...
        for v in a:                                                            # trace_info : t_4469, t_4472, t_4475, t_4478, t_4481, ...
            init = init * v                                                    # trace_info : t_4470, t_4473, t_4476, t_4479, t_4488, ...
            r.append(init)                                                     # trace_info : t_4471, t_4474, t_4477, t_4480, t_4489, ...
        return r                                                               # trace_info : t_4482, t_4491, t_4509, t_4523, t_4551, ...

    def inner_product(a: List[int], b: List[int]) -> int:                      # trace_info : t_4463, t_4916, t_5214, t_5485, t_5826, ...
        return sum([x * y for x, y in zip(a, b)])                              # trace_info : t_4529, t_4531, t_4571, t_4573, t_4613, ...

    def decompose(index, shape, stride=None):                                  # trace_info : t_4464, t_4917, t_5215, t_5486, t_5827, ...
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:                                                     # trace_info : t_4496, t_4516, t_4538, t_4558, t_4580, ...
            stride = prefix_product(shape)                                     # trace_info : t_4497, t_4517, t_4539, t_4559, t_4581, ...
        idx = [(index // d) % s for s, d in zip(shape, stride)]                # trace_info : t_4510, t_4524, t_4552, t_4566, t_4594, ...
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index            # trace_info : t_4511, t_4525, t_4553, t_4567, t_4595, ...
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx                                                             # trace_info : t_4512, t_4526, t_4554, t_4568, t_4596, ...

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]               # trace_info : t_4465, t_4918, t_5216, t_5487, t_5828, ...
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]         # trace_info : t_4466, t_4919, t_5217, t_5488, t_5829, ...

    global_stride = prefix_product(parallel_size)                              # trace_info : t_4467, t_4920, t_5218, t_5489, t_5830, ...
    masked_stride = [d for d, m in zip(global_stride, mask) if m]              # trace_info : t_4483, t_4936, t_5234, t_5505, t_5846, ...
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]        # trace_info : t_4484, t_4937, t_5235, t_5506, t_5847, ...

    group_size = prefix_product(masked_shape)[-1]                              # trace_info : t_4485, t_4938, t_5236, t_5507, t_5848, ...
    num_of_group = world_size // group_size                                    # trace_info : t_4492, t_4948, t_5243, t_5517, t_5855, ...

    ranks = []                                                                 # trace_info : t_4493, t_4949, t_5244, t_5518, t_5856, ...
    for group_index in range(num_of_group):                                    # trace_info : t_4494, t_4536, t_4578, t_4620, t_4662, ...
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)          # trace_info : t_4495, t_4537, t_4579, t_4621, t_4663, ...
        rank = []                                                              # trace_info : t_4513, t_4555, t_4597, t_4639, t_4681, ...
        for rank_in_group in range(group_size):                                # trace_info : t_4514, t_4534, t_4556, t_4576, t_4598, ...
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)       # trace_info : t_4515, t_4557, t_4599, t_4641, t_4683, ...
            rank.append(                                                       # trace_info : t_4527, t_4533, t_4569, t_4575, t_4611, ...
                inner_product(decomposed_rank_idx, masked_stride)              # trace_info : t_4528, t_4532, t_4570, t_4574, t_4612, ...
                + inner_product(decomposed_group_idx, unmasked_stride)         # trace_info : t_4530, t_4572, t_4614, t_4656, t_4698, ...
            )
        ranks.append(rank)                                                     # trace_info : t_4535, t_4577, t_4619, t_4661, t_4703, ...
    return ranks                                                               # trace_info : t_4831, t_5173, t_5450, t_5780, t_6106, ...


class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp                                                           # trace_info : t_4387
        self.ep = ep                                                           # trace_info : t_4388
        self.dp = dp                                                           # trace_info : t_4389
        self.pp = pp                                                           # trace_info : t_4390
        self.cp = cp                                                           # trace_info : t_4391
        self.world_size = tp * dp * pp * cp                                    # trace_info : t_4392

        self.name_to_size = {                                                  # trace_info : t_4398
            "tp": self.tp,                                                     # trace_info : t_4393
            "pp": self.pp,                                                     # trace_info : t_4394
            "dp": self.dp,                                                     # trace_info : t_4395
            "ep": self.ep,                                                     # trace_info : t_4396
            "cp": self.cp,                                                     # trace_info : t_4397
        }
        self.order = order                                                     # trace_info : t_4399
        order = order.lower()                                                  # trace_info : t_4400

        if 'ep' in order:                                                      # trace_info : t_4401
            if 'ep-dp' not in order and 'dp-ep' not in order:                  # trace_info : t_4402
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():                                  # trace_info : t_4403, t_4406, t_4409, t_4412, t_4415, ...
            if name not in order and self.name_to_size[name] != 1:             # trace_info : t_4404, t_4407, t_4410, t_4413, t_4416
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:                                            # trace_info : t_4405, t_4408, t_4411, t_4414, t_4417
                order = order + '-' + name

        self.order_w_ep = order                                                # trace_info : t_4419
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])# trace_info : t_4420
        self.ordered_size_wo_ep = []                                           # trace_info : t_4421
        self.ordered_size_w_ep = []                                            # trace_info : t_4422

        for token in order.split('-'):                                         # trace_info : t_4423, t_4428, t_4433, t_4437, t_4441, ...
            if token == 'dp':                                                  # trace_info : t_4424, t_4429, t_4434, t_4438, t_4442
                self.ordered_size_w_ep.append(self.dp // self.ep)              # trace_info : t_4439
                self.ordered_size_wo_ep.append(self.dp)                        # trace_info : t_4440
            elif token == 'ep':                                                # trace_info : t_4425, t_4430, t_4435, t_4443
                self.ordered_size_w_ep.append(self.ep)                         # trace_info : t_4436
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])        # trace_info : t_4426, t_4431, t_4444
                self.ordered_size_wo_ep.append(self.name_to_size[token])       # trace_info : t_4427, t_4432, t_4445

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')                                       # trace_info : t_4454, t_4905, t_5205, t_5474, t_5817, ...
        token = token.split('-')                                               # trace_info : t_4455, t_4906, t_5206, t_5475, t_5818, ...
        mask = [False] * len(ordered_token)                                    # trace_info : t_4456, t_4907, t_5207, t_5476, t_5819, ...
        for t in token:                                                        # trace_info : t_4457, t_4459, t_4908, t_4910, t_4912, ...
            mask[ordered_token.index(t)] = True                                # trace_info : t_4458, t_4909, t_4911, t_5209, t_5478, ...
        return mask                                                            # trace_info : t_4460, t_4913, t_5211, t_5482, t_5823, ...

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
        if independent_ep:                                                     # trace_info : t_4450, t_4901, t_5201, t_5470, t_5813, ...
            parallel_size = self.ordered_size_w_ep                             # trace_info : t_7377, t_7734, t_8202
            order = self.order_w_ep                                            # trace_info : t_7378, t_7735, t_8203
        else:
            parallel_size = self.ordered_size_wo_ep                            # trace_info : t_4451, t_4902, t_5202, t_5471, t_5814, ...
            order = self.order_wo_ep                                           # trace_info : t_4452, t_4903, t_5203, t_5472, t_5815, ...
        mask = self.get_mask(order, token)                                     # trace_info : t_4453, t_4904, t_5204, t_5473, t_5816, ...
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)# trace_info : t_4461, t_4914, t_5212, t_5483, t_5824, ...
        return ranks                                                           # trace_info : t_4832, t_5174, t_5451, t_5781, t_6107, ...


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
    assert torch.distributed.is_initialized()                                  # trace_info : t_4360
    world_size: int = torch.distributed.get_world_size()                       # trace_info : t_4361

    if (
        world_size                                                             # trace_info : t_4362, t_4364, t_4366
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)# trace_info : t_4363
        != 0                                                                   # trace_info : t_4365
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (                                  # trace_info : t_4367, t_4369
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size# trace_info : t_4368
    )

    if data_parallel_size % expert_model_parallel_size != 0:                   # trace_info : t_4370
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:           # trace_info : t_4371
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size# trace_info : t_4372
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size# trace_info : t_4373

    if virtual_pipeline_model_parallel_size is not None:                       # trace_info : t_4374
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:                         # trace_info : t_4375
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()                                        # trace_info : t_4376

    nccl_comm_cfgs = {}                                                        # trace_info : t_4377
    if nccl_communicator_config_path is not None:                              # trace_info : t_4378
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    rank_generator = RankGenerator(                                            # trace_info : t_4379, t_4386
        tp=tensor_model_parallel_size,                                         # trace_info : t_4380
        ep=expert_model_parallel_size,                                         # trace_info : t_4381
        dp=data_parallel_size,                                                 # trace_info : t_4382
        pp=pipeline_model_parallel_size,                                       # trace_info : t_4383
        cp=context_parallel_size,                                              # trace_info : t_4384
        order=order,                                                           # trace_info : t_4385
    )
    timeout = timedelta(minutes=distributed_timeout_minutes)                   # trace_info : t_4447

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'# trace_info : t_4448

    for ranks in rank_generator.get_ranks('dp'):                               # trace_info : t_4449, t_4843, t_4851, t_4859, t_4867, ...
        group = torch.distributed.new_group(                                   # trace_info : t_4833, t_4837, t_4844, t_4848, t_4852, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)# trace_info : t_4834, t_4845, t_4853, t_4861, t_4869, ...
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")# trace_info : t_4838, t_4849, t_4857, t_4865, t_4873, ...
        if rank in ranks:                                                      # trace_info : t_4839, t_4850, t_4858, t_4866, t_4874, ...
            _DATA_PARALLEL_GROUP = group                                       # trace_info : t_4840
            _DATA_PARALLEL_GROUP_GLOO = group_gloo                             # trace_info : t_4841
            _DATA_PARALLEL_GLOBAL_RANKS = ranks                                # trace_info : t_4842
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):                    # trace_info : t_4900, t_5187, t_5197
        group_with_cp = torch.distributed.new_group(                           # trace_info : t_5175, t_5179, t_5188, t_5192
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)# trace_info : t_5176, t_5189
        )
        group_with_cp_gloo = torch.distributed.new_group(                      # trace_info : t_5180, t_5182, t_5193, t_5195
            ranks_with_cp, timeout=timeout, backend="gloo"                     # trace_info : t_5181, t_5194
        )
        if rank in ranks_with_cp:                                              # trace_info : t_5183, t_5196
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp                       # trace_info : t_5184
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo             # trace_info : t_5185
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp                # trace_info : t_5186

    # Apply SHARP to DP process groups
    if use_sharp:                                                              # trace_info : t_5198
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
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'# trace_info : t_5199
    for ranks in rank_generator.get_ranks('cp'):                               # trace_info : t_5200, t_5460, t_5467
        group = torch.distributed.new_group(                                   # trace_info : t_5452, t_5456, t_5461, t_5465
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)# trace_info : t_5453, t_5462
        )
        if rank in ranks:                                                      # trace_info : t_5457, t_5466
            _CONTEXT_PARALLEL_GROUP = group                                    # trace_info : t_5458
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks                             # trace_info : t_5459

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'# trace_info : t_5468
    for ranks in rank_generator.get_ranks('tp-pp'):                            # trace_info : t_5469, t_5789, t_5796, t_5803, t_5810
        group = torch.distributed.new_group(                                   # trace_info : t_5782, t_5786, t_5790, t_5794, t_5797, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)# trace_info : t_5783, t_5791, t_5798, t_5805
        )
        if rank in ranks:                                                      # trace_info : t_5787, t_5795, t_5802, t_5809
            _MODEL_PARALLEL_GROUP = group                                      # trace_info : t_5788

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_5811
    ), 'tensor model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp'):                               # trace_info : t_5812, t_6116, t_6123, t_6130, t_6137
        group = torch.distributed.new_group(                                   # trace_info : t_6108, t_6112, t_6117, t_6121, t_6124, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)# trace_info : t_6109, t_6118, t_6125, t_6132
        )
        if rank in ranks:                                                      # trace_info : t_6113, t_6122, t_6129, t_6136
            _TENSOR_MODEL_PARALLEL_GROUP = group                               # trace_info : t_6114
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks                        # trace_info : t_6115

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None                                 # trace_info : t_6138
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'  # trace_info : t_6139
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'# trace_info : t_6140
    for ranks in rank_generator.get_ranks('pp'):                               # trace_info : t_6141, t_6556, t_6582, t_6608, t_6634, ...
        group = torch.distributed.new_group(                                   # trace_info : t_6525, t_6529, t_6557, t_6561, t_6583, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)# trace_info : t_6526, t_6558, t_6584, t_6610, t_6636, ...
        )
        if rank in ranks:                                                      # trace_info : t_6530, t_6562, t_6588, t_6614, t_6640, ...
            _PIPELINE_MODEL_PARALLEL_GROUP = group                             # trace_info : t_6531
            _PIPELINE_GLOBAL_RANKS = ranks                                     # trace_info : t_6532
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:                                                     # trace_info : t_6533, t_6563, t_6589, t_6615, t_6641, ...
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
            embedding_ranks = ranks                                            # trace_info : t_6534, t_6564, t_6590, t_6616, t_6642, ...
            position_embedding_ranks = ranks                                   # trace_info : t_6535, t_6565, t_6591, t_6617, t_6643, ...

        group = torch.distributed.new_group(                                   # trace_info : t_6536, t_6540, t_6566, t_6570, t_6592, ...
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)# trace_info : t_6537, t_6567, t_6593, t_6619, t_6645, ...
        )
        if rank in embedding_ranks:                                            # trace_info : t_6541, t_6571, t_6597, t_6623, t_6649, ...
            _EMBEDDING_GROUP = group                                           # trace_info : t_6542
        if rank in ranks:                                                      # trace_info : t_6543, t_6572, t_6598, t_6624, t_6650, ...
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks                          # trace_info : t_6544

        group = torch.distributed.new_group(                                   # trace_info : t_6545, t_6551, t_6573, t_6579, t_6599, ...
            position_embedding_ranks,                                          # trace_info : t_6546, t_6574, t_6600, t_6626, t_6652, ...
            timeout=timeout,                                                   # trace_info : t_6547, t_6575, t_6601, t_6627, t_6653, ...
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),               # trace_info : t_6548, t_6576, t_6602, t_6628, t_6654, ...
        )
        if rank in position_embedding_ranks:                                   # trace_info : t_6552, t_6580, t_6606, t_6632, t_6658, ...
            _POSITION_EMBEDDING_GROUP = group                                  # trace_info : t_6553
        if rank in ranks:                                                      # trace_info : t_6554, t_6581, t_6607, t_6633, t_6659, ...
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks        # trace_info : t_6555

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None                                # trace_info : t_6739
    ), 'Tensor + data parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-dp-cp'):                         # trace_info : t_6740, t_7029
        group = torch.distributed.new_group(                                   # trace_info : t_7022, t_7026
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)# trace_info : t_7023
        )
        if rank in ranks:                                                      # trace_info : t_7027
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group                    # trace_info : t_7028
    for ranks in rank_generator.get_ranks('tp-dp'):                            # trace_info : t_7030, t_7350, t_7357, t_7364, t_7371
        group = torch.distributed.new_group(                                   # trace_info : t_7343, t_7347, t_7351, t_7355, t_7358, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)# trace_info : t_7344, t_7352, t_7359, t_7366
        )
        if rank in ranks:                                                      # trace_info : t_7348, t_7356, t_7363, t_7370
            _TENSOR_AND_DATA_PARALLEL_GROUP = group                            # trace_info : t_7349

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'# trace_info : t_7372
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None                              # trace_info : t_7373
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None                             # trace_info : t_7374
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO

    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):       # trace_info : t_7375, t_7710, t_7717, t_7724, t_7731
        group = torch.distributed.new_group(                                   # trace_info : t_7703, t_7707, t_7711, t_7715, t_7718, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)# trace_info : t_7704, t_7712, t_7719, t_7726
        )
        if rank in ranks:                                                      # trace_info : t_7708, t_7716, t_7723, t_7730
            _TENSOR_AND_EXPERT_PARALLEL_GROUP = group                          # trace_info : t_7709

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):          # trace_info : t_7732, t_8150, t_8157, t_8164, t_8171, ...
        group = torch.distributed.new_group(                                   # trace_info : t_8143, t_8147, t_8151, t_8155, t_8158, ...
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)          # trace_info : t_8144, t_8152, t_8159, t_8166, t_8173, ...
        )
        if rank in ranks:                                                      # trace_info : t_8148, t_8156, t_8163, t_8170, t_8177, ...
            _EXPERT_MODEL_PARALLEL_GROUP = group                               # trace_info : t_8149

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):          # trace_info : t_8200, t_8620, t_8628, t_8636, t_8644, ...
        group = torch.distributed.new_group(                                   # trace_info : t_8611, t_8615, t_8621, t_8625, t_8629, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)# trace_info : t_8612, t_8622, t_8630, t_8638, t_8646, ...
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")        # trace_info : t_8616, t_8626, t_8634, t_8642, t_8650, ...
        if rank in ranks:                                                      # trace_info : t_8617, t_8627, t_8635, t_8643, t_8651, ...
            _DATA_MODULO_EXPERT_PARALLEL_GROUP = group                         # trace_info : t_8618
            _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo               # trace_info : t_8619

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()                                                # trace_info : t_8677


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
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_4347
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False                                                           # trace_info : t_4348
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'# trace_info : t_14899, t_19392, t_19916, t_22578, t_23102, ...
    return _MODEL_PARALLEL_GROUP                                               # trace_info : t_14900, t_19393, t_19917, t_22579, t_23103, ...


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_8687, t_8720, t_8788, t_8793, t_9753, ...
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None                           # trace_info : t_8688, t_8721, t_8789, t_8794, t_9754, ...
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP                                        # trace_info : t_8689, t_8722, t_8790, t_8795, t_9755, ...


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None                             # trace_info : t_8697, t_8708, t_9108, t_9117, t_9128, ...
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP                                      # trace_info : t_8698, t_8709, t_9109, t_9118, t_9129, ...


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_11973, t_12038, t_12052, t_13725, t_14906, ...
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None                           # trace_info : t_12039, t_13726, t_14907
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP                                    # trace_info : t_12040, t_13727, t_14908
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'# trace_info : t_11974, t_12053, t_16915, t_16923, t_16965, ...
        return _DATA_PARALLEL_GROUP                                            # trace_info : t_11975, t_12054, t_16916, t_16924, t_16966, ...


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_14910
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None                      # trace_info : t_14911
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO                               # trace_info : t_14912
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_10062, t_10840, t_18025, t_18860, t_21211, ...
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'# trace_info : t_18026, t_18861, t_21212, t_22047, t_24398, ...
    return _CONTEXT_PARALLEL_GROUP                                             # trace_info : t_10063, t_10841, t_18027, t_18862, t_21213, ...


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_10067, t_10845
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS                                      # trace_info : t_10068, t_10846


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
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None                          # trace_info : t_8782
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP                                   # trace_info : t_8783


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None                         # trace_info : t_12041
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP                                  # trace_info : t_12042


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
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:                      # trace_info : t_8685, t_8786, t_9751, t_10005, t_10784, ...
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())# trace_info : t_8686, t_8787, t_9752, t_10006, t_10785, ...


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:                    # trace_info : t_8695, t_9106, t_9130, t_9865, t_9931, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())# trace_info : t_8696, t_9107, t_9131, t_9866, t_9932, ...


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
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:                            # trace_info : t_8718, t_8791, t_9758, t_11519, t_11980, ...
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group()) # trace_info : t_8719, t_8792, t_9759, t_11520, t_11981, ...


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:                          # trace_info : t_8706, t_9115, t_9126, t_9926, t_10705, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())# trace_info : t_8707, t_9116, t_9127, t_9927, t_10706, ...


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_9111, t_16049, t_16248, t_16463, t_16678, ...
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None       # trace_info : t_9112, t_16050, t_16249, t_16464, t_16679, ...
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0                             # trace_info : t_9114, t_16052, t_16251, t_16466, t_16681, ...


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_9120, t_18799, t_18834, t_20343, t_21985, ...
        virtual_pipeline_model_parallel_world_size = (                         # trace_info : t_9123, t_18802, t_18837, t_21988, t_22023, ...
            get_virtual_pipeline_model_parallel_world_size()                   # trace_info : t_9121, t_18800, t_18835, t_21986, t_22021, ...
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (# trace_info : t_9124, t_18803, t_18838, t_21989, t_22024, ...
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)# trace_info : t_9125, t_18804, t_18839, t_20344, t_21990, ...


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()                                        # trace_info : t_19083, t_22269, t_25455
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:                                                         # trace_info : t_19084, t_22270, t_25456
        return rank in _EMBEDDING_GLOBAL_RANKS                                 # trace_info : t_19085, t_22271, t_25457
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
    rank = torch.distributed.get_rank()                                        # trace_info : t_19094, t_22280, t_25466
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS                            # trace_info : t_19095, t_22281, t_25467


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
    if get_pipeline_model_parallel_world_size() == 1:                          # trace_info : t_18908, t_22094, t_25280
        return True                                                            # trace_info : t_18913, t_22099, t_25285
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
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK                               # trace_info : t_16045


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE                         # trace_info : t_9113, t_9122, t_9871, t_9937, t_10716, ...


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None                        # trace_info : t_17977, t_17985, t_17993, t_18001, t_18009, ...
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]                              # trace_info : t_17978, t_17986, t_17994, t_18002, t_18010, ...


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
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_12049, t_16919, t_16969, t_17016, t_20369, ...
        return torch.distributed.get_world_size(                               # trace_info : t_12050, t_12055, t_16920, t_16925, t_16970, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_12051, t_16921, t_16971, t_17018, t_20371, ...
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_11970, t_13722, t_16911, t_16961, t_17008
        return torch.distributed.get_rank(                                     # trace_info : t_11971, t_11976, t_13723, t_13728, t_16912, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_11972, t_13724, t_16913, t_16963, t_17010
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_18023, t_21209, t_24395
        return torch.distributed.get_rank(group=get_context_parallel_group())  # trace_info : t_18024, t_21210, t_24396
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
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:                                        # trace_info : t_8778
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_8779
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(          # trace_info : t_8780, t_8784
            group=get_tensor_and_expert_parallel_group()                       # trace_info : t_8781
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()# trace_info : t_8785
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
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'# trace_info : t_8678
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()                               # trace_info : t_8679


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


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
