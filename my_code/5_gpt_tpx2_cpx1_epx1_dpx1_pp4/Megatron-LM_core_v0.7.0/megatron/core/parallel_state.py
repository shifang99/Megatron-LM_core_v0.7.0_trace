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
    if pg_name in nccl_comm_cfgs:                                              # trace_info : t_4836, t_4847, t_4855, t_4863, t_4871, ...
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None                                                            # trace_info : t_4837, t_4848, t_4856, t_4864, t_4872, ...


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

    def prefix_product(a: List[int], init=1) -> List[int]:                     # trace_info : t_4463, t_4916, t_5388, t_5833, t_6096, ...
        r = [init]                                                             # trace_info : t_4469, t_4487, t_4499, t_4519, t_4541, ...
        for v in a:                                                            # trace_info : t_4470, t_4473, t_4476, t_4479, t_4482, ...
            init = init * v                                                    # trace_info : t_4471, t_4474, t_4477, t_4480, t_4489, ...
            r.append(init)                                                     # trace_info : t_4472, t_4475, t_4478, t_4481, t_4490, ...
        return r                                                               # trace_info : t_4483, t_4492, t_4510, t_4524, t_4552, ...

    def inner_product(a: List[int], b: List[int]) -> int:                      # trace_info : t_4464, t_4917, t_5389, t_5834, t_6097, ...
        return sum([x * y for x, y in zip(a, b)])                              # trace_info : t_4530, t_4532, t_4572, t_4574, t_4614, ...

    def decompose(index, shape, stride=None):                                  # trace_info : t_4465, t_4918, t_5390, t_5835, t_6098, ...
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:                                                     # trace_info : t_4497, t_4517, t_4539, t_4559, t_4581, ...
            stride = prefix_product(shape)                                     # trace_info : t_4498, t_4518, t_4540, t_4560, t_4582, ...
        idx = [(index // d) % s for s, d in zip(shape, stride)]                # trace_info : t_4511, t_4525, t_4553, t_4567, t_4595, ...
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index            # trace_info : t_4512, t_4526, t_4554, t_4568, t_4596, ...
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx                                                             # trace_info : t_4513, t_4527, t_4555, t_4569, t_4597, ...

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]               # trace_info : t_4466, t_4919, t_5391, t_5836, t_6099, ...
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]         # trace_info : t_4467, t_4920, t_5392, t_5837, t_6100, ...

    global_stride = prefix_product(parallel_size)                              # trace_info : t_4468, t_4921, t_5393, t_5838, t_6101, ...
    masked_stride = [d for d, m in zip(global_stride, mask) if m]              # trace_info : t_4484, t_4937, t_5409, t_5854, t_6117, ...
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]        # trace_info : t_4485, t_4938, t_5410, t_5855, t_6118, ...

    group_size = prefix_product(masked_shape)[-1]                              # trace_info : t_4486, t_4939, t_5411, t_5856, t_6119, ...
    num_of_group = world_size // group_size                                    # trace_info : t_4493, t_4949, t_5418, t_5866, t_6126, ...

    ranks = []                                                                 # trace_info : t_4494, t_4950, t_5419, t_5867, t_6127, ...
    for group_index in range(num_of_group):                                    # trace_info : t_4495, t_4537, t_4579, t_4621, t_4663, ...
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)          # trace_info : t_4496, t_4538, t_4580, t_4622, t_4664, ...
        rank = []                                                              # trace_info : t_4514, t_4556, t_4598, t_4640, t_4682, ...
        for rank_in_group in range(group_size):                                # trace_info : t_4515, t_4535, t_4557, t_4577, t_4599, ...
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)       # trace_info : t_4516, t_4558, t_4600, t_4642, t_4684, ...
            rank.append(                                                       # trace_info : t_4528, t_4534, t_4570, t_4576, t_4612, ...
                inner_product(decomposed_rank_idx, masked_stride)              # trace_info : t_4529, t_4533, t_4571, t_4575, t_4613, ...
                + inner_product(decomposed_group_idx, unmasked_stride)         # trace_info : t_4531, t_4573, t_4615, t_4657, t_4699, ...
            )
        ranks.append(rank)                                                     # trace_info : t_4536, t_4578, t_4620, t_4662, t_4704, ...
    return ranks                                                               # trace_info : t_4832, t_5288, t_5757, t_6072, t_6377, ...


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
        ordered_token = order.split('-')                                       # trace_info : t_4455, t_4906, t_5380, t_5823, t_6088, ...
        token = token.split('-')                                               # trace_info : t_4456, t_4907, t_5381, t_5824, t_6089, ...
        mask = [False] * len(ordered_token)                                    # trace_info : t_4457, t_4908, t_5382, t_5825, t_6090, ...
        for t in token:                                                        # trace_info : t_4458, t_4460, t_4909, t_4911, t_4913, ...
            mask[ordered_token.index(t)] = True                                # trace_info : t_4459, t_4910, t_4912, t_5384, t_5827, ...
        return mask                                                            # trace_info : t_4461, t_4914, t_5386, t_5831, t_6094, ...

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
        if independent_ep:                                                     # trace_info : t_4451, t_4902, t_5376, t_5819, t_6084, ...
            parallel_size = self.ordered_size_w_ep                             # trace_info : t_7431, t_7788, t_8256
            order = self.order_w_ep                                            # trace_info : t_7432, t_7789, t_8257
        else:
            parallel_size = self.ordered_size_wo_ep                            # trace_info : t_4452, t_4903, t_5377, t_5820, t_6085, ...
            order = self.order_wo_ep                                           # trace_info : t_4453, t_4904, t_5378, t_5821, t_6086, ...
        mask = self.get_mask(order, token)                                     # trace_info : t_4454, t_4905, t_5379, t_5822, t_6087, ...
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)# trace_info : t_4462, t_4915, t_5387, t_5832, t_6095, ...
        return ranks                                                           # trace_info : t_4833, t_5289, t_5758, t_6073, t_6378, ...


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

    for ranks in rank_generator.get_ranks('dp'):                               # trace_info : t_4450, t_4844, t_4852, t_4860, t_4868, ...
        group = torch.distributed.new_group(                                   # trace_info : t_4834, t_4838, t_4845, t_4849, t_4853, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)# trace_info : t_4835, t_4846, t_4854, t_4862, t_4870, ...
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")# trace_info : t_4839, t_4850, t_4858, t_4866, t_4874, ...
        if rank in ranks:                                                      # trace_info : t_4840, t_4851, t_4859, t_4867, t_4875, ...
            _DATA_PARALLEL_GROUP = group                                       # trace_info : t_4841
            _DATA_PARALLEL_GROUP_GLOO = group_gloo                             # trace_info : t_4842
            _DATA_PARALLEL_GLOBAL_RANKS = ranks                                # trace_info : t_4843
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):                    # trace_info : t_4901, t_5302, t_5312, t_5322, t_5332, ...
        group_with_cp = torch.distributed.new_group(                           # trace_info : t_5290, t_5294, t_5303, t_5307, t_5313, ...
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)# trace_info : t_5291, t_5304, t_5314, t_5324, t_5334, ...
        )
        group_with_cp_gloo = torch.distributed.new_group(                      # trace_info : t_5295, t_5297, t_5308, t_5310, t_5318, ...
            ranks_with_cp, timeout=timeout, backend="gloo"                     # trace_info : t_5296, t_5309, t_5319, t_5329, t_5339, ...
        )
        if rank in ranks_with_cp:                                              # trace_info : t_5298, t_5311, t_5321, t_5331, t_5341, ...
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp                       # trace_info : t_5299
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo             # trace_info : t_5300
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp                # trace_info : t_5301

    # Apply SHARP to DP process groups
    if use_sharp:                                                              # trace_info : t_5373
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
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'# trace_info : t_5374
    for ranks in rank_generator.get_ranks('cp'):                               # trace_info : t_5375, t_5767, t_5774, t_5781, t_5788, ...
        group = torch.distributed.new_group(                                   # trace_info : t_5759, t_5763, t_5768, t_5772, t_5775, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)# trace_info : t_5760, t_5769, t_5776, t_5783, t_5790, ...
        )
        if rank in ranks:                                                      # trace_info : t_5764, t_5773, t_5780, t_5787, t_5794, ...
            _CONTEXT_PARALLEL_GROUP = group                                    # trace_info : t_5765
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks                             # trace_info : t_5766

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'# trace_info : t_5817
    for ranks in rank_generator.get_ranks('tp-pp'):                            # trace_info : t_5818, t_6081
        group = torch.distributed.new_group(                                   # trace_info : t_6074, t_6078
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)# trace_info : t_6075
        )
        if rank in ranks:                                                      # trace_info : t_6079
            _MODEL_PARALLEL_GROUP = group                                      # trace_info : t_6080

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_6082
    ), 'tensor model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp'):                               # trace_info : t_6083, t_6387, t_6394, t_6401, t_6408
        group = torch.distributed.new_group(                                   # trace_info : t_6379, t_6383, t_6388, t_6392, t_6395, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)# trace_info : t_6380, t_6389, t_6396, t_6403
        )
        if rank in ranks:                                                      # trace_info : t_6384, t_6393, t_6400, t_6407
            _TENSOR_MODEL_PARALLEL_GROUP = group                               # trace_info : t_6385
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks                        # trace_info : t_6386

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None                                 # trace_info : t_6409
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'  # trace_info : t_6410
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'# trace_info : t_6411
    for ranks in rank_generator.get_ranks('pp'):                               # trace_info : t_6412, t_6696, t_6723
        group = torch.distributed.new_group(                                   # trace_info : t_6664, t_6668, t_6697, t_6701
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)# trace_info : t_6665, t_6698
        )
        if rank in ranks:                                                      # trace_info : t_6669, t_6702
            _PIPELINE_MODEL_PARALLEL_GROUP = group                             # trace_info : t_6670
            _PIPELINE_GLOBAL_RANKS = ranks                                     # trace_info : t_6671
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:                                                     # trace_info : t_6672, t_6703
            embedding_ranks = [ranks[0], ranks[-1]]                            # trace_info : t_6673, t_6704
            position_embedding_ranks = [ranks[0]]                              # trace_info : t_6674, t_6705
            if pipeline_model_parallel_split_rank is not None:                 # trace_info : t_6675, t_6706
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(                                   # trace_info : t_6676, t_6680, t_6707, t_6711
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)# trace_info : t_6677, t_6708
        )
        if rank in embedding_ranks:                                            # trace_info : t_6681, t_6712
            _EMBEDDING_GROUP = group                                           # trace_info : t_6682
        if rank in ranks:                                                      # trace_info : t_6683, t_6713
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks                          # trace_info : t_6684

        group = torch.distributed.new_group(                                   # trace_info : t_6685, t_6691, t_6714, t_6720
            position_embedding_ranks,                                          # trace_info : t_6686, t_6715
            timeout=timeout,                                                   # trace_info : t_6687, t_6716
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),               # trace_info : t_6688, t_6717
        )
        if rank in position_embedding_ranks:                                   # trace_info : t_6692, t_6721
            _POSITION_EMBEDDING_GROUP = group                                  # trace_info : t_6693
        if rank in ranks:                                                      # trace_info : t_6694, t_6722
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks        # trace_info : t_6695

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None                                # trace_info : t_6724
    ), 'Tensor + data parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-dp-cp'):                         # trace_info : t_6725, t_7062, t_7069, t_7076, t_7083
        group = torch.distributed.new_group(                                   # trace_info : t_7055, t_7059, t_7063, t_7067, t_7070, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)# trace_info : t_7056, t_7064, t_7071, t_7078
        )
        if rank in ranks:                                                      # trace_info : t_7060, t_7068, t_7075, t_7082
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group                    # trace_info : t_7061
    for ranks in rank_generator.get_ranks('tp-dp'):                            # trace_info : t_7084, t_7404, t_7411, t_7418, t_7425
        group = torch.distributed.new_group(                                   # trace_info : t_7397, t_7401, t_7405, t_7409, t_7412, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)# trace_info : t_7398, t_7406, t_7413, t_7420
        )
        if rank in ranks:                                                      # trace_info : t_7402, t_7410, t_7417, t_7424
            _TENSOR_AND_DATA_PARALLEL_GROUP = group                            # trace_info : t_7403

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'# trace_info : t_7426
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None                              # trace_info : t_7427
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None                             # trace_info : t_7428
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO

    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):       # trace_info : t_7429, t_7764, t_7771, t_7778, t_7785
        group = torch.distributed.new_group(                                   # trace_info : t_7757, t_7761, t_7765, t_7769, t_7772, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)# trace_info : t_7758, t_7766, t_7773, t_7780
        )
        if rank in ranks:                                                      # trace_info : t_7762, t_7770, t_7777, t_7784
            _TENSOR_AND_EXPERT_PARALLEL_GROUP = group                          # trace_info : t_7763

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):          # trace_info : t_7786, t_8204, t_8211, t_8218, t_8225, ...
        group = torch.distributed.new_group(                                   # trace_info : t_8197, t_8201, t_8205, t_8209, t_8212, ...
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)          # trace_info : t_8198, t_8206, t_8213, t_8220, t_8227, ...
        )
        if rank in ranks:                                                      # trace_info : t_8202, t_8210, t_8217, t_8224, t_8231, ...
            _EXPERT_MODEL_PARALLEL_GROUP = group                               # trace_info : t_8203

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):          # trace_info : t_8254, t_8674, t_8682, t_8690, t_8698, ...
        group = torch.distributed.new_group(                                   # trace_info : t_8665, t_8669, t_8675, t_8679, t_8683, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)# trace_info : t_8666, t_8676, t_8684, t_8692, t_8700, ...
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")        # trace_info : t_8670, t_8680, t_8688, t_8696, t_8704, ...
        if rank in ranks:                                                      # trace_info : t_8671, t_8681, t_8689, t_8697, t_8705, ...
            _DATA_MODULO_EXPERT_PARALLEL_GROUP = group                         # trace_info : t_8672
            _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo               # trace_info : t_8673

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()                                                # trace_info : t_8731


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
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'# trace_info : t_15154, t_20224, t_20708, t_23952, t_24436, ...
    return _MODEL_PARALLEL_GROUP                                               # trace_info : t_15155, t_20225, t_20709, t_23953, t_24437, ...


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_8741, t_8774, t_8842, t_8847, t_9813, ...
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None                           # trace_info : t_8742, t_8775, t_8843, t_8848, t_9814, ...
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP                                        # trace_info : t_8743, t_8776, t_8844, t_8849, t_9815, ...


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None                             # trace_info : t_8751, t_8762, t_9162, t_9172, t_9183, ...
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP                                      # trace_info : t_8752, t_8763, t_9163, t_9173, t_9184, ...


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_12402, t_12467, t_12481, t_14048, t_15161, ...
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None                           # trace_info : t_12468, t_14049, t_15162
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP                                    # trace_info : t_12469, t_14050, t_15163
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'# trace_info : t_12403, t_12482, t_17122, t_17130, t_17172, ...
        return _DATA_PARALLEL_GROUP                                            # trace_info : t_12404, t_12483, t_17123, t_17131, t_17173, ...


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_15165
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None                      # trace_info : t_15166
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO                               # trace_info : t_15167
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_17956, t_17976, t_21686, t_21706, t_25414, ...
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'# trace_info : t_17957, t_17977, t_21687, t_21707, t_25415, ...
    return _CONTEXT_PARALLEL_GROUP                                             # trace_info : t_17958, t_17978, t_21688, t_21708, t_25416, ...


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'  # trace_info : t_12011, t_19935, t_23663, t_27391
    return _EMBEDDING_GROUP                                                    # trace_info : t_12012, t_19936, t_23664, t_27392


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
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None                          # trace_info : t_8836
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP                                   # trace_info : t_8837


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None                         # trace_info : t_12470
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP                                  # trace_info : t_12471


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
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:                      # trace_info : t_8739, t_8840, t_9811, t_10086, t_10137, ...
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())# trace_info : t_8740, t_8841, t_9812, t_10087, t_10138, ...


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:                    # trace_info : t_8749, t_9160, t_9185, t_9925, t_9991, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())# trace_info : t_8750, t_9161, t_9186, t_9926, t_9992, ...


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
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:                            # trace_info : t_8772, t_8845, t_9818, t_10234, t_10375, ...
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group()) # trace_info : t_8773, t_8846, t_9819, t_10235, t_10376, ...


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:                          # trace_info : t_8760, t_9170, t_9181, t_9986, t_10988, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())# trace_info : t_8761, t_9171, t_9182, t_9987, t_10989, ...


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_9166, t_11978, t_11997, t_16256, t_16455, ...
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None       # trace_info : t_9167, t_11979, t_11998, t_16257, t_16456, ...
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0                             # trace_info : t_9169, t_11981, t_12000, t_16259, t_16458, ...


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_9175, t_19393, t_19418, t_19483, t_19606, ...
        virtual_pipeline_model_parallel_world_size = (                         # trace_info : t_9178, t_19396, t_19421, t_19486, t_19609, ...
            get_virtual_pipeline_model_parallel_world_size()                   # trace_info : t_9176, t_19394, t_19419, t_19484, t_19607, ...
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (# trace_info : t_9179, t_19397, t_19422, t_19487, t_19610, ...
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)# trace_info : t_9180, t_19398, t_19423, t_19488, t_19611, ...


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()                                        # trace_info : t_11992, t_19896, t_23624, t_27352
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:                                                         # trace_info : t_11993, t_19897, t_23625, t_27353
        return rank in _EMBEDDING_GLOBAL_RANKS                                 # trace_info : t_19898, t_23626, t_27354
    if rank in _EMBEDDING_GLOBAL_RANKS:                                        # trace_info : t_11994
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:                                 # trace_info : t_11995
            return is_pipeline_first_stage(ignore_virtual=False)               # trace_info : t_11996
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()                                        # trace_info : t_19939, t_23667, t_27395
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS                            # trace_info : t_19940, t_23668, t_27396


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
    if get_pipeline_model_parallel_world_size() == 1:                          # trace_info : t_19460, t_19758, t_23188, t_23486, t_26916, ...
        return True
    if rank is None:                                                           # trace_info : t_19465, t_19763, t_23193, t_23491, t_26921, ...
        rank = get_pipeline_model_parallel_rank()                              # trace_info : t_19466, t_19764, t_23194, t_23492, t_26922, ...
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:                            # trace_info : t_19471, t_19769, t_23199, t_23497, t_26927, ...
        return True                                                            # trace_info : t_19472, t_19770, t_23200, t_23498, t_26928, ...
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
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK                               # trace_info : t_16252


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE                         # trace_info : t_9168, t_9177, t_9931, t_9997, t_10999, ...


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None                        # trace_info : t_18247, t_18255, t_18263, t_21977, t_21985, ...
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]                              # trace_info : t_18248, t_18256, t_18264, t_21978, t_21986, ...


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
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"# trace_info : t_19541, t_19673, t_23269, t_23401, t_26997, ...
    rank_in_pipeline = get_pipeline_model_parallel_rank()                      # trace_info : t_19542, t_19674, t_23270, t_23402, t_26998, ...
    world_size = get_pipeline_model_parallel_world_size()                      # trace_info : t_19547, t_19679, t_23275, t_23407, t_27003, ...
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]         # trace_info : t_19552, t_19684, t_23280, t_23412, t_27008, ...


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_12478, t_17126, t_17176, t_17223, t_21131, ...
        return torch.distributed.get_world_size(                               # trace_info : t_12479, t_12484, t_17127, t_17132, t_17177, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_12480, t_17128, t_17178, t_17225, t_21133, ...
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_12399, t_14045, t_17118, t_17168, t_17215
        return torch.distributed.get_rank(                                     # trace_info : t_12400, t_12405, t_14046, t_14051, t_17119, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_12401, t_14047, t_17120, t_17170, t_17217
        )
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_17954, t_17974, t_21684, t_21704, t_25412, ...
        return torch.distributed.get_world_size(group=get_context_parallel_group())# trace_info : t_17955, t_17975, t_21685, t_21705, t_25413, ...
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
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:                                        # trace_info : t_8832
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_8833
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(          # trace_info : t_8834, t_8838
            group=get_tensor_and_expert_parallel_group()                       # trace_info : t_8835
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()# trace_info : t_8839
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
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'# trace_info : t_8732
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()                               # trace_info : t_8733


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'# trace_info : t_18544, t_19039, t_22274, t_22767, t_26002, ...
    return _GLOBAL_MEMORY_BUFFER                                               # trace_info : t_18545, t_19040, t_22275, t_22768, t_26003, ...


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
