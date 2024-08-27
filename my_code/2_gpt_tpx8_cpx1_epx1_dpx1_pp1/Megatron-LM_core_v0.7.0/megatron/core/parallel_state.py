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
    if pg_name in nccl_comm_cfgs:                                              # trace_info : t_6371, t_6382, t_6390, t_6398, t_6406, ...
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None                                                            # trace_info : t_6372, t_6383, t_6391, t_6399, t_6407, ...


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

    def prefix_product(a: List[int], init=1) -> List[int]:                     # trace_info : t_5998, t_6451, t_6923, t_7368, t_7631, ...
        r = [init]                                                             # trace_info : t_6004, t_6022, t_6034, t_6054, t_6076, ...
        for v in a:                                                            # trace_info : t_6005, t_6008, t_6011, t_6014, t_6017, ...
            init = init * v                                                    # trace_info : t_6006, t_6009, t_6012, t_6015, t_6024, ...
            r.append(init)                                                     # trace_info : t_6007, t_6010, t_6013, t_6016, t_6025, ...
        return r                                                               # trace_info : t_6018, t_6027, t_6045, t_6059, t_6087, ...

    def inner_product(a: List[int], b: List[int]) -> int:                      # trace_info : t_5999, t_6452, t_6924, t_7369, t_7632, ...
        return sum([x * y for x, y in zip(a, b)])                              # trace_info : t_6065, t_6067, t_6107, t_6109, t_6149, ...

    def decompose(index, shape, stride=None):                                  # trace_info : t_6000, t_6453, t_6925, t_7370, t_7633, ...
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:                                                     # trace_info : t_6032, t_6052, t_6074, t_6094, t_6116, ...
            stride = prefix_product(shape)                                     # trace_info : t_6033, t_6053, t_6075, t_6095, t_6117, ...
        idx = [(index // d) % s for s, d in zip(shape, stride)]                # trace_info : t_6046, t_6060, t_6088, t_6102, t_6130, ...
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index            # trace_info : t_6047, t_6061, t_6089, t_6103, t_6131, ...
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx                                                             # trace_info : t_6048, t_6062, t_6090, t_6104, t_6132, ...

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]               # trace_info : t_6001, t_6454, t_6926, t_7371, t_7634, ...
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]         # trace_info : t_6002, t_6455, t_6927, t_7372, t_7635, ...

    global_stride = prefix_product(parallel_size)                              # trace_info : t_6003, t_6456, t_6928, t_7373, t_7636, ...
    masked_stride = [d for d, m in zip(global_stride, mask) if m]              # trace_info : t_6019, t_6472, t_6944, t_7389, t_7652, ...
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]        # trace_info : t_6020, t_6473, t_6945, t_7390, t_7653, ...

    group_size = prefix_product(masked_shape)[-1]                              # trace_info : t_6021, t_6474, t_6946, t_7391, t_7654, ...
    num_of_group = world_size // group_size                                    # trace_info : t_6028, t_6484, t_6953, t_7401, t_7661, ...

    ranks = []                                                                 # trace_info : t_6029, t_6485, t_6954, t_7402, t_7662, ...
    for group_index in range(num_of_group):                                    # trace_info : t_6030, t_6072, t_6114, t_6156, t_6198, ...
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)          # trace_info : t_6031, t_6073, t_6115, t_6157, t_6199, ...
        rank = []                                                              # trace_info : t_6049, t_6091, t_6133, t_6175, t_6217, ...
        for rank_in_group in range(group_size):                                # trace_info : t_6050, t_6070, t_6092, t_6112, t_6134, ...
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)       # trace_info : t_6051, t_6093, t_6135, t_6177, t_6219, ...
            rank.append(                                                       # trace_info : t_6063, t_6069, t_6105, t_6111, t_6147, ...
                inner_product(decomposed_rank_idx, masked_stride)              # trace_info : t_6064, t_6068, t_6106, t_6110, t_6148, ...
                + inner_product(decomposed_group_idx, unmasked_stride)         # trace_info : t_6066, t_6108, t_6150, t_6192, t_6234, ...
            )
        ranks.append(rank)                                                     # trace_info : t_6071, t_6113, t_6155, t_6197, t_6239, ...
    return ranks                                                               # trace_info : t_6367, t_6823, t_7292, t_7607, t_7846, ...


class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp                                                           # trace_info : t_5923
        self.ep = ep                                                           # trace_info : t_5924
        self.dp = dp                                                           # trace_info : t_5925
        self.pp = pp                                                           # trace_info : t_5926
        self.cp = cp                                                           # trace_info : t_5927
        self.world_size = tp * dp * pp * cp                                    # trace_info : t_5928

        self.name_to_size = {                                                  # trace_info : t_5934
            "tp": self.tp,                                                     # trace_info : t_5929
            "pp": self.pp,                                                     # trace_info : t_5930
            "dp": self.dp,                                                     # trace_info : t_5931
            "ep": self.ep,                                                     # trace_info : t_5932
            "cp": self.cp,                                                     # trace_info : t_5933
        }
        self.order = order                                                     # trace_info : t_5935
        order = order.lower()                                                  # trace_info : t_5936

        if 'ep' in order:                                                      # trace_info : t_5937
            if 'ep-dp' not in order and 'dp-ep' not in order:                  # trace_info : t_5938
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():                                  # trace_info : t_5939, t_5942, t_5945, t_5948, t_5951, ...
            if name not in order and self.name_to_size[name] != 1:             # trace_info : t_5940, t_5943, t_5946, t_5949, t_5952
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:                                            # trace_info : t_5941, t_5944, t_5947, t_5950, t_5953
                order = order + '-' + name

        self.order_w_ep = order                                                # trace_info : t_5955
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])# trace_info : t_5956
        self.ordered_size_wo_ep = []                                           # trace_info : t_5957
        self.ordered_size_w_ep = []                                            # trace_info : t_5958

        for token in order.split('-'):                                         # trace_info : t_5959, t_5964, t_5969, t_5973, t_5977, ...
            if token == 'dp':                                                  # trace_info : t_5960, t_5965, t_5970, t_5974, t_5978
                self.ordered_size_w_ep.append(self.dp // self.ep)              # trace_info : t_5975
                self.ordered_size_wo_ep.append(self.dp)                        # trace_info : t_5976
            elif token == 'ep':                                                # trace_info : t_5961, t_5966, t_5971, t_5979
                self.ordered_size_w_ep.append(self.ep)                         # trace_info : t_5972
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])        # trace_info : t_5962, t_5967, t_5980
                self.ordered_size_wo_ep.append(self.name_to_size[token])       # trace_info : t_5963, t_5968, t_5981

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')                                       # trace_info : t_5990, t_6441, t_6915, t_7358, t_7623, ...
        token = token.split('-')                                               # trace_info : t_5991, t_6442, t_6916, t_7359, t_7624, ...
        mask = [False] * len(ordered_token)                                    # trace_info : t_5992, t_6443, t_6917, t_7360, t_7625, ...
        for t in token:                                                        # trace_info : t_5993, t_5995, t_6444, t_6446, t_6448, ...
            mask[ordered_token.index(t)] = True                                # trace_info : t_5994, t_6445, t_6447, t_6919, t_7362, ...
        return mask                                                            # trace_info : t_5996, t_6449, t_6921, t_7366, t_7629, ...

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
        if independent_ep:                                                     # trace_info : t_5986, t_6437, t_6911, t_7354, t_7619, ...
            parallel_size = self.ordered_size_w_ep                             # trace_info : t_9018, t_9288, t_9756
            order = self.order_w_ep                                            # trace_info : t_9019, t_9289, t_9757
        else:
            parallel_size = self.ordered_size_wo_ep                            # trace_info : t_5987, t_6438, t_6912, t_7355, t_7620, ...
            order = self.order_wo_ep                                           # trace_info : t_5988, t_6439, t_6913, t_7356, t_7621, ...
        mask = self.get_mask(order, token)                                     # trace_info : t_5989, t_6440, t_6914, t_7357, t_7622, ...
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)# trace_info : t_5997, t_6450, t_6922, t_7367, t_7630, ...
        return ranks                                                           # trace_info : t_6368, t_6824, t_7293, t_7608, t_7847, ...


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
    assert torch.distributed.is_initialized()                                  # trace_info : t_5896
    world_size: int = torch.distributed.get_world_size()                       # trace_info : t_5897

    if (
        world_size                                                             # trace_info : t_5898, t_5900, t_5902
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)# trace_info : t_5899
        != 0                                                                   # trace_info : t_5901
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (                                  # trace_info : t_5903, t_5905
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size# trace_info : t_5904
    )

    if data_parallel_size % expert_model_parallel_size != 0:                   # trace_info : t_5906
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:           # trace_info : t_5907
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size# trace_info : t_5908
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size# trace_info : t_5909

    if virtual_pipeline_model_parallel_size is not None:                       # trace_info : t_5910
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:                         # trace_info : t_5911
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()                                        # trace_info : t_5912

    nccl_comm_cfgs = {}                                                        # trace_info : t_5913
    if nccl_communicator_config_path is not None:                              # trace_info : t_5914
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    rank_generator = RankGenerator(                                            # trace_info : t_5915, t_5922
        tp=tensor_model_parallel_size,                                         # trace_info : t_5916
        ep=expert_model_parallel_size,                                         # trace_info : t_5917
        dp=data_parallel_size,                                                 # trace_info : t_5918
        pp=pipeline_model_parallel_size,                                       # trace_info : t_5919
        cp=context_parallel_size,                                              # trace_info : t_5920
        order=order,                                                           # trace_info : t_5921
    )
    timeout = timedelta(minutes=distributed_timeout_minutes)                   # trace_info : t_5983

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'# trace_info : t_5984

    for ranks in rank_generator.get_ranks('dp'):                               # trace_info : t_5985, t_6379, t_6387, t_6395, t_6403, ...
        group = torch.distributed.new_group(                                   # trace_info : t_6369, t_6373, t_6380, t_6384, t_6388, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)# trace_info : t_6370, t_6381, t_6389, t_6397, t_6405, ...
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")# trace_info : t_6374, t_6385, t_6393, t_6401, t_6409, ...
        if rank in ranks:                                                      # trace_info : t_6375, t_6386, t_6394, t_6402, t_6410, ...
            _DATA_PARALLEL_GROUP = group                                       # trace_info : t_6376
            _DATA_PARALLEL_GROUP_GLOO = group_gloo                             # trace_info : t_6377
            _DATA_PARALLEL_GLOBAL_RANKS = ranks                                # trace_info : t_6378
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):                    # trace_info : t_6436, t_6837, t_6847, t_6857, t_6867, ...
        group_with_cp = torch.distributed.new_group(                           # trace_info : t_6825, t_6829, t_6838, t_6842, t_6848, ...
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)# trace_info : t_6826, t_6839, t_6849, t_6859, t_6869, ...
        )
        group_with_cp_gloo = torch.distributed.new_group(                      # trace_info : t_6830, t_6832, t_6843, t_6845, t_6853, ...
            ranks_with_cp, timeout=timeout, backend="gloo"                     # trace_info : t_6831, t_6844, t_6854, t_6864, t_6874, ...
        )
        if rank in ranks_with_cp:                                              # trace_info : t_6833, t_6846, t_6856, t_6866, t_6876, ...
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp                       # trace_info : t_6834
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo             # trace_info : t_6835
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp                # trace_info : t_6836

    # Apply SHARP to DP process groups
    if use_sharp:                                                              # trace_info : t_6908
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
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'# trace_info : t_6909
    for ranks in rank_generator.get_ranks('cp'):                               # trace_info : t_6910, t_7302, t_7309, t_7316, t_7323, ...
        group = torch.distributed.new_group(                                   # trace_info : t_7294, t_7298, t_7303, t_7307, t_7310, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)# trace_info : t_7295, t_7304, t_7311, t_7318, t_7325, ...
        )
        if rank in ranks:                                                      # trace_info : t_7299, t_7308, t_7315, t_7322, t_7329, ...
            _CONTEXT_PARALLEL_GROUP = group                                    # trace_info : t_7300
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks                             # trace_info : t_7301

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'# trace_info : t_7352
    for ranks in rank_generator.get_ranks('tp-pp'):                            # trace_info : t_7353, t_7616
        group = torch.distributed.new_group(                                   # trace_info : t_7609, t_7613
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)# trace_info : t_7610
        )
        if rank in ranks:                                                      # trace_info : t_7614
            _MODEL_PARALLEL_GROUP = group                                      # trace_info : t_7615

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_7617
    ), 'tensor model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp'):                               # trace_info : t_7618, t_7856
        group = torch.distributed.new_group(                                   # trace_info : t_7848, t_7852
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)# trace_info : t_7849
        )
        if rank in ranks:                                                      # trace_info : t_7853
            _TENSOR_MODEL_PARALLEL_GROUP = group                               # trace_info : t_7854
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks                        # trace_info : t_7855

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None                                 # trace_info : t_7857
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'  # trace_info : t_7858
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'# trace_info : t_7859
    for ranks in rank_generator.get_ranks('pp'):                               # trace_info : t_7860, t_8275, t_8301, t_8327, t_8353, ...
        group = torch.distributed.new_group(                                   # trace_info : t_8244, t_8248, t_8276, t_8280, t_8302, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)# trace_info : t_8245, t_8277, t_8303, t_8329, t_8355, ...
        )
        if rank in ranks:                                                      # trace_info : t_8249, t_8281, t_8307, t_8333, t_8359, ...
            _PIPELINE_MODEL_PARALLEL_GROUP = group                             # trace_info : t_8250
            _PIPELINE_GLOBAL_RANKS = ranks                                     # trace_info : t_8251
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:                                                     # trace_info : t_8252, t_8282, t_8308, t_8334, t_8360, ...
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
            embedding_ranks = ranks                                            # trace_info : t_8253, t_8283, t_8309, t_8335, t_8361, ...
            position_embedding_ranks = ranks                                   # trace_info : t_8254, t_8284, t_8310, t_8336, t_8362, ...

        group = torch.distributed.new_group(                                   # trace_info : t_8255, t_8259, t_8285, t_8289, t_8311, ...
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)# trace_info : t_8256, t_8286, t_8312, t_8338, t_8364, ...
        )
        if rank in embedding_ranks:                                            # trace_info : t_8260, t_8290, t_8316, t_8342, t_8368, ...
            _EMBEDDING_GROUP = group                                           # trace_info : t_8261
        if rank in ranks:                                                      # trace_info : t_8262, t_8291, t_8317, t_8343, t_8369, ...
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks                          # trace_info : t_8263

        group = torch.distributed.new_group(                                   # trace_info : t_8264, t_8270, t_8292, t_8298, t_8318, ...
            position_embedding_ranks,                                          # trace_info : t_8265, t_8293, t_8319, t_8345, t_8371, ...
            timeout=timeout,                                                   # trace_info : t_8266, t_8294, t_8320, t_8346, t_8372, ...
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),               # trace_info : t_8267, t_8295, t_8321, t_8347, t_8373, ...
        )
        if rank in position_embedding_ranks:                                   # trace_info : t_8271, t_8299, t_8325, t_8351, t_8377, ...
            _POSITION_EMBEDDING_GROUP = group                                  # trace_info : t_8272
        if rank in ranks:                                                      # trace_info : t_8273, t_8300, t_8326, t_8352, t_8378, ...
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks        # trace_info : t_8274

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None                                # trace_info : t_8458
    ), 'Tensor + data parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-dp-cp'):                         # trace_info : t_8459, t_8748
        group = torch.distributed.new_group(                                   # trace_info : t_8741, t_8745
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)# trace_info : t_8742
        )
        if rank in ranks:                                                      # trace_info : t_8746
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group                    # trace_info : t_8747
    for ranks in rank_generator.get_ranks('tp-dp'):                            # trace_info : t_8749, t_9012
        group = torch.distributed.new_group(                                   # trace_info : t_9005, t_9009
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)# trace_info : t_9006
        )
        if rank in ranks:                                                      # trace_info : t_9010
            _TENSOR_AND_DATA_PARALLEL_GROUP = group                            # trace_info : t_9011

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'# trace_info : t_9013
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None                              # trace_info : t_9014
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None                             # trace_info : t_9015
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO

    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):       # trace_info : t_9016, t_9285
        group = torch.distributed.new_group(                                   # trace_info : t_9278, t_9282
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)# trace_info : t_9279
        )
        if rank in ranks:                                                      # trace_info : t_9283
            _TENSOR_AND_EXPERT_PARALLEL_GROUP = group                          # trace_info : t_9284

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):          # trace_info : t_9286, t_9704, t_9711, t_9718, t_9725, ...
        group = torch.distributed.new_group(                                   # trace_info : t_9697, t_9701, t_9705, t_9709, t_9712, ...
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)          # trace_info : t_9698, t_9706, t_9713, t_9720, t_9727, ...
        )
        if rank in ranks:                                                      # trace_info : t_9702, t_9710, t_9717, t_9724, t_9731, ...
            _EXPERT_MODEL_PARALLEL_GROUP = group                               # trace_info : t_9703

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):          # trace_info : t_9754, t_10174, t_10182, t_10190, t_10198, ...
        group = torch.distributed.new_group(                                   # trace_info : t_10165, t_10169, t_10175, t_10179, t_10183, ...
            ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)# trace_info : t_10166, t_10176, t_10184, t_10192, t_10200, ...
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")        # trace_info : t_10170, t_10180, t_10188, t_10196, t_10204, ...
        if rank in ranks:                                                      # trace_info : t_10171, t_10181, t_10189, t_10197, t_10205, ...
            _DATA_MODULO_EXPERT_PARALLEL_GROUP = group                         # trace_info : t_10172
            _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo               # trace_info : t_10173

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()                                                # trace_info : t_10231


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
        _TENSOR_MODEL_PARALLEL_GROUP is None                                   # trace_info : t_5883
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False                                                           # trace_info : t_5884
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'# trace_info : t_16907, t_21826, t_22350, t_25436, t_25960, ...
    return _MODEL_PARALLEL_GROUP                                               # trace_info : t_16908, t_21827, t_22351, t_25437, t_25961, ...


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:                                                      # trace_info : t_10241, t_10274, t_10342, t_10347, t_11314, ...
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None                           # trace_info : t_10242, t_10275, t_10343, t_10348, t_11315, ...
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP                                        # trace_info : t_10243, t_10276, t_10344, t_10349, t_11316, ...


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None                             # trace_info : t_10251, t_10262, t_10665, t_10674, t_10685, ...
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP                                      # trace_info : t_10252, t_10263, t_10666, t_10675, t_10686, ...


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_13981, t_14046, t_14060, t_15733, t_16914, ...
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None                           # trace_info : t_14047, t_15734, t_16915
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP                                    # trace_info : t_14048, t_15735, t_16916
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'# trace_info : t_13982, t_14061, t_18923, t_18931, t_18973, ...
        return _DATA_PARALLEL_GROUP                                            # trace_info : t_13983, t_14062, t_18924, t_18932, t_18974, ...


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:                                                  # trace_info : t_16918
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None                      # trace_info : t_16919
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO                               # trace_info : t_16920
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
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None                          # trace_info : t_10336
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP                                   # trace_info : t_10337


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None                         # trace_info : t_14049
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP                                  # trace_info : t_14050


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
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:                      # trace_info : t_10239, t_10340, t_11312, t_11587, t_11638, ...
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())# trace_info : t_10240, t_10341, t_11313, t_11588, t_11639, ...


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:                    # trace_info : t_10249, t_10663, t_10687, t_11426, t_11492, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())# trace_info : t_10250, t_10664, t_10688, t_11427, t_11493, ...


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
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:                            # trace_info : t_10272, t_10345, t_11319, t_11735, t_11876, ...
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group()) # trace_info : t_10273, t_10346, t_11320, t_11736, t_11877, ...


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:                          # trace_info : t_10260, t_10672, t_10683, t_11487, t_12489, ...
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())# trace_info : t_10261, t_10673, t_10684, t_11488, t_12490, ...


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_10668, t_18057, t_18256, t_18471, t_18686, ...
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None       # trace_info : t_10669, t_18058, t_18257, t_18472, t_18687, ...
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0                             # trace_info : t_10671, t_18060, t_18259, t_18474, t_18689, ...


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:                                                     # trace_info : t_10677, t_21238, t_21273, t_22777, t_24848, ...
        virtual_pipeline_model_parallel_world_size = (                         # trace_info : t_10680, t_21241, t_21276, t_24851, t_24886, ...
            get_virtual_pipeline_model_parallel_world_size()                   # trace_info : t_10678, t_21239, t_21274, t_24849, t_24884, ...
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (# trace_info : t_10681, t_21242, t_21277, t_24852, t_24887, ...
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)# trace_info : t_10682, t_21243, t_21278, t_22778, t_24853, ...


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()                                        # trace_info : t_21517, t_25127, t_28737
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:                                                         # trace_info : t_21518, t_25128, t_28738
        return rank in _EMBEDDING_GLOBAL_RANKS                                 # trace_info : t_21519, t_25129, t_28739
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
    rank = torch.distributed.get_rank()                                        # trace_info : t_21528, t_25138, t_28748
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS                            # trace_info : t_21529, t_25139, t_28749


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
    if get_pipeline_model_parallel_world_size() == 1:                          # trace_info : t_21343, t_24953, t_28563
        return True                                                            # trace_info : t_21348, t_24958, t_28568
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
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK                               # trace_info : t_18053


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE                         # trace_info : t_10670, t_10679, t_11432, t_11498, t_12500, ...


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None                        # trace_info : t_19965, t_19973, t_19981, t_19989, t_19997, ...
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]                              # trace_info : t_19966, t_19974, t_19982, t_19990, t_19998, ...


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
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_14057, t_18927, t_18977, t_19024, t_22803, ...
        return torch.distributed.get_world_size(                               # trace_info : t_14058, t_14063, t_18928, t_18933, t_18978, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_14059, t_18929, t_18979, t_19026, t_22805, ...
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_13978, t_15730, t_18919, t_18969, t_19016
        return torch.distributed.get_rank(                                     # trace_info : t_13979, t_13984, t_15731, t_15736, t_18920, ...
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)# trace_info : t_13980, t_15732, t_18921, t_18971, t_19018
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
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:                                        # trace_info : t_10332
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():# trace_info : t_10333
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(          # trace_info : t_10334, t_10338
            group=get_tensor_and_expert_parallel_group()                       # trace_info : t_10335
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()# trace_info : t_10339
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
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'# trace_info : t_10232
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()                               # trace_info : t_10233


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'# trace_info : t_20278, t_20765, t_23890, t_24375, t_27500, ...
    return _GLOBAL_MEMORY_BUFFER                                               # trace_info : t_20279, t_20766, t_23891, t_24376, t_27501, ...


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
