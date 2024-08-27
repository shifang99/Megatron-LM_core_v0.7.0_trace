# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
from megatron.core.transformer.moe.moe_utils import moe_gather, moe_scatter, permute, unpermute
from megatron.core.transformer.transformer_config import TransformerConfig


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config                                                   # trace_info : t_10669, t_11809

    @abstractmethod
    def token_permutation(
        self, tokens: torch.Tensor, indices: torch.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            indices (torch.Tensor): indices tensor.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self, expert_output: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor,
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (torch.Tensor): The output tensor from the expert models.
            probs (torch.Tensor): Each token's score with each expert.
            indices (torch.Tensor): The indices used to reorder the expert output.

        Returns: 
            (torch.Tensor, torch.Tensor): Unpermuted activation and optional bias.            
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig,
    ) -> None:
        """
        Initialize the zero token dropping router.
        """
        super().__init__(config=config)                                        # trace_info : t_10668, t_11808
        self.num_local_experts = num_local_experts                             # trace_info : t_10670, t_11810
        assert self.num_local_experts > 0, "Expected at least one expert"      # trace_info : t_10671, t_11811
        self.local_expert_indices = local_expert_indices                       # trace_info : t_10672, t_11812
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"# trace_info : t_10673, t_11813
        self.router_topk = config.moe_router_topk                              # trace_info : t_10674, t_11814
        self.add_bias = config.add_bias_linear                                 # trace_info : t_10675, t_11815

        # self.local_probs: probs of global token assignment to local experts.
        self.local_probs = None                                                # trace_info : t_10676, t_11816

        # self.indices: The indices of `local_indices` (which holds the un-sorted expert indices of tokens that local expert can process) that give its sorted order along dim 0.
        self.indices = None                                                    # trace_info : t_10677, t_11817

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where each element is True if it's between the local_expert_indices. Only useful when cross device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None                                           # trace_info : t_10678, t_11818

    def token_permutation(
        self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: input tokens of shape [SeqLen/TP, MBS, HiddenSize]
            max_prob: probs of local token assignment to global experts.
            max_ind: token assignment to local experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """
        self.hidden_shape = hidden_states.shape                                # trace_info : t_18760, t_19515, t_23105, t_23860, t_27450, ...
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])          # trace_info : t_18761, t_19516, t_23106, t_23861, t_27451, ...

        # Permute the tokens across the expert parallel devices.
        if (self.config.tensor_model_parallel_size > 1) or (                   # trace_info : t_18762, t_19517, t_23107, t_23862, t_27452, ...
            self.config.expert_model_parallel_size > 1
        ):
            with torch.no_grad():                                              # trace_info : t_18763, t_18785, t_19518, t_19540, t_23108, ...
                global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(# trace_info : t_18764, t_18766, t_19519, t_19521, t_23109, ...
                    max_ind                                                    # trace_info : t_18765, t_19520, t_23110, t_23865, t_27455, ...
                )
                # Create a mask of mapping between global and local tokens where each
                # element is True if it's between the local_expert_indices
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (# trace_info : t_18781, t_18783, t_19536, t_19538, t_23126, ...
                    global_indices <= self.local_expert_indices[-1]            # trace_info : t_18782, t_19537, t_23127, t_23882, t_27472, ...
                )
                local_indices = global_indices.masked_select(global_local_mask)# trace_info : t_18784, t_19539, t_23129, t_23884, t_27474, ...

            if self.router_topk > 1:  # k > 1                                  # trace_info : t_18786, t_19541, t_23131, t_23886, t_27476, ...
                global_probs = tensor_parallel.gather_from_sequence_parallel_region_to_moe(max_prob)# trace_info : t_18787, t_19542, t_23132, t_23887, t_27477, ...
                self.local_probs = global_probs.masked_select(global_local_mask)# trace_info : t_18802, t_19557, t_23147, t_23902, t_27492, ...
            else:
                self.local_probs = max_prob

            # [S*B/TP, H] -> [S*B, H]
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(# trace_info : t_18803, t_18805, t_19558, t_19560, t_23148, ...
                hidden_states, use_global_buffer=True                          # trace_info : t_18804, t_19559, t_23149, t_23904, t_27494, ...
            )
            # Reshape global_local_mask to be compatible with Tensor.gather
            global_local_map = global_local_mask.nonzero()[:, 0]               # trace_info : t_18826, t_19581, t_23171, t_23926, t_27516, ...
            self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])# trace_info : t_18827, t_19582, t_23172, t_23927, t_27517, ...
            local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)# trace_info : t_18828, t_19583, t_23173, t_23928, t_27518, ...
        else:
            if self.router_topk > 1:
                global_local_mask = torch.ones_like(max_ind).bool()
                local_indices = max_ind.masked_select(global_local_mask)
                self.local_probs = max_prob.masked_select(global_local_mask)
                global_local_map = global_local_mask.nonzero()[:, 0]
                self.global_local_map = global_local_map.view(-1, 1).expand(
                    -1, hidden_states.shape[-1]
                )
                local_hidden_states = torch.gather(hidden_states, 0, self.global_local_map)
            else:
                local_indices = max_ind
                self.local_probs = max_prob
                local_hidden_states = hidden_states
                self.global_local_map = None

        with torch.no_grad():                                                  # trace_info : t_18832, t_18841, t_19587, t_19596, t_23177, ...
            # The indices of local_indices that give its sorted order along dim 0.
            self.indices = torch.argsort(local_indices, dim=0)                 # trace_info : t_18833, t_19588, t_23178, t_23933, t_27523, ...
            tokens_per_expert = torch.histc(                                   # trace_info : t_18834, t_18839, t_19589, t_19594, t_23179, ...
                local_indices,                                                 # trace_info : t_18835, t_19590, t_23180, t_23935, t_27525, ...
                bins=self.num_local_experts,                                   # trace_info : t_18836, t_19591, t_23181, t_23936, t_27526, ...
                min=self.local_expert_indices[0],                              # trace_info : t_18837, t_19592, t_23182, t_23937, t_27527, ...
                max=self.local_expert_indices[-1],                             # trace_info : t_18838, t_19593, t_23183, t_23938, t_27528, ...
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)         # trace_info : t_18840, t_19595, t_23185, t_23940, t_27530, ...

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather
        self.indices = self.indices.view(-1, 1).expand(-1, hidden_states.shape[-1])# trace_info : t_18842, t_19597, t_23187, t_23942, t_27532, ...
        if self.num_local_experts > 1:                                         # trace_info : t_18843, t_19598, t_23188, t_23943, t_27533, ...
            permuted_local_hidden_states = moe_gather.apply(local_hidden_states, self.indices)
        else:
            permuted_local_hidden_states = local_hidden_states                 # trace_info : t_18844, t_19599, t_23189, t_23944, t_27534, ...
        return (                                                               # trace_info : t_18847, t_19602, t_23192, t_23947, t_27537, ...
            permuted_local_hidden_states,                                      # trace_info : t_18845, t_19600, t_23190, t_23945, t_27535, ...
            tokens_per_expert,                                                 # trace_info : t_18846, t_19601, t_23191, t_23946, t_27536, ...
        )

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: torch.Tensor = None,
    ):
        """
        Reverse process of `dispatch()` which permutes the ouput of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            ouput of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        # Stage1: unpermute the tokens and bias locally respectively.
        scores = self.local_probs.to(dtype=hidden_states.dtype)                # trace_info : t_18978, t_19733, t_23323, t_24078, t_27668, ...
        if self.num_local_experts > 1:                                         # trace_info : t_18979, t_19734, t_23324, t_24079, t_27669, ...
            assert self.indices.shape == hidden_states.shape
            unpermuted_local_hidden = moe_scatter.apply(hidden_states, self.indices)
        else:
            unpermuted_local_hidden = hidden_states                            # trace_info : t_18980, t_19735, t_23325, t_24080, t_27670, ...

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:                                               # trace_info : t_18981, t_19736, t_23326, t_24081, t_27671, ...
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)# trace_info : t_18982, t_19737, t_23327, t_24082, t_27672, ...

        unpermuted_local_bias = None                                           # trace_info : t_18983, t_19738, t_23328, t_24083, t_27673, ...
        if self.add_bias:                                                      # trace_info : t_18984, t_19739, t_23329, t_24084, t_27674, ...
            assert bias is not None                                            # trace_info : t_18985, t_19740, t_23330, t_24085, t_27675, ...
            unpermuted_local_bias = torch.zeros_like(hidden_states)            # trace_info : t_18986, t_19741, t_23331, t_24086, t_27676, ...
            assert self.indices.shape == bias.shape                            # trace_info : t_18987, t_19742, t_23332, t_24087, t_27677, ...
            unpermuted_local_bias = unpermuted_local_bias.scatter(0, self.indices, bias)# trace_info : t_18988, t_19743, t_23333, t_24088, t_27678, ...
            if self.router_topk > 1:                                           # trace_info : t_18989, t_19744, t_23334, t_24089, t_27679, ...
                unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)# trace_info : t_18990, t_19745, t_23335, t_24090, t_27680, ...

        output_total = unpermuted_local_hidden                                 # trace_info : t_18991, t_19746, t_23336, t_24091, t_27681, ...
        output_bias_total = unpermuted_local_bias                              # trace_info : t_18992, t_19747, t_23337, t_24092, t_27682, ...

        # Unpermute the tokens across expert parallel devices.
        if (self.config.tensor_model_parallel_size > 1) or (                   # trace_info : t_18993, t_19748, t_23338, t_24093, t_27683, ...
            self.config.expert_model_parallel_size > 1
        ):
            assert (
                self.global_local_map is not None                              # trace_info : t_18994, t_19749, t_23339, t_24094, t_27684, ...
            ), "global_local_map is necessary for `AllGather`."
            ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()# trace_info : t_18995, t_19750, t_23340, t_24095, t_27685, ...
            # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size# trace_info : t_19003, t_19758, t_23348, t_24103, t_27693, ...
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]] # trace_info : t_19004, t_19759, t_23349, t_24104, t_27694, ...
            assert self.global_local_map.shape == unpermuted_local_hidden.shape# trace_info : t_19005, t_19760, t_23350, t_24105, t_27695, ...
            unpermuted_global_hidden = moe_scatter.apply(                      # trace_info : t_19006, t_19008, t_19761, t_19763, t_23351, ...
                unpermuted_local_hidden, self.global_local_map, global_hidden_shape# trace_info : t_19007, t_19762, t_23352, t_24107, t_27697, ...
            )
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(# trace_info : t_19016, t_19018, t_19771, t_19773, t_23361, ...
                unpermuted_global_hidden                                       # trace_info : t_19017, t_19772, t_23362, t_24117, t_27707, ...
            )
            if self.add_bias:                                                  # trace_info : t_19034, t_19789, t_23379, t_24134, t_27724, ...
                # Unpermute the bias across expert parallel devices.
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)# trace_info : t_19035, t_19790, t_23380, t_24135, t_27725, ...
                unpermuted_global_bias = unpermuted_global_bias.scatter_add(   # trace_info : t_19036, t_19038, t_19791, t_19793, t_23381, ...
                    0, self.global_local_map, unpermuted_local_bias            # trace_info : t_19037, t_19792, t_23382, t_24137, t_27727, ...
                )
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(# trace_info : t_19039, t_19041, t_19794, t_19796, t_23384, ...
                    unpermuted_global_bias                                     # trace_info : t_19040, t_19795, t_23385, t_24140, t_27730, ...
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (                                          # trace_info : t_19063, t_19818, t_23408, t_24163, t_27753, ...
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()# trace_info : t_19057, t_19812, t_23402, t_24157, t_27747, ...
                )
        else:
            if self.router_topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.scatter_add(
                    0, self.global_local_map, unpermuted_local_hidden
                )
                if self.add_bias:
                    unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                    output_bias_total = unpermuted_global_bias.scatter_add(
                        0, self.global_local_map, unpermuted_local_bias
                    )

        if self.router_topk == 1:                                              # trace_info : t_19064, t_19819, t_23409, t_24164, t_27754, ...
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)                    # trace_info : t_19065, t_19820, t_23410, t_24165, t_27755, ...
        if self.add_bias:                                                      # trace_info : t_19066, t_19821, t_23411, t_24166, t_27756, ...
            assert output_bias_total is not None                               # trace_info : t_19067, t_19822, t_23412, t_24167, t_27757, ...
            if self.router_topk == 1:                                          # trace_info : t_19068, t_19823, t_23413, t_24168, t_27758, ...
                output_bias_total = output_bias_total * scores
            output_bias_total = output_bias_total.view(self.hidden_shape)      # trace_info : t_19069, t_19824, t_23414, t_24169, t_27759, ...
        else:
            output_bias_total = None

        return output_total, output_bias_total                                 # trace_info : t_19070, t_19825, t_23415, t_24170, t_27760, ...


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll Based Token dispatcher.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config=config)
        self.hidden_shape = None
        self.num_input_tokens = None
        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear
        self.ep_size = config.expert_model_parallel_size
        self.probs = None
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None

        # Token drop and padding.
        # We need to keep track of the token num if we drop tokens without padding them.
        self.num_out_tokens = None
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
        self.capacity = None

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation. This method computes the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = torch.histc(
            indices, bins=self.num_experts, min=0, max=self.num_experts
        )
        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.config.expert_model_parallel_size
        if self.drop_and_pad:
            # probs: [num_experts, capacity]
            self.capacity = self.probs.size(1)
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
            )
            return num_tokens_per_local_expert
        elif self.config.moe_expert_capacity_factor is not None:
            self.num_out_tokens = num_local_tokens_per_expert.sum().cpu()

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"))
                .numpy()
            )
            num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
                num_local_tokens_per_expert
            ).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, self.local_expert_indices
            ]
            self.output_splits = (
                self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
            )
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
                torch.device("cpu"), non_blocking=True
            )
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                -1, self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(
                torch.device("cpu"), non_blocking=True
            )

        if self.num_local_experts > 1:
            expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.config.num_moe_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

        return num_tokens_per_local_expert

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(indices)

        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

        # Permutation 1: input to AlltoAll input
        self.hiddden_shape_before_permute = hidden_states.shape
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            indices,
            num_out_tokens=self.num_out_tokens,
            padded_mode=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        global_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )

        # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                    global_input_tokens, self.global_input_tokens_local_experts_indices
                )
            else:
                global_input_tokens = global_input_tokens.reshape(
                    self.ep_size, self.num_local_experts, self.capacity, -1
                )
                global_input_tokens = (
                    global_input_tokens.transpose(0, 1)
                    .reshape(self.num_local_experts * self.ep_size * self.capacity, -1)
                    .contiguous()
                )

        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(
                hidden_states
            )

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                hidden_states = unpermute(
                    hidden_states, self.reversed_global_input_permutation_mapping,
                )
            else:
                hidden_states = hidden_states.reshape(
                    self.num_local_experts, self.ep_size, self.capacity, -1
                )
                hidden_states = (
                    hidden_states.transpose(0, 1)
                    .reshape(self.ep_size * self.num_local_experts * self.capacity, -1)
                    .contiguous()
                )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            hidden_states,
            self.input_splits,
            self.output_splits,
        )

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=self.probs,
            padded_mode=self.drop_and_pad,
            restore_shape=self.hiddden_shape_before_permute,
        )

        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None
