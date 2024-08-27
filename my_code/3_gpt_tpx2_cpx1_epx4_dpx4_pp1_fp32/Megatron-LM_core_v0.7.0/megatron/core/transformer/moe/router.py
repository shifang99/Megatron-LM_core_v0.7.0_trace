# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)                                               # trace_info : t_10287, t_11427
        self.config = config                                                   # trace_info : t_10290, t_11430
        self.num_experts = self.config.num_moe_experts                         # trace_info : t_10291, t_11431
        self.moe_aux_loss_func = None                                          # trace_info : t_10292, t_11432
        self.layer_number = None                                               # trace_info : t_10293, t_11433

        # Initialize the gate weights.
        self.weight = torch.nn.Parameter(                                      # trace_info : t_10294, t_10296, t_11434, t_11436
            torch.empty((self.config.num_moe_experts, self.config.hidden_size))# trace_info : t_10295, t_11435
        )
        with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):# trace_info : t_10297, t_10320, t_11437, t_11460
            config.init_method(self.weight)                                    # trace_info : t_10318, t_11458
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)    # trace_info : t_10333, t_11473

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        logits = torch.nn.functional.linear(input, self.weight)                # trace_info : t_18686, t_19444, t_23034, t_23789, t_27379, ...
        return logits                                                          # trace_info : t_18687, t_19445, t_23035, t_23790, t_27380, ...

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors representing max probs and the indices.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number                                       # trace_info : t_10684, t_11824


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig,) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)                                        # trace_info : t_10286, t_11426
        self.topk = self.config.moe_router_topk                                # trace_info : t_10334, t_11474
        self.routing_type = self.config.moe_router_load_balancing_type         # trace_info : t_10335, t_11475
        self.input_jitter = None                                               # trace_info : t_10336, t_11476

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

            Args:
                logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

            Returns:
                probs (torch.Tensor): the probabilities tensor after load balancing.
                indices (torch.Tensor): the indices tensor after top-k selection.
        """
        probs, indices, tokens_per_expert = topk_softmax_with_capacity(        # trace_info : t_18704, t_18710, t_19462, t_19468, t_23052, ...
            logits,                                                            # trace_info : t_18705, t_19463, t_23053, t_23808, t_27398, ...
            self.topk,                                                         # trace_info : t_18706, t_19464, t_23054, t_23809, t_27399, ...
            capacity_factor=self.config.moe_expert_capacity_factor,            # trace_info : t_18707, t_19465, t_23055, t_23810, t_27400, ...
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,      # trace_info : t_18708, t_19466, t_23056, t_23811, t_27401, ...
            drop_policy=self.config.moe_token_drop_policy,                     # trace_info : t_18709, t_19467, t_23057, t_23812, t_27402, ...
        )

        # Apply load balancing loss
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)            # trace_info : t_18719, t_19477, t_23067, t_23822, t_27412, ...
        probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)# trace_info : t_18720, t_19478, t_23068, t_23823, t_27413, ...
        return probs, indices                                                  # trace_info : t_18754, t_19509, t_23099, t_23854, t_27444, ...

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_aux_loss_coeff = (                                                 # trace_info : t_18727, t_19485, t_23075, t_23830, t_27420, ...
            self.config.moe_aux_loss_coeff / parallel_state.get_tensor_model_parallel_world_size()# trace_info : t_18721, t_19479, t_23069, t_23824, t_27414, ...
        )
        aux_loss = switch_load_balancing_loss_func(                            # trace_info : t_18728, t_18730, t_19486, t_19488, t_23076, ...
            probs, num_local_tokens_per_expert, self.topk, moe_aux_loss_coeff  # trace_info : t_18729, t_19487, t_23077, t_23832, t_27422, ...
        )
        save_to_aux_losses_tracker(                                            # trace_info : t_18738, t_18743, t_19496, t_19501, t_23086, ...
            "load_balancing_loss",                                             # trace_info : t_18739, t_19497, t_23087, t_23842, t_27432, ...
            aux_loss / moe_aux_loss_coeff,                                     # trace_info : t_18740, t_19498, t_23088, t_23843, t_27433, ...
            self.layer_number,                                                 # trace_info : t_18741, t_19499, t_23089, t_23844, t_27434, ...
            self.config.num_layers,                                            # trace_info : t_18742, t_19500, t_23090, t_23845, t_27435, ...
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)          # trace_info : t_18750, t_19505, t_23095, t_23850, t_27440, ...
        return activation                                                      # trace_info : t_18753, t_19508, t_23098, t_23853, t_27443, ...

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None:                           # trace_info : t_18692, t_19450, t_23040, t_23795, t_27385, ...
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff / parallel_state.get_tensor_model_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss",
                z_loss / self.config.moe_z_loss_coeff,
                self.layer_number,
                self.config.num_layers,
            )
        return logits                                                          # trace_info : t_18693, t_19451, t_23041, t_23796, t_27386, ...

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:                       # trace_info : t_18683, t_19441, t_23031, t_23786, t_27376, ...
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input                                                       # trace_info : t_18684, t_19442, t_23032, t_23787, t_27377, ...

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        logits = logits.view(-1, self.config.num_moe_experts)                  # trace_info : t_18690, t_19448, t_23038, t_23793, t_27383, ...

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)                                     # trace_info : t_18691, t_19449, t_23039, t_23794, t_27384, ...

        if (
            parallel_state.get_tensor_model_parallel_world_size() > 1          # trace_info : t_18694, t_19452, t_23042, t_23797, t_27387, ...
            and self.config.moe_token_dispatcher_type == "alltoall"            # trace_info : t_18700, t_19458, t_23048, t_23803, t_27393, ...
        ):
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":                                    # trace_info : t_18701, t_19459, t_23049, t_23804, t_27394, ...
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":                                  # trace_info : t_18702, t_19460, t_23050, t_23805, t_27395, ...
            scores, indices = self.aux_loss_load_balancing(logits)             # trace_info : t_18703, t_19461, t_23051, t_23806, t_27396, ...
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, indices, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, indices                                                 # trace_info : t_18755, t_19510, t_23100, t_23855, t_27445, ...

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]                                          # trace_info : t_18681, t_19439, t_23029, t_23784, t_27374, ...

        # Apply input jitter
        input = self.apply_input_jitter(input)                                 # trace_info : t_18682, t_19440, t_23030, t_23785, t_27375, ...
        logits = self.gating(input)                                            # trace_info : t_18685, t_19443, t_23033, t_23788, t_27378, ...
        logits = logits.view(-1, self.config.num_moe_experts)                  # trace_info : t_18688, t_19446, t_23036, t_23791, t_27381, ...

        scores, indices = self.routing(logits)                                 # trace_info : t_18689, t_19447, t_23037, t_23792, t_27382, ...

        return scores, indices                                                 # trace_info : t_18756, t_19511, t_23101, t_23856, t_27446, ...
