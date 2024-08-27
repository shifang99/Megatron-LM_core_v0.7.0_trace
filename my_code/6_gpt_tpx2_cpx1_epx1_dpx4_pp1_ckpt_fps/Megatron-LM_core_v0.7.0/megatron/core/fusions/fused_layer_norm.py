# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib
import inspect
import numbers

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

    HAVE_FUSED_LAYER_NORM = True
except:
    HAVE_FUSED_LAYER_NORM = False


class FusedLayerNorm(torch.nn.Module):

    """Layer Norm, fused into a single CUDA kernel.

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      persist_layer_norm (bool): Use persistent fused layer norm kernel.
      This kernel supports only a set of hidden sizes. Please
      check persist_ln_hidden_sizes if your hidden size is supported.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      config (TransformerConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'LayerNorm' here.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        super().__init__()                                                     # trace_info : t_9689, t_10273, t_10691, t_11275

        self.config = config                                                   # trace_info : t_9690, t_10274, t_10692, t_11276

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma   # trace_info : t_9691, t_10275, t_10693, t_11277
        assert (
            self.config.normalization == "LayerNorm"                           # trace_info : t_9692, t_10276, t_10694, t_11278
        ), f'({self.config.normalization}) is not supported in FusedLayerNorm'

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [                                            # trace_info : t_9693, t_10277, t_10695, t_11279
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        persist_layer_norm = self.config.persist_layer_norm                    # trace_info : t_9694, t_10278, t_10696, t_11280
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:# trace_info : t_9695, t_10279, t_10697, t_11281
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:               # trace_info : t_9696, t_10280, t_10698, t_11282
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must currently be installed to use megatron core.')

        if isinstance(hidden_size, numbers.Integral):                          # trace_info : t_9697, t_10281, t_10699, t_11283
            hidden_size = (hidden_size,)                                       # trace_info : t_9698, t_10282, t_10700, t_11284
        self.hidden_size = torch.Size(hidden_size)                             # trace_info : t_9699, t_10283, t_10701, t_11285
        self.eps = eps                                                         # trace_info : t_9700, t_10284, t_10702, t_11286
        self.weight = Parameter(torch.Tensor(*hidden_size))                    # trace_info : t_9701, t_10285, t_10703, t_11287
        self.bias = Parameter(torch.Tensor(*hidden_size))                      # trace_info : t_9702, t_10286, t_10704, t_11288
        self.reset_parameters()                                                # trace_info : t_9703, t_10287, t_10705, t_11289
        self.persist_layer_norm = persist_layer_norm                           # trace_info : t_9707, t_10291, t_10709, t_11293
        self.sequence_parallel = self.config.sequence_parallel                 # trace_info : t_9708, t_10292, t_10710, t_11294

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)      # trace_info : t_9709, t_10293, t_10711, t_11295
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)        # trace_info : t_9710, t_10294, t_10712, t_11296

    def reset_parameters(self):

        if self.zero_centered_gamma:                                           # trace_info : t_9704, t_10288, t_10706, t_11290
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)                                            # trace_info : t_9705, t_10289, t_10707, t_11291
            init.zeros_(self.bias)                                             # trace_info : t_9706, t_10290, t_10708, t_11292

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight  # trace_info : t_18425, t_18746, t_18920, t_19239, t_22064, ...

        if self.persist_layer_norm:                                            # trace_info : t_18426, t_18747, t_18921, t_19240, t_22065, ...
            if 'memory_efficient' in inspect.getfullargspec(FastLayerNormFN.forward).args:# trace_info : t_18427, t_18748, t_18922, t_19241, t_22066, ...
                output = FastLayerNormFN.apply(                                # trace_info : t_18428, t_18430, t_18749, t_18751, t_18923, ...
                    input, weight, self.bias, self.eps, self.config.memory_efficient_layer_norm# trace_info : t_18429, t_18750, t_18924, t_19243, t_22068, ...
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(                                     # trace_info : t_18431, t_18433, t_18752, t_18754, t_18926, ...
                inp=output, requires_grad=input.requires_grad, keep_graph=True # trace_info : t_18432, t_18753, t_18927, t_19246, t_22071, ...
            )

        else:
            if (
                'memory_efficient'
                in inspect.getfullargspec(FusedLayerNormAffineFunction.forward).args
            ):
                return FusedLayerNormAffineFunction.apply(
                    input,
                    weight,
                    self.bias,
                    self.hidden_size,
                    self.eps,
                    self.config.memory_efficient_layer_norm,
                )
            else:
                return FusedLayerNormAffineFunction.apply(
                    input, weight, self.bias, self.hidden_size, self.eps
                )

        return output                                                          # trace_info : t_18441, t_18762, t_18936, t_19255, t_22080, ...
