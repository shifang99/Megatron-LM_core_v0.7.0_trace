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
        super().__init__()                                                     # trace_info : t_6660, t_7244, t_7662, t_8246

        self.config = config                                                   # trace_info : t_6661, t_7245, t_7663, t_8247

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma   # trace_info : t_6662, t_7246, t_7664, t_8248
        assert (
            self.config.normalization == "LayerNorm"                           # trace_info : t_6663, t_7247, t_7665, t_8249
        ), f'({self.config.normalization}) is not supported in FusedLayerNorm'

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [                                            # trace_info : t_6664, t_7248, t_7666, t_8250
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
        persist_layer_norm = self.config.persist_layer_norm                    # trace_info : t_6665, t_7249, t_7667, t_8251
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:# trace_info : t_6666, t_7250, t_7668, t_8252
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:               # trace_info : t_6667, t_7251, t_7669, t_8253
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must currently be installed to use megatron core.')

        if isinstance(hidden_size, numbers.Integral):                          # trace_info : t_6668, t_7252, t_7670, t_8254
            hidden_size = (hidden_size,)                                       # trace_info : t_6669, t_7253, t_7671, t_8255
        self.hidden_size = torch.Size(hidden_size)                             # trace_info : t_6670, t_7254, t_7672, t_8256
        self.eps = eps                                                         # trace_info : t_6671, t_7255, t_7673, t_8257
        self.weight = Parameter(torch.Tensor(*hidden_size))                    # trace_info : t_6672, t_7256, t_7674, t_8258
        self.bias = Parameter(torch.Tensor(*hidden_size))                      # trace_info : t_6673, t_7257, t_7675, t_8259
        self.reset_parameters()                                                # trace_info : t_6674, t_7258, t_7676, t_8260
        self.persist_layer_norm = persist_layer_norm                           # trace_info : t_6678, t_7262, t_7680, t_8264
        self.sequence_parallel = self.config.sequence_parallel                 # trace_info : t_6679, t_7263, t_7681, t_8265

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)      # trace_info : t_6680, t_7264, t_7682, t_8266
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)        # trace_info : t_6681, t_7265, t_7683, t_8267

    def reset_parameters(self):

        if self.zero_centered_gamma:                                           # trace_info : t_6675, t_7259, t_7677, t_8261
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)                                            # trace_info : t_6676, t_7260, t_7678, t_8262
            init.zeros_(self.bias)                                             # trace_info : t_6677, t_7261, t_7679, t_8263

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight  # trace_info : t_15273, t_15598, t_15776, t_16099, t_18914, ...

        if self.persist_layer_norm:                                            # trace_info : t_15274, t_15599, t_15777, t_16100, t_18915, ...
            if 'memory_efficient' in inspect.getfullargspec(FastLayerNormFN.forward).args:# trace_info : t_15275, t_15600, t_15778, t_16101, t_18916, ...
                output = FastLayerNormFN.apply(                                # trace_info : t_15276, t_15278, t_15601, t_15603, t_15779, ...
                    input, weight, self.bias, self.eps, self.config.memory_efficient_layer_norm# trace_info : t_15277, t_15602, t_15780, t_16103, t_18918, ...
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(                                     # trace_info : t_15279, t_15281, t_15604, t_15606, t_15782, ...
                inp=output, requires_grad=input.requires_grad, keep_graph=True # trace_info : t_15280, t_15605, t_15783, t_16106, t_18921, ...
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

        return output                                                          # trace_info : t_15289, t_15614, t_15792, t_16115, t_18930, ...
