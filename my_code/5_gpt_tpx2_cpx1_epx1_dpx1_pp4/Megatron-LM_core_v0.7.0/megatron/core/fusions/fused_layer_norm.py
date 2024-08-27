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
        super().__init__()                                                     # trace_info : t_10026, t_10610, t_11028, t_11612

        self.config = config                                                   # trace_info : t_10027, t_10611, t_11029, t_11613

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma   # trace_info : t_10028, t_10612, t_11030, t_11614
        assert (
            self.config.normalization == "LayerNorm"                           # trace_info : t_10029, t_10613, t_11031, t_11615
        ), f'({self.config.normalization}) is not supported in FusedLayerNorm'

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [                                            # trace_info : t_10030, t_10614, t_11032, t_11616
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
        persist_layer_norm = self.config.persist_layer_norm                    # trace_info : t_10031, t_10615, t_11033, t_11617
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:# trace_info : t_10032, t_10616, t_11034, t_11618
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:               # trace_info : t_10033, t_10617, t_11035, t_11619
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must currently be installed to use megatron core.')

        if isinstance(hidden_size, numbers.Integral):                          # trace_info : t_10034, t_10618, t_11036, t_11620
            hidden_size = (hidden_size,)                                       # trace_info : t_10035, t_10619, t_11037, t_11621
        self.hidden_size = torch.Size(hidden_size)                             # trace_info : t_10036, t_10620, t_11038, t_11622
        self.eps = eps                                                         # trace_info : t_10037, t_10621, t_11039, t_11623
        self.weight = Parameter(torch.Tensor(*hidden_size))                    # trace_info : t_10038, t_10622, t_11040, t_11624
        self.bias = Parameter(torch.Tensor(*hidden_size))                      # trace_info : t_10039, t_10623, t_11041, t_11625
        self.reset_parameters()                                                # trace_info : t_10040, t_10624, t_11042, t_11626
        self.persist_layer_norm = persist_layer_norm                           # trace_info : t_10044, t_10628, t_11046, t_11630
        self.sequence_parallel = self.config.sequence_parallel                 # trace_info : t_10045, t_10629, t_11047, t_11631

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)      # trace_info : t_10046, t_10630, t_11048, t_11632
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)        # trace_info : t_10047, t_10631, t_11049, t_11633

    def reset_parameters(self):

        if self.zero_centered_gamma:                                           # trace_info : t_10041, t_10625, t_11043, t_11627
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)                                            # trace_info : t_10042, t_10626, t_11044, t_11628
            init.zeros_(self.bias)                                             # trace_info : t_10043, t_10627, t_11045, t_11629

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight  # trace_info : t_18412, t_18733, t_18907, t_19226, t_22142, ...

        if self.persist_layer_norm:                                            # trace_info : t_18413, t_18734, t_18908, t_19227, t_22143, ...
            if 'memory_efficient' in inspect.getfullargspec(FastLayerNormFN.forward).args:# trace_info : t_18414, t_18735, t_18909, t_19228, t_22144, ...
                output = FastLayerNormFN.apply(                                # trace_info : t_18415, t_18417, t_18736, t_18738, t_18910, ...
                    input, weight, self.bias, self.eps, self.config.memory_efficient_layer_norm# trace_info : t_18416, t_18737, t_18911, t_19230, t_22146, ...
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(                                     # trace_info : t_18418, t_18420, t_18739, t_18741, t_18913, ...
                inp=output, requires_grad=input.requires_grad, keep_graph=True # trace_info : t_18419, t_18740, t_18914, t_19233, t_22149, ...
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

        return output                                                          # trace_info : t_18428, t_18749, t_18923, t_19242, t_22158, ...
