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
        super().__init__()                                                     # trace_info : t_9614, t_10196, t_10754, t_11336

        self.config = config                                                   # trace_info : t_9615, t_10197, t_10755, t_11337

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma   # trace_info : t_9616, t_10198, t_10756, t_11338
        assert (
            self.config.normalization == "LayerNorm"                           # trace_info : t_9617, t_10199, t_10757, t_11339
        ), f'({self.config.normalization}) is not supported in FusedLayerNorm'

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [                                            # trace_info : t_9618, t_10200, t_10758, t_11340
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
        persist_layer_norm = self.config.persist_layer_norm                    # trace_info : t_9619, t_10201, t_10759, t_11341
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:# trace_info : t_9620, t_10202, t_10760, t_11342
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:               # trace_info : t_9621, t_10203, t_10761, t_11343
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must currently be installed to use megatron core.')

        if isinstance(hidden_size, numbers.Integral):                          # trace_info : t_9622, t_10204, t_10762, t_11344
            hidden_size = (hidden_size,)                                       # trace_info : t_9623, t_10205, t_10763, t_11345
        self.hidden_size = torch.Size(hidden_size)                             # trace_info : t_9624, t_10206, t_10764, t_11346
        self.eps = eps                                                         # trace_info : t_9625, t_10207, t_10765, t_11347
        self.weight = Parameter(torch.Tensor(*hidden_size))                    # trace_info : t_9626, t_10208, t_10766, t_11348
        self.bias = Parameter(torch.Tensor(*hidden_size))                      # trace_info : t_9627, t_10209, t_10767, t_11349
        self.reset_parameters()                                                # trace_info : t_9628, t_10210, t_10768, t_11350
        self.persist_layer_norm = persist_layer_norm                           # trace_info : t_9632, t_10214, t_10772, t_11354
        self.sequence_parallel = self.config.sequence_parallel                 # trace_info : t_9633, t_10215, t_10773, t_11355

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)      # trace_info : t_9634, t_10216, t_10774, t_11356
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)        # trace_info : t_9635, t_10217, t_10775, t_11357

    def reset_parameters(self):

        if self.zero_centered_gamma:                                           # trace_info : t_9629, t_10211, t_10769, t_11351
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)                                            # trace_info : t_9630, t_10212, t_10770, t_11352
            init.zeros_(self.bias)                                             # trace_info : t_9631, t_10213, t_10771, t_11353

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight  # trace_info : t_18345, t_18654, t_19108, t_19412, t_22698, ...

        if self.persist_layer_norm:                                            # trace_info : t_18346, t_18655, t_19109, t_19413, t_22699, ...
            if 'memory_efficient' in inspect.getfullargspec(FastLayerNormFN.forward).args:# trace_info : t_18347, t_18656, t_19110, t_19414, t_22700, ...
                output = FastLayerNormFN.apply(                                # trace_info : t_18348, t_18350, t_18657, t_18659, t_19111, ...
                    input, weight, self.bias, self.eps, self.config.memory_efficient_layer_norm# trace_info : t_18349, t_18658, t_19112, t_19416, t_22702, ...
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(                                     # trace_info : t_18351, t_18353, t_18660, t_18662, t_19114, ...
                inp=output, requires_grad=input.requires_grad, keep_graph=True # trace_info : t_18352, t_18661, t_19115, t_19419, t_22705, ...
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

        return output                                                          # trace_info : t_18361, t_18670, t_19124, t_19428, t_22714, ...
