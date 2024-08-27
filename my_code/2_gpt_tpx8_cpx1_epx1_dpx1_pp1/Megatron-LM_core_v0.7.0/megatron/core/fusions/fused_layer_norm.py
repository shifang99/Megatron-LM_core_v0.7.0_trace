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
        super().__init__()                                                     # trace_info : t_11527, t_12111, t_12529, t_13113

        self.config = config                                                   # trace_info : t_11528, t_12112, t_12530, t_13114

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma   # trace_info : t_11529, t_12113, t_12531, t_13115
        assert (
            self.config.normalization == "LayerNorm"                           # trace_info : t_11530, t_12114, t_12532, t_13116
        ), f'({self.config.normalization}) is not supported in FusedLayerNorm'

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [                                            # trace_info : t_11531, t_12115, t_12533, t_13117
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
        persist_layer_norm = self.config.persist_layer_norm                    # trace_info : t_11532, t_12116, t_12534, t_13118
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:# trace_info : t_11533, t_12117, t_12535, t_13119
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:               # trace_info : t_11534, t_12118, t_12536, t_13120
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must currently be installed to use megatron core.')

        if isinstance(hidden_size, numbers.Integral):                          # trace_info : t_11535, t_12119, t_12537, t_13121
            hidden_size = (hidden_size,)                                       # trace_info : t_11536, t_12120, t_12538, t_13122
        self.hidden_size = torch.Size(hidden_size)                             # trace_info : t_11537, t_12121, t_12539, t_13123
        self.eps = eps                                                         # trace_info : t_11538, t_12122, t_12540, t_13124
        self.weight = Parameter(torch.Tensor(*hidden_size))                    # trace_info : t_11539, t_12123, t_12541, t_13125
        self.bias = Parameter(torch.Tensor(*hidden_size))                      # trace_info : t_11540, t_12124, t_12542, t_13126
        self.reset_parameters()                                                # trace_info : t_11541, t_12125, t_12543, t_13127
        self.persist_layer_norm = persist_layer_norm                           # trace_info : t_11545, t_12129, t_12547, t_13131
        self.sequence_parallel = self.config.sequence_parallel                 # trace_info : t_11546, t_12130, t_12548, t_13132

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)      # trace_info : t_11547, t_12131, t_12549, t_13133
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)        # trace_info : t_11548, t_12132, t_12550, t_13134

    def reset_parameters(self):

        if self.zero_centered_gamma:                                           # trace_info : t_11542, t_12126, t_12544, t_13128
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)                                            # trace_info : t_11543, t_12127, t_12545, t_13129
            init.zeros_(self.bias)                                             # trace_info : t_11544, t_12128, t_12546, t_13130

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight  # trace_info : t_20146, t_20459, t_20633, t_20944, t_23758, ...

        if self.persist_layer_norm:                                            # trace_info : t_20147, t_20460, t_20634, t_20945, t_23759, ...
            if 'memory_efficient' in inspect.getfullargspec(FastLayerNormFN.forward).args:# trace_info : t_20148, t_20461, t_20635, t_20946, t_23760, ...
                output = FastLayerNormFN.apply(                                # trace_info : t_20149, t_20151, t_20462, t_20464, t_20636, ...
                    input, weight, self.bias, self.eps, self.config.memory_efficient_layer_norm# trace_info : t_20150, t_20463, t_20637, t_20948, t_23762, ...
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(                                     # trace_info : t_20152, t_20154, t_20465, t_20467, t_20639, ...
                inp=output, requires_grad=input.requires_grad, keep_graph=True # trace_info : t_20153, t_20466, t_20640, t_20951, t_23765, ...
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

        return output                                                          # trace_info : t_20162, t_20475, t_20649, t_20960, t_23774, ...
