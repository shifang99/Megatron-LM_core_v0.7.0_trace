# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_10508, t_10530, t_10554, t_10572, t_10589, ...

    def forward(self, x, *args, **kwargs):
        return x                                                               # trace_info : t_18511, t_18514, t_18715, t_18722, t_18729, ...


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_10588, t_11590

    def forward(self, *args, **kwargs):
        return super().forward                                                 # trace_info : t_18726, t_19219, t_22454, t_22947, t_26182, ...
