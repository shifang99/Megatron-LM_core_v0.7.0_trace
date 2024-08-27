# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_10171, t_10193, t_10217, t_10235, t_10252, ...

    def forward(self, x, *args, **kwargs):
        return x                                                               # trace_info : t_18524, t_18527, t_18728, t_18735, t_18742, ...


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_10251, t_11253

    def forward(self, *args, **kwargs):
        return super().forward                                                 # trace_info : t_18739, t_19232, t_22376, t_22869, t_89983, ...
