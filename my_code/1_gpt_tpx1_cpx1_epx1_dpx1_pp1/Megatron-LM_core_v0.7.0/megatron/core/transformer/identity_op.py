# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_7142, t_7164, t_7188, t_7206, t_7223, ...

    def forward(self, x, *args, **kwargs):
        return x                                                               # trace_info : t_15380, t_15383, t_15580, t_15587, t_15594, ...


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_7222, t_8224

    def forward(self, *args, **kwargs):
        return super().forward                                                 # trace_info : t_15591, t_16092, t_19230, t_19731, t_22869, ...
