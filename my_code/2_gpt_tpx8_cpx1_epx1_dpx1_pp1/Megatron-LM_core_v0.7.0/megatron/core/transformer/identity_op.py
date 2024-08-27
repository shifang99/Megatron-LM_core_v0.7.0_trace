# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_12009, t_12031, t_12055, t_12073, t_12090, ...

    def forward(self, x, *args, **kwargs):
        return x                                                               # trace_info : t_20245, t_20248, t_20441, t_20448, t_20455, ...


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_12089, t_13091

    def forward(self, *args, **kwargs):
        return super().forward                                                 # trace_info : t_20452, t_20937, t_24062, t_24547, t_27672, ...
