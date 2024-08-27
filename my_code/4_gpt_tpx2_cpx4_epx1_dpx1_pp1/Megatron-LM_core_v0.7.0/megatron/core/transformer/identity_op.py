# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_9966, t_10332, t_10354, t_10378, t_10396, ...

    def forward(self, x, *args, **kwargs):
        return x                                                               # trace_info : t_18265, t_18302, t_18305, t_18394, t_18401, ...


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_10412, t_11190

    def forward(self, *args, **kwargs):
        return super().forward                                                 # trace_info : t_18405, t_18612, t_21591, t_21798, t_24777, ...
