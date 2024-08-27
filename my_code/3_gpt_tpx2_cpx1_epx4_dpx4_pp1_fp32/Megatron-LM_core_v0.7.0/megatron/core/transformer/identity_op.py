# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_10094, t_10116, t_10140, t_10158, t_10175, ...

    def forward(self, x, *args, **kwargs):
        return x                                                               # trace_info : t_18469, t_18472, t_18636, t_18643, t_18650, ...


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__()                                                     # trace_info : t_10174, t_11314

    def forward(self, *args, **kwargs):
        return super().forward                                                 # trace_info : t_18647, t_19405, t_22995, t_23750, t_27340, ...
