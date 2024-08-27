# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass                                              # trace_info : t_14366, t_14397
from typing import Callable, Optional                                          # trace_info : t_14367
                                                                               # trace_info : t_14368
import torch                                                                   # trace_info : t_14369
                                                                               # trace_info : t_14370
                                                                               # trace_info : t_14371
@dataclass                                                                     # trace_info : t_14372
class OptimizerConfig:                                                         # trace_info : t_14373
    """Configuration for optimizer."""                                         # trace_info : t_14374
                                                                               # trace_info : t_14375
    ##############                                                             # trace_info : t_14376
    # General                                                                  # trace_info : t_14377
    ##############                                                             # trace_info : t_14378
    optimizer: str = 'adam'                                                    # trace_info : t_14379
    """Optimizer to use (one of Adam or SGD)."""                               # trace_info : t_14380
                                                                               # trace_info : t_14381
    lr: Optional[float] = None                                                 # trace_info : t_14382
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each# trace_info : t_14383
       iteration would be different.                                           # trace_info : t_14384
    """                                                                        # trace_info : t_14385
                                                                               # trace_info : t_14386
    min_lr: Optional[float] = None                                             # trace_info : t_14387
    """Minumum value for learning rate. The scheduler clip values below this threshold."""# trace_info : t_14388
                                                                               # trace_info : t_14389
    decoupled_lr: Optional[float] = None                                       # trace_info : t_14390
    """Separate learning rate for the input and output layer."""

    decoupled_min_lr: Optional[float] = None
    """Minimum value for learning rate for the input and output layer. The scheduler clip values
       below this threshold.
    """

    weight_decay: float = 0.01
    """Weight decay coefficient for L2 regularization."""

    ##############
    # Precision
    ##############
    fp16: bool = False
    """If true, train with fp16 mixed precision training. Defaults to False."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training. Defaults to False."""

    params_dtype: torch.dtype = torch.float32
    """dtype used when intializing the weights. Defaults to torch.float32."""

    ###############
    # Loss scaling
    ###############
    loss_scale: Optional[float] = None
    """Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,
       dynamic loss scaling is used.
    """

    initial_loss_scale: float = 2 ** 32
    """Initial loss-scale for dynamic loss scaling."""

    min_loss_scale: float = 1.0
    """Minimum loss scale for dynamic loss scaling."""

    loss_scale_window: float = 1000
    """Window over which to raise/lower dynamic scale."""

    hysteresis: int = 2
    """Hysteresis for dynamic loss scaling."""

    ##############
    # Optimizer
    ##############
    # Adam
    adam_beta1: float = 0.9
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_beta2: float = 0.999
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_eps: float = 1e-08
    """Term added to the denominator to improve numerical stability in Adam optimizer."""

    # SGD.
    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_grad_reduce: bool = False
    """If true, overlap grad reduce-scatter with backward compute in distributed optimizer."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute in distributed optimizer."""

    ################
    # Miscellaneous
    ################
    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    log_num_zeros_in_grad: bool = False
    """If true, calculate and log the number of zeros in gradient."""

    barrier_with_L1_time: bool = False
    """If true, use barrier with level 1 time measurements."""

    timers: Callable = None
    """Function to get timers."""
