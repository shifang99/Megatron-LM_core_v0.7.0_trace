# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from logging import getLogger
from typing import Callable, Dict, List, Optional

import torch
from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron.core import mpu

from ..distributed import ParamAndGradBuffer
from ..transformer.module import MegatronModule
from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

logger = getLogger(__name__)


def _get_param_groups(
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Callable,
    scale_lr_cond: Callable,
    lr_mult: float,
    use_decoupled_learning_rate: bool,
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func): function to determine whether a parameter
            should not perform weight decay.
        scale_lr_cond (func): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        use_decoupled_learning_rate (bool): true if using decoupled learning rate.

    Returns:
        List of parameter groups.
    """

    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}                                                            # trace_info : t_11550
    for model_chunk in model_chunks:                                           # trace_info : t_11551, t_11983
        for name, param in model_chunk.named_parameters():                     # trace_info : t_11552, t_11567, t_11581, t_11598, t_11614, ...
            if not param.requires_grad:                                        # trace_info : t_11553, t_11568, t_11582, t_11599, t_11615, ...
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)         # trace_info : t_11554, t_11569, t_11583, t_11600, t_11616, ...

            if no_weight_decay_cond is not None:                               # trace_info : t_11555, t_11570, t_11584, t_11601, t_11617, ...
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1        # trace_info : t_11556, t_11571, t_11585, t_11602, t_11618, ...

            if scale_lr_cond is not None:                                      # trace_info : t_11557, t_11572, t_11586, t_11603, t_11619, ...
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False                                               # trace_info : t_11558, t_11573, t_11587, t_11604, t_11620, ...

            if not no_wd and not scale_lr:                                     # trace_info : t_11559, t_11574, t_11588, t_11605, t_11621, ...
                wd_mult, lr_mult = 1.0, 1.0                                    # trace_info : t_11560, t_11575, t_11622, t_11652, t_11714, ...
            elif not no_wd and scale_lr:                                       # trace_info : t_11589, t_11606, t_11636, t_11666, t_11682, ...
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:                                       # trace_info : t_11590, t_11607, t_11637, t_11667, t_11683, ...
                wd_mult, lr_mult = 0.0, 1.0                                    # trace_info : t_11591, t_11608, t_11638, t_11668, t_11684, ...
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            is_decoupled_lr = False                                            # trace_info : t_11561, t_11576, t_11592, t_11609, t_11623, ...
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(                        # trace_info : t_11562, t_11577, t_11593, t_11610, t_11624, ...
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)      # trace_info : t_11563, t_11578, t_11594, t_11611, t_11625, ...
            if key not in params_map:                                          # trace_info : t_11564, t_11579, t_11595, t_11612, t_11626, ...
                params_map[key] = []                                           # trace_info : t_11565, t_11596
            params_map[key].append(param)                                      # trace_info : t_11566, t_11580, t_11597, t_11613, t_11627, ...

    param_groups = []                                                          # trace_info : t_11984
    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():# trace_info : t_11985, t_11995, t_12005
        assert len(params) > 0                                                 # trace_info : t_11986, t_11996
        param_groups.append(                                                   # trace_info : t_11987, t_11994, t_11997, t_12004
            {                                                                  # trace_info : t_11993, t_12003
                'params': params,                                              # trace_info : t_11988, t_11998
                'wd_mult': wd_mult,                                            # trace_info : t_11989, t_11999
                'lr_mult': lr_mult,                                            # trace_info : t_11990, t_12000
                'is_expert_parallel': is_expert_parallel,                      # trace_info : t_11991, t_12001
                'is_decoupled_lr': is_decoupled_lr,                            # trace_info : t_11992, t_12002
            }
        )

    return param_groups                                                        # trace_info : t_12006


def _update_min_and_max_lr_in_param_groups(
    param_groups: List[Dict],
    lr: float,
    min_lr: float,
    decoupled_lr: Optional[float],
    decoupled_min_lr: Optional[float],
) -> List[Dict]:
    """
    Updates `max_lr` and `min_lr` values in each parameter group, and returns new list.
    By default, each group will use `lr` / `min_lr` as `max_lr` / `min_lr`.
    If `decoupled_lr` is provided, then `decoupled_lr` / `decoupled_min_lr` will be used
    as `max_lr` / `min_lr` for the input and output layer.

    Args:
        param_groups (List): parameter groups whose 'max_lr' and `min_lr` fields need to
            be adjusted.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        decoupled_lr (Optional[float]): optional decoupled learning rate.
        decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.

    Returns:
        List of adjusted parameter groups.
    """

    if decoupled_min_lr is None:                                               # trace_info : t_12014
        decoupled_min_lr = min_lr                                              # trace_info : t_12015

    for param_group in param_groups:                                           # trace_info : t_12016, t_12020, t_12024
        if param_group['is_decoupled_lr']:                                     # trace_info : t_12017, t_12021
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr                                         # trace_info : t_12018, t_12022
            param_group['min_lr'] = min_lr                                     # trace_info : t_12019, t_12023
    return param_groups                                                        # trace_info : t_12025


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[ParamAndGradBuffer]]] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.

    Returns:
        Instance of MegatronOptimizer.
    """
    if config.optimizer == 'adam':                                             # trace_info : t_12056
        optimizer = Adam(                                                      # trace_info : t_12057, t_12063
            param_groups,                                                      # trace_info : t_12058
            lr=config.lr,                                                      # trace_info : t_12059
            weight_decay=config.weight_decay,                                  # trace_info : t_12060
            betas=(config.adam_beta1, config.adam_beta2),                      # trace_info : t_12061
            eps=config.adam_eps,                                               # trace_info : t_12062
        )

        def init_state_fn(opt):                                                # trace_info : t_12064
            for group in opt.param_groups:
                for p in group['params']:
                    if len(opt.state[p]) == 0:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    elif config.optimizer == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.sgd_momentum,
        )
        init_state_fn = None
    else:
        raise Exception('{} optimizer is not supported.'.format(config.optimizer))

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:         # trace_info : t_12065

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None                                                     # trace_info : t_12066

        # Constant loss scale.
        if config.loss_scale:                                                  # trace_info : t_12067
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:                                                    # trace_info : t_12068
                grad_scaler = DynamicGradScaler(                               # trace_info : t_12069, t_12076
                    initial_scale=config.initial_loss_scale,                   # trace_info : t_12070
                    min_scale=config.min_loss_scale,                           # trace_info : t_12071
                    growth_factor=2.0,                                         # trace_info : t_12072
                    backoff_factor=0.5,                                        # trace_info : t_12073
                    growth_interval=config.loss_scale_window,                  # trace_info : t_12074
                    hysteresis=config.hysteresis,                              # trace_info : t_12075
                )

        optimizer_args = [                                                     # trace_info : t_12098
            optimizer,                                                         # trace_info : t_12094
            config,                                                            # trace_info : t_12095
            grad_scaler,                                                       # trace_info : t_12096
            init_state_fn,                                                     # trace_info : t_12097
        ]
        if config.use_distributed_optimizer:                                   # trace_info : t_12099
            optimizer = DistributedOptimizer(
                *optimizer_args,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)     # trace_info : t_12100

        return optimizer                                                       # trace_info : t_12810

    # FP32.
    return FP32Optimizer(optimizer, config, init_state_fn,)


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.

    Returns:
        Instance of MegatronOptimizer.
    """

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_11536
        logger.info(f'Setting up optimizer with config {config}')              # trace_info : t_11537

    # Collect param groups.
    param_groups = _get_param_groups(                                          # trace_info : t_11543, t_11549
        model_chunks,                                                          # trace_info : t_11544
        no_weight_decay_cond,                                                  # trace_info : t_11545
        scale_lr_cond,                                                         # trace_info : t_11546
        lr_mult,                                                               # trace_info : t_11547
        use_decoupled_learning_rate=config.decoupled_lr is not None,           # trace_info : t_11548
    )
    param_groups = _update_min_and_max_lr_in_param_groups(                     # trace_info : t_12007, t_12013
        param_groups,                                                          # trace_info : t_12008
        lr=config.lr,                                                          # trace_info : t_12009
        min_lr=config.min_lr,                                                  # trace_info : t_12010
        decoupled_lr=config.decoupled_lr,                                      # trace_info : t_12011
        decoupled_min_lr=config.decoupled_min_lr,                              # trace_info : t_12012
    )

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}                                                     # trace_info : t_12026
    per_model_ep_buffers = {}                                                  # trace_info : t_12027
    for model_idx, model_chunk in enumerate(model_chunks):                     # trace_info : t_12028, t_12032
        if hasattr(model_chunk, 'buffers'):                                    # trace_info : t_12029
            per_model_buffers[model_idx] = model_chunk.buffers                 # trace_info : t_12030
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers# trace_info : t_12031

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))# trace_info : t_12033, t_12034, t_12035
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))# trace_info : t_12036, t_12037, t_12038

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())# trace_info : t_12039
    optimizers = [                                                             # trace_info : t_12811
        _get_megatron_optimizer_based_on_param_groups(                         # trace_info : t_12042, t_12055
            config,                                                            # trace_info : t_12043
            param_groups=dense_param_groups,                                   # trace_info : t_12044
            per_model_buffers=per_model_buffers,                               # trace_info : t_12045
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),# trace_info : t_12046
            data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),# trace_info : t_12050
            data_parallel_group_idx=model_parallel_rank,                       # trace_info : t_12054
        )
    ]
    if len(moe_param_groups) > 0:                                              # trace_info : t_12812
        model_parallel_world_size = torch.distributed.get_world_size(mpu.get_model_parallel_group())
        expert_parallel_rank = mpu.get_expert_model_parallel_rank()
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                param_groups=moe_param_groups,
                per_model_buffers=per_model_ep_buffers,
                data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                data_parallel_group_gloo=mpu.get_data_modulo_expert_parallel_group_gloo(),
                data_parallel_group_idx=expert_parallel_rank * model_parallel_world_size
                + model_parallel_rank,
            )
        )

    if len(optimizers) == 1:                                                   # trace_info : t_12813
        return optimizers[0]                                                   # trace_info : t_12814

    return ChainedOptimizer(optimizers)
