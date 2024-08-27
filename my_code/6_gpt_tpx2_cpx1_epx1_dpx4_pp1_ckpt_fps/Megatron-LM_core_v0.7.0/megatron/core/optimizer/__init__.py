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
    params_map = {}                                                            # trace_info : t_14579
    for model_chunk in model_chunks:                                           # trace_info : t_14580, t_15012
        for name, param in model_chunk.named_parameters():                     # trace_info : t_14581, t_14596, t_14610, t_14627, t_14643, ...
            if not param.requires_grad:                                        # trace_info : t_14582, t_14597, t_14611, t_14628, t_14644, ...
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)         # trace_info : t_14583, t_14598, t_14612, t_14629, t_14645, ...

            if no_weight_decay_cond is not None:                               # trace_info : t_14584, t_14599, t_14613, t_14630, t_14646, ...
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1        # trace_info : t_14585, t_14600, t_14614, t_14631, t_14647, ...

            if scale_lr_cond is not None:                                      # trace_info : t_14586, t_14601, t_14615, t_14632, t_14648, ...
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False                                               # trace_info : t_14587, t_14602, t_14616, t_14633, t_14649, ...

            if not no_wd and not scale_lr:                                     # trace_info : t_14588, t_14603, t_14617, t_14634, t_14650, ...
                wd_mult, lr_mult = 1.0, 1.0                                    # trace_info : t_14589, t_14604, t_14651, t_14681, t_14743, ...
            elif not no_wd and scale_lr:                                       # trace_info : t_14618, t_14635, t_14665, t_14695, t_14711, ...
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:                                       # trace_info : t_14619, t_14636, t_14666, t_14696, t_14712, ...
                wd_mult, lr_mult = 0.0, 1.0                                    # trace_info : t_14620, t_14637, t_14667, t_14697, t_14713, ...
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            is_decoupled_lr = False                                            # trace_info : t_14590, t_14605, t_14621, t_14638, t_14652, ...
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(                        # trace_info : t_14591, t_14606, t_14622, t_14639, t_14653, ...
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)      # trace_info : t_14592, t_14607, t_14623, t_14640, t_14654, ...
            if key not in params_map:                                          # trace_info : t_14593, t_14608, t_14624, t_14641, t_14655, ...
                params_map[key] = []                                           # trace_info : t_14594, t_14625
            params_map[key].append(param)                                      # trace_info : t_14595, t_14609, t_14626, t_14642, t_14656, ...

    param_groups = []                                                          # trace_info : t_15013
    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():# trace_info : t_15014, t_15024, t_15034
        assert len(params) > 0                                                 # trace_info : t_15015, t_15025
        param_groups.append(                                                   # trace_info : t_15016, t_15023, t_15026, t_15033
            {                                                                  # trace_info : t_15022, t_15032
                'params': params,                                              # trace_info : t_15017, t_15027
                'wd_mult': wd_mult,                                            # trace_info : t_15018, t_15028
                'lr_mult': lr_mult,                                            # trace_info : t_15019, t_15029
                'is_expert_parallel': is_expert_parallel,                      # trace_info : t_15020, t_15030
                'is_decoupled_lr': is_decoupled_lr,                            # trace_info : t_15021, t_15031
            }
        )

    return param_groups                                                        # trace_info : t_15035


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

    if decoupled_min_lr is None:                                               # trace_info : t_15043
        decoupled_min_lr = min_lr                                              # trace_info : t_15044

    for param_group in param_groups:                                           # trace_info : t_15045, t_15049, t_15053
        if param_group['is_decoupled_lr']:                                     # trace_info : t_15046, t_15050
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr                                         # trace_info : t_15047, t_15051
            param_group['min_lr'] = min_lr                                     # trace_info : t_15048, t_15052
    return param_groups                                                        # trace_info : t_15054


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
    if config.optimizer == 'adam':                                             # trace_info : t_15085
        optimizer = Adam(                                                      # trace_info : t_15086, t_15092
            param_groups,                                                      # trace_info : t_15087
            lr=config.lr,                                                      # trace_info : t_15088
            weight_decay=config.weight_decay,                                  # trace_info : t_15089
            betas=(config.adam_beta1, config.adam_beta2),                      # trace_info : t_15090
            eps=config.adam_eps,                                               # trace_info : t_15091
        )

        def init_state_fn(opt):                                                # trace_info : t_15093
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
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:         # trace_info : t_15094

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None                                                     # trace_info : t_15095

        # Constant loss scale.
        if config.loss_scale:                                                  # trace_info : t_15096
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:                                                    # trace_info : t_15097
                grad_scaler = DynamicGradScaler(                               # trace_info : t_15098, t_15105
                    initial_scale=config.initial_loss_scale,                   # trace_info : t_15099
                    min_scale=config.min_loss_scale,                           # trace_info : t_15100
                    growth_factor=2.0,                                         # trace_info : t_15101
                    backoff_factor=0.5,                                        # trace_info : t_15102
                    growth_interval=config.loss_scale_window,                  # trace_info : t_15103
                    hysteresis=config.hysteresis,                              # trace_info : t_15104
                )

        optimizer_args = [                                                     # trace_info : t_15127
            optimizer,                                                         # trace_info : t_15123
            config,                                                            # trace_info : t_15124
            grad_scaler,                                                       # trace_info : t_15125
            init_state_fn,                                                     # trace_info : t_15126
        ]
        if config.use_distributed_optimizer:                                   # trace_info : t_15128
            optimizer = DistributedOptimizer(
                *optimizer_args,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)     # trace_info : t_15129

        return optimizer                                                       # trace_info : t_15839

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

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_14565
        logger.info(f'Setting up optimizer with config {config}')              # trace_info : t_14566

    # Collect param groups.
    param_groups = _get_param_groups(                                          # trace_info : t_14572, t_14578
        model_chunks,                                                          # trace_info : t_14573
        no_weight_decay_cond,                                                  # trace_info : t_14574
        scale_lr_cond,                                                         # trace_info : t_14575
        lr_mult,                                                               # trace_info : t_14576
        use_decoupled_learning_rate=config.decoupled_lr is not None,           # trace_info : t_14577
    )
    param_groups = _update_min_and_max_lr_in_param_groups(                     # trace_info : t_15036, t_15042
        param_groups,                                                          # trace_info : t_15037
        lr=config.lr,                                                          # trace_info : t_15038
        min_lr=config.min_lr,                                                  # trace_info : t_15039
        decoupled_lr=config.decoupled_lr,                                      # trace_info : t_15040
        decoupled_min_lr=config.decoupled_min_lr,                              # trace_info : t_15041
    )

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}                                                     # trace_info : t_15055
    per_model_ep_buffers = {}                                                  # trace_info : t_15056
    for model_idx, model_chunk in enumerate(model_chunks):                     # trace_info : t_15057, t_15061
        if hasattr(model_chunk, 'buffers'):                                    # trace_info : t_15058
            per_model_buffers[model_idx] = model_chunk.buffers                 # trace_info : t_15059
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers# trace_info : t_15060

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))# trace_info : t_15062, t_15063, t_15064
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))# trace_info : t_15065, t_15066, t_15067

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())# trace_info : t_15068
    optimizers = [                                                             # trace_info : t_15840
        _get_megatron_optimizer_based_on_param_groups(                         # trace_info : t_15071, t_15084
            config,                                                            # trace_info : t_15072
            param_groups=dense_param_groups,                                   # trace_info : t_15073
            per_model_buffers=per_model_buffers,                               # trace_info : t_15074
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),# trace_info : t_15075
            data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),# trace_info : t_15079
            data_parallel_group_idx=model_parallel_rank,                       # trace_info : t_15083
        )
    ]
    if len(moe_param_groups) > 0:                                              # trace_info : t_15841
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

    if len(optimizers) == 1:                                                   # trace_info : t_15842
        return optimizers[0]                                                   # trace_info : t_15843

    return ChainedOptimizer(optimizers)
