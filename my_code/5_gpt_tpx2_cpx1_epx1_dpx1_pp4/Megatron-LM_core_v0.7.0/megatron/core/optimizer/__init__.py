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
    params_map = {}                                                            # trace_info : t_14696
    for model_chunk in model_chunks:                                           # trace_info : t_14697, t_15097
        for name, param in model_chunk.named_parameters():                     # trace_info : t_14698, t_14713, t_14727, t_14744, t_14760, ...
            if not param.requires_grad:                                        # trace_info : t_14699, t_14714, t_14728, t_14745, t_14761, ...
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)         # trace_info : t_14700, t_14715, t_14729, t_14746, t_14762, ...

            if no_weight_decay_cond is not None:                               # trace_info : t_14701, t_14716, t_14730, t_14747, t_14763, ...
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1        # trace_info : t_14702, t_14717, t_14731, t_14748, t_14764, ...

            if scale_lr_cond is not None:                                      # trace_info : t_14703, t_14718, t_14732, t_14749, t_14765, ...
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False                                               # trace_info : t_14704, t_14719, t_14733, t_14750, t_14766, ...

            if not no_wd and not scale_lr:                                     # trace_info : t_14705, t_14720, t_14734, t_14751, t_14767, ...
                wd_mult, lr_mult = 1.0, 1.0                                    # trace_info : t_14706, t_14721, t_14768, t_14798, t_14860, ...
            elif not no_wd and scale_lr:                                       # trace_info : t_14735, t_14752, t_14782, t_14812, t_14828, ...
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:                                       # trace_info : t_14736, t_14753, t_14783, t_14813, t_14829, ...
                wd_mult, lr_mult = 0.0, 1.0                                    # trace_info : t_14737, t_14754, t_14784, t_14814, t_14830, ...
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            is_decoupled_lr = False                                            # trace_info : t_14707, t_14722, t_14738, t_14755, t_14769, ...
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(                        # trace_info : t_14708, t_14723, t_14739, t_14756, t_14770, ...
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)      # trace_info : t_14709, t_14724, t_14740, t_14757, t_14771, ...
            if key not in params_map:                                          # trace_info : t_14710, t_14725, t_14741, t_14758, t_14772, ...
                params_map[key] = []                                           # trace_info : t_14711, t_14742
            params_map[key].append(param)                                      # trace_info : t_14712, t_14726, t_14743, t_14759, t_14773, ...

    param_groups = []                                                          # trace_info : t_15098
    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():# trace_info : t_15099, t_15109, t_15119
        assert len(params) > 0                                                 # trace_info : t_15100, t_15110
        param_groups.append(                                                   # trace_info : t_15101, t_15108, t_15111, t_15118
            {                                                                  # trace_info : t_15107, t_15117
                'params': params,                                              # trace_info : t_15102, t_15112
                'wd_mult': wd_mult,                                            # trace_info : t_15103, t_15113
                'lr_mult': lr_mult,                                            # trace_info : t_15104, t_15114
                'is_expert_parallel': is_expert_parallel,                      # trace_info : t_15105, t_15115
                'is_decoupled_lr': is_decoupled_lr,                            # trace_info : t_15106, t_15116
            }
        )

    return param_groups                                                        # trace_info : t_15120


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

    if decoupled_min_lr is None:                                               # trace_info : t_15128
        decoupled_min_lr = min_lr                                              # trace_info : t_15129

    for param_group in param_groups:                                           # trace_info : t_15130, t_15134, t_15138
        if param_group['is_decoupled_lr']:                                     # trace_info : t_15131, t_15135
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr                                         # trace_info : t_15132, t_15136
            param_group['min_lr'] = min_lr                                     # trace_info : t_15133, t_15137
    return param_groups                                                        # trace_info : t_15139


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
    if config.optimizer == 'adam':                                             # trace_info : t_15170
        optimizer = Adam(                                                      # trace_info : t_15171, t_15177
            param_groups,                                                      # trace_info : t_15172
            lr=config.lr,                                                      # trace_info : t_15173
            weight_decay=config.weight_decay,                                  # trace_info : t_15174
            betas=(config.adam_beta1, config.adam_beta2),                      # trace_info : t_15175
            eps=config.adam_eps,                                               # trace_info : t_15176
        )

        def init_state_fn(opt):                                                # trace_info : t_15178
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
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:         # trace_info : t_15179

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None                                                     # trace_info : t_15180

        # Constant loss scale.
        if config.loss_scale:                                                  # trace_info : t_15181
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:                                                    # trace_info : t_15182
                grad_scaler = DynamicGradScaler(                               # trace_info : t_15183, t_15190
                    initial_scale=config.initial_loss_scale,                   # trace_info : t_15184
                    min_scale=config.min_loss_scale,                           # trace_info : t_15185
                    growth_factor=2.0,                                         # trace_info : t_15186
                    backoff_factor=0.5,                                        # trace_info : t_15187
                    growth_interval=config.loss_scale_window,                  # trace_info : t_15188
                    hysteresis=config.hysteresis,                              # trace_info : t_15189
                )

        optimizer_args = [                                                     # trace_info : t_15212
            optimizer,                                                         # trace_info : t_15208
            config,                                                            # trace_info : t_15209
            grad_scaler,                                                       # trace_info : t_15210
            init_state_fn,                                                     # trace_info : t_15211
        ]
        if config.use_distributed_optimizer:                                   # trace_info : t_15213
            optimizer = DistributedOptimizer(
                *optimizer_args,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)     # trace_info : t_15214

        return optimizer                                                       # trace_info : t_15876

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

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_14682
        logger.info(f'Setting up optimizer with config {config}')              # trace_info : t_14683

    # Collect param groups.
    param_groups = _get_param_groups(                                          # trace_info : t_14689, t_14695
        model_chunks,                                                          # trace_info : t_14690
        no_weight_decay_cond,                                                  # trace_info : t_14691
        scale_lr_cond,                                                         # trace_info : t_14692
        lr_mult,                                                               # trace_info : t_14693
        use_decoupled_learning_rate=config.decoupled_lr is not None,           # trace_info : t_14694
    )
    param_groups = _update_min_and_max_lr_in_param_groups(                     # trace_info : t_15121, t_15127
        param_groups,                                                          # trace_info : t_15122
        lr=config.lr,                                                          # trace_info : t_15123
        min_lr=config.min_lr,                                                  # trace_info : t_15124
        decoupled_lr=config.decoupled_lr,                                      # trace_info : t_15125
        decoupled_min_lr=config.decoupled_min_lr,                              # trace_info : t_15126
    )

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}                                                     # trace_info : t_15140
    per_model_ep_buffers = {}                                                  # trace_info : t_15141
    for model_idx, model_chunk in enumerate(model_chunks):                     # trace_info : t_15142, t_15146
        if hasattr(model_chunk, 'buffers'):                                    # trace_info : t_15143
            per_model_buffers[model_idx] = model_chunk.buffers                 # trace_info : t_15144
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers# trace_info : t_15145

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))# trace_info : t_15147, t_15148, t_15149
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))# trace_info : t_15150, t_15151, t_15152

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())# trace_info : t_15153
    optimizers = [                                                             # trace_info : t_15877
        _get_megatron_optimizer_based_on_param_groups(                         # trace_info : t_15156, t_15169
            config,                                                            # trace_info : t_15157
            param_groups=dense_param_groups,                                   # trace_info : t_15158
            per_model_buffers=per_model_buffers,                               # trace_info : t_15159
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),# trace_info : t_15160
            data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),# trace_info : t_15164
            data_parallel_group_idx=model_parallel_rank,                       # trace_info : t_15168
        )
    ]
    if len(moe_param_groups) > 0:                                              # trace_info : t_15878
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

    if len(optimizers) == 1:                                                   # trace_info : t_15879
        return optimizers[0]                                                   # trace_info : t_15880

    return ChainedOptimizer(optimizers)
