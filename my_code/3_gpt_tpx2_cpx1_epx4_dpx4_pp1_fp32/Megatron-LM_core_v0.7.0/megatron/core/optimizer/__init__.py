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
    params_map = {}                                                            # trace_info : t_15093
    for model_chunk in model_chunks:                                           # trace_info : t_15094, t_15556
        for name, param in model_chunk.named_parameters():                     # trace_info : t_15095, t_15110, t_15124, t_15141, t_15157, ...
            if not param.requires_grad:                                        # trace_info : t_15096, t_15111, t_15125, t_15142, t_15158, ...
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)         # trace_info : t_15097, t_15112, t_15126, t_15143, t_15159, ...

            if no_weight_decay_cond is not None:                               # trace_info : t_15098, t_15113, t_15127, t_15144, t_15160, ...
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1        # trace_info : t_15099, t_15114, t_15128, t_15145, t_15161, ...

            if scale_lr_cond is not None:                                      # trace_info : t_15100, t_15115, t_15129, t_15146, t_15162, ...
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False                                               # trace_info : t_15101, t_15116, t_15130, t_15147, t_15163, ...

            if not no_wd and not scale_lr:                                     # trace_info : t_15102, t_15117, t_15131, t_15148, t_15164, ...
                wd_mult, lr_mult = 1.0, 1.0                                    # trace_info : t_15103, t_15118, t_15165, t_15195, t_15257, ...
            elif not no_wd and scale_lr:                                       # trace_info : t_15132, t_15149, t_15179, t_15209, t_15225, ...
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:                                       # trace_info : t_15133, t_15150, t_15180, t_15210, t_15226, ...
                wd_mult, lr_mult = 0.0, 1.0                                    # trace_info : t_15134, t_15151, t_15181, t_15211, t_15227, ...
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            is_decoupled_lr = False                                            # trace_info : t_15104, t_15119, t_15135, t_15152, t_15166, ...
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(                        # trace_info : t_15105, t_15120, t_15136, t_15153, t_15167, ...
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)      # trace_info : t_15106, t_15121, t_15137, t_15154, t_15168, ...
            if key not in params_map:                                          # trace_info : t_15107, t_15122, t_15138, t_15155, t_15169, ...
                params_map[key] = []                                           # trace_info : t_15108, t_15139, t_15276, t_15293
            params_map[key].append(param)                                      # trace_info : t_15109, t_15123, t_15140, t_15156, t_15170, ...

    param_groups = []                                                          # trace_info : t_15557
    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():# trace_info : t_15558, t_15568, t_15578, t_15588, t_15598
        assert len(params) > 0                                                 # trace_info : t_15559, t_15569, t_15579, t_15589
        param_groups.append(                                                   # trace_info : t_15560, t_15567, t_15570, t_15577, t_15580, ...
            {                                                                  # trace_info : t_15566, t_15576, t_15586, t_15596
                'params': params,                                              # trace_info : t_15561, t_15571, t_15581, t_15591
                'wd_mult': wd_mult,                                            # trace_info : t_15562, t_15572, t_15582, t_15592
                'lr_mult': lr_mult,                                            # trace_info : t_15563, t_15573, t_15583, t_15593
                'is_expert_parallel': is_expert_parallel,                      # trace_info : t_15564, t_15574, t_15584, t_15594
                'is_decoupled_lr': is_decoupled_lr,                            # trace_info : t_15565, t_15575, t_15585, t_15595
            }
        )

    return param_groups                                                        # trace_info : t_15599


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

    if decoupled_min_lr is None:                                               # trace_info : t_15607
        decoupled_min_lr = min_lr                                              # trace_info : t_15608

    for param_group in param_groups:                                           # trace_info : t_15609, t_15613, t_15617, t_15621, t_15625
        if param_group['is_decoupled_lr']:                                     # trace_info : t_15610, t_15614, t_15618, t_15622
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr                                         # trace_info : t_15611, t_15615, t_15619, t_15623
            param_group['min_lr'] = min_lr                                     # trace_info : t_15612, t_15616, t_15620, t_15624
    return param_groups                                                        # trace_info : t_15626


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
    if config.optimizer == 'adam':                                             # trace_info : t_15661, t_15714
        optimizer = Adam(                                                      # trace_info : t_15662, t_15668, t_15715, t_15721
            param_groups,                                                      # trace_info : t_15663, t_15716
            lr=config.lr,                                                      # trace_info : t_15664, t_15717
            weight_decay=config.weight_decay,                                  # trace_info : t_15665, t_15718
            betas=(config.adam_beta1, config.adam_beta2),                      # trace_info : t_15666, t_15719
            eps=config.adam_eps,                                               # trace_info : t_15667, t_15720
        )

        def init_state_fn(opt):                                                # trace_info : t_15669, t_15722
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
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:         # trace_info : t_15670, t_15723

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
        ]
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)

        return optimizer

    # FP32.
    return FP32Optimizer(optimizer, config, init_state_fn,)                    # trace_info : t_15671, t_15724


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

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_15079
        logger.info(f'Setting up optimizer with config {config}')              # trace_info : t_15080

    # Collect param groups.
    param_groups = _get_param_groups(                                          # trace_info : t_15086, t_15092
        model_chunks,                                                          # trace_info : t_15087
        no_weight_decay_cond,                                                  # trace_info : t_15088
        scale_lr_cond,                                                         # trace_info : t_15089
        lr_mult,                                                               # trace_info : t_15090
        use_decoupled_learning_rate=config.decoupled_lr is not None,           # trace_info : t_15091
    )
    param_groups = _update_min_and_max_lr_in_param_groups(                     # trace_info : t_15600, t_15606
        param_groups,                                                          # trace_info : t_15601
        lr=config.lr,                                                          # trace_info : t_15602
        min_lr=config.min_lr,                                                  # trace_info : t_15603
        decoupled_lr=config.decoupled_lr,                                      # trace_info : t_15604
        decoupled_min_lr=config.decoupled_min_lr,                              # trace_info : t_15605
    )

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}                                                     # trace_info : t_15627
    per_model_ep_buffers = {}                                                  # trace_info : t_15628
    for model_idx, model_chunk in enumerate(model_chunks):                     # trace_info : t_15629, t_15633
        if hasattr(model_chunk, 'buffers'):                                    # trace_info : t_15630
            per_model_buffers[model_idx] = model_chunk.buffers                 # trace_info : t_15631
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers# trace_info : t_15632

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))# trace_info : t_15634, t_15635, t_15636, t_15637, t_15638
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))# trace_info : t_15639, t_15640, t_15641, t_15642, t_15643

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())# trace_info : t_15644
    optimizers = [                                                             # trace_info : t_15680
        _get_megatron_optimizer_based_on_param_groups(                         # trace_info : t_15647, t_15660
            config,                                                            # trace_info : t_15648
            param_groups=dense_param_groups,                                   # trace_info : t_15649
            per_model_buffers=per_model_buffers,                               # trace_info : t_15650
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),# trace_info : t_15651
            data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),# trace_info : t_15655
            data_parallel_group_idx=model_parallel_rank,                       # trace_info : t_15659
        )
    ]
    if len(moe_param_groups) > 0:                                              # trace_info : t_15681
        model_parallel_world_size = torch.distributed.get_world_size(mpu.get_model_parallel_group())# trace_info : t_15682
        expert_parallel_rank = mpu.get_expert_model_parallel_rank()            # trace_info : t_15685
        optimizers.append(                                                     # trace_info : t_15699, t_15733
            _get_megatron_optimizer_based_on_param_groups(                     # trace_info : t_15700, t_15713
                config,                                                        # trace_info : t_15701
                param_groups=moe_param_groups,                                 # trace_info : t_15702
                per_model_buffers=per_model_ep_buffers,                        # trace_info : t_15703
                data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),# trace_info : t_15704
                data_parallel_group_gloo=mpu.get_data_modulo_expert_parallel_group_gloo(),# trace_info : t_15707
                data_parallel_group_idx=expert_parallel_rank * model_parallel_world_size# trace_info : t_15710, t_15712
                + model_parallel_rank,                                         # trace_info : t_15711
            )
        )

    if len(optimizers) == 1:                                                   # trace_info : t_15734
        return optimizers[0]

    return ChainedOptimizer(optimizers)                                        # trace_info : t_15735
