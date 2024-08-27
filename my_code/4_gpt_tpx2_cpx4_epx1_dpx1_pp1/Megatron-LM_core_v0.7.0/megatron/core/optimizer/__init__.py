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
    params_map = {}                                                            # trace_info : t_14409
    for model_chunk in model_chunks:                                           # trace_info : t_14410, t_14842
        for name, param in model_chunk.named_parameters():                     # trace_info : t_14411, t_14426, t_14440, t_14454, t_14471, ...
            if not param.requires_grad:                                        # trace_info : t_14412, t_14427, t_14441, t_14455, t_14472, ...
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)         # trace_info : t_14413, t_14428, t_14442, t_14456, t_14473, ...

            if no_weight_decay_cond is not None:                               # trace_info : t_14414, t_14429, t_14443, t_14457, t_14474, ...
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1        # trace_info : t_14415, t_14430, t_14444, t_14458, t_14475, ...

            if scale_lr_cond is not None:                                      # trace_info : t_14416, t_14431, t_14445, t_14459, t_14476, ...
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False                                               # trace_info : t_14417, t_14432, t_14446, t_14460, t_14477, ...

            if not no_wd and not scale_lr:                                     # trace_info : t_14418, t_14433, t_14447, t_14461, t_14478, ...
                wd_mult, lr_mult = 1.0, 1.0                                    # trace_info : t_14419, t_14434, t_14448, t_14511, t_14573, ...
            elif not no_wd and scale_lr:                                       # trace_info : t_14462, t_14479, t_14495, t_14525, t_14541, ...
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:                                       # trace_info : t_14463, t_14480, t_14496, t_14526, t_14542, ...
                wd_mult, lr_mult = 0.0, 1.0                                    # trace_info : t_14464, t_14481, t_14497, t_14527, t_14543, ...
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            is_decoupled_lr = False                                            # trace_info : t_14420, t_14435, t_14449, t_14465, t_14482, ...
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(                        # trace_info : t_14421, t_14436, t_14450, t_14466, t_14483, ...
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)      # trace_info : t_14422, t_14437, t_14451, t_14467, t_14484, ...
            if key not in params_map:                                          # trace_info : t_14423, t_14438, t_14452, t_14468, t_14485, ...
                params_map[key] = []                                           # trace_info : t_14424, t_14469
            params_map[key].append(param)                                      # trace_info : t_14425, t_14439, t_14453, t_14470, t_14486, ...

    param_groups = []                                                          # trace_info : t_14843
    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():# trace_info : t_14844, t_14854, t_14864
        assert len(params) > 0                                                 # trace_info : t_14845, t_14855
        param_groups.append(                                                   # trace_info : t_14846, t_14853, t_14856, t_14863
            {                                                                  # trace_info : t_14852, t_14862
                'params': params,                                              # trace_info : t_14847, t_14857
                'wd_mult': wd_mult,                                            # trace_info : t_14848, t_14858
                'lr_mult': lr_mult,                                            # trace_info : t_14849, t_14859
                'is_expert_parallel': is_expert_parallel,                      # trace_info : t_14850, t_14860
                'is_decoupled_lr': is_decoupled_lr,                            # trace_info : t_14851, t_14861
            }
        )

    return param_groups                                                        # trace_info : t_14865


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

    if decoupled_min_lr is None:                                               # trace_info : t_14873
        decoupled_min_lr = min_lr                                              # trace_info : t_14874

    for param_group in param_groups:                                           # trace_info : t_14875, t_14879, t_14883
        if param_group['is_decoupled_lr']:                                     # trace_info : t_14876, t_14880
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr                                         # trace_info : t_14877, t_14881
            param_group['min_lr'] = min_lr                                     # trace_info : t_14878, t_14882
    return param_groups                                                        # trace_info : t_14884


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
    if config.optimizer == 'adam':                                             # trace_info : t_14915
        optimizer = Adam(                                                      # trace_info : t_14916, t_14922
            param_groups,                                                      # trace_info : t_14917
            lr=config.lr,                                                      # trace_info : t_14918
            weight_decay=config.weight_decay,                                  # trace_info : t_14919
            betas=(config.adam_beta1, config.adam_beta2),                      # trace_info : t_14920
            eps=config.adam_eps,                                               # trace_info : t_14921
        )

        def init_state_fn(opt):                                                # trace_info : t_14923
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
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:         # trace_info : t_14924

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None                                                     # trace_info : t_14925

        # Constant loss scale.
        if config.loss_scale:                                                  # trace_info : t_14926
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:                                                    # trace_info : t_14927
                grad_scaler = DynamicGradScaler(                               # trace_info : t_14928, t_14935
                    initial_scale=config.initial_loss_scale,                   # trace_info : t_14929
                    min_scale=config.min_loss_scale,                           # trace_info : t_14930
                    growth_factor=2.0,                                         # trace_info : t_14931
                    backoff_factor=0.5,                                        # trace_info : t_14932
                    growth_interval=config.loss_scale_window,                  # trace_info : t_14933
                    hysteresis=config.hysteresis,                              # trace_info : t_14934
                )

        optimizer_args = [                                                     # trace_info : t_14957
            optimizer,                                                         # trace_info : t_14953
            config,                                                            # trace_info : t_14954
            grad_scaler,                                                       # trace_info : t_14955
            init_state_fn,                                                     # trace_info : t_14956
        ]
        if config.use_distributed_optimizer:                                   # trace_info : t_14958
            optimizer = DistributedOptimizer(
                *optimizer_args,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)     # trace_info : t_14959

        return optimizer                                                       # trace_info : t_15669

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

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_14395
        logger.info(f'Setting up optimizer with config {config}')              # trace_info : t_14396

    # Collect param groups.
    param_groups = _get_param_groups(                                          # trace_info : t_14402, t_14408
        model_chunks,                                                          # trace_info : t_14403
        no_weight_decay_cond,                                                  # trace_info : t_14404
        scale_lr_cond,                                                         # trace_info : t_14405
        lr_mult,                                                               # trace_info : t_14406
        use_decoupled_learning_rate=config.decoupled_lr is not None,           # trace_info : t_14407
    )
    param_groups = _update_min_and_max_lr_in_param_groups(                     # trace_info : t_14866, t_14872
        param_groups,                                                          # trace_info : t_14867
        lr=config.lr,                                                          # trace_info : t_14868
        min_lr=config.min_lr,                                                  # trace_info : t_14869
        decoupled_lr=config.decoupled_lr,                                      # trace_info : t_14870
        decoupled_min_lr=config.decoupled_min_lr,                              # trace_info : t_14871
    )

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}                                                     # trace_info : t_14885
    per_model_ep_buffers = {}                                                  # trace_info : t_14886
    for model_idx, model_chunk in enumerate(model_chunks):                     # trace_info : t_14887, t_14891
        if hasattr(model_chunk, 'buffers'):                                    # trace_info : t_14888
            per_model_buffers[model_idx] = model_chunk.buffers                 # trace_info : t_14889
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers# trace_info : t_14890

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))# trace_info : t_14892, t_14893, t_14894
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))# trace_info : t_14895, t_14896, t_14897

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())# trace_info : t_14898
    optimizers = [                                                             # trace_info : t_15670
        _get_megatron_optimizer_based_on_param_groups(                         # trace_info : t_14901, t_14914
            config,                                                            # trace_info : t_14902
            param_groups=dense_param_groups,                                   # trace_info : t_14903
            per_model_buffers=per_model_buffers,                               # trace_info : t_14904
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),# trace_info : t_14905
            data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),# trace_info : t_14909
            data_parallel_group_idx=model_parallel_rank,                       # trace_info : t_14913
        )
    ]
    if len(moe_param_groups) > 0:                                              # trace_info : t_15671
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

    if len(optimizers) == 1:                                                   # trace_info : t_15672
        return optimizers[0]                                                   # trace_info : t_15673

    return ChainedOptimizer(optimizers)
