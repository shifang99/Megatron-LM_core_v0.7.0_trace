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
    params_map = {}                                                            # trace_info : t_16417
    for model_chunk in model_chunks:                                           # trace_info : t_16418, t_16850
        for name, param in model_chunk.named_parameters():                     # trace_info : t_16419, t_16434, t_16448, t_16465, t_16481, ...
            if not param.requires_grad:                                        # trace_info : t_16420, t_16435, t_16449, t_16466, t_16482, ...
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)         # trace_info : t_16421, t_16436, t_16450, t_16467, t_16483, ...

            if no_weight_decay_cond is not None:                               # trace_info : t_16422, t_16437, t_16451, t_16468, t_16484, ...
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1        # trace_info : t_16423, t_16438, t_16452, t_16469, t_16485, ...

            if scale_lr_cond is not None:                                      # trace_info : t_16424, t_16439, t_16453, t_16470, t_16486, ...
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False                                               # trace_info : t_16425, t_16440, t_16454, t_16471, t_16487, ...

            if not no_wd and not scale_lr:                                     # trace_info : t_16426, t_16441, t_16455, t_16472, t_16488, ...
                wd_mult, lr_mult = 1.0, 1.0                                    # trace_info : t_16427, t_16442, t_16489, t_16519, t_16581, ...
            elif not no_wd and scale_lr:                                       # trace_info : t_16456, t_16473, t_16503, t_16533, t_16549, ...
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:                                       # trace_info : t_16457, t_16474, t_16504, t_16534, t_16550, ...
                wd_mult, lr_mult = 0.0, 1.0                                    # trace_info : t_16458, t_16475, t_16505, t_16535, t_16551, ...
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            is_decoupled_lr = False                                            # trace_info : t_16428, t_16443, t_16459, t_16476, t_16490, ...
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(                        # trace_info : t_16429, t_16444, t_16460, t_16477, t_16491, ...
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)      # trace_info : t_16430, t_16445, t_16461, t_16478, t_16492, ...
            if key not in params_map:                                          # trace_info : t_16431, t_16446, t_16462, t_16479, t_16493, ...
                params_map[key] = []                                           # trace_info : t_16432, t_16463
            params_map[key].append(param)                                      # trace_info : t_16433, t_16447, t_16464, t_16480, t_16494, ...

    param_groups = []                                                          # trace_info : t_16851
    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():# trace_info : t_16852, t_16862, t_16872
        assert len(params) > 0                                                 # trace_info : t_16853, t_16863
        param_groups.append(                                                   # trace_info : t_16854, t_16861, t_16864, t_16871
            {                                                                  # trace_info : t_16860, t_16870
                'params': params,                                              # trace_info : t_16855, t_16865
                'wd_mult': wd_mult,                                            # trace_info : t_16856, t_16866
                'lr_mult': lr_mult,                                            # trace_info : t_16857, t_16867
                'is_expert_parallel': is_expert_parallel,                      # trace_info : t_16858, t_16868
                'is_decoupled_lr': is_decoupled_lr,                            # trace_info : t_16859, t_16869
            }
        )

    return param_groups                                                        # trace_info : t_16873


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

    if decoupled_min_lr is None:                                               # trace_info : t_16881
        decoupled_min_lr = min_lr                                              # trace_info : t_16882

    for param_group in param_groups:                                           # trace_info : t_16883, t_16887, t_16891
        if param_group['is_decoupled_lr']:                                     # trace_info : t_16884, t_16888
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr                                         # trace_info : t_16885, t_16889
            param_group['min_lr'] = min_lr                                     # trace_info : t_16886, t_16890
    return param_groups                                                        # trace_info : t_16892


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
    if config.optimizer == 'adam':                                             # trace_info : t_16923
        optimizer = Adam(                                                      # trace_info : t_16924, t_16930
            param_groups,                                                      # trace_info : t_16925
            lr=config.lr,                                                      # trace_info : t_16926
            weight_decay=config.weight_decay,                                  # trace_info : t_16927
            betas=(config.adam_beta1, config.adam_beta2),                      # trace_info : t_16928
            eps=config.adam_eps,                                               # trace_info : t_16929
        )

        def init_state_fn(opt):                                                # trace_info : t_16931
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
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:         # trace_info : t_16932

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None                                                     # trace_info : t_16933

        # Constant loss scale.
        if config.loss_scale:                                                  # trace_info : t_16934
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:                                                    # trace_info : t_16935
                grad_scaler = DynamicGradScaler(                               # trace_info : t_16936, t_16943
                    initial_scale=config.initial_loss_scale,                   # trace_info : t_16937
                    min_scale=config.min_loss_scale,                           # trace_info : t_16938
                    growth_factor=2.0,                                         # trace_info : t_16939
                    backoff_factor=0.5,                                        # trace_info : t_16940
                    growth_interval=config.loss_scale_window,                  # trace_info : t_16941
                    hysteresis=config.hysteresis,                              # trace_info : t_16942
                )

        optimizer_args = [                                                     # trace_info : t_16965
            optimizer,                                                         # trace_info : t_16961
            config,                                                            # trace_info : t_16962
            grad_scaler,                                                       # trace_info : t_16963
            init_state_fn,                                                     # trace_info : t_16964
        ]
        if config.use_distributed_optimizer:                                   # trace_info : t_16966
            optimizer = DistributedOptimizer(
                *optimizer_args,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
            )
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)     # trace_info : t_16967

        return optimizer                                                       # trace_info : t_17677

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

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_16403
        logger.info(f'Setting up optimizer with config {config}')              # trace_info : t_16404

    # Collect param groups.
    param_groups = _get_param_groups(                                          # trace_info : t_16410, t_16416
        model_chunks,                                                          # trace_info : t_16411
        no_weight_decay_cond,                                                  # trace_info : t_16412
        scale_lr_cond,                                                         # trace_info : t_16413
        lr_mult,                                                               # trace_info : t_16414
        use_decoupled_learning_rate=config.decoupled_lr is not None,           # trace_info : t_16415
    )
    param_groups = _update_min_and_max_lr_in_param_groups(                     # trace_info : t_16874, t_16880
        param_groups,                                                          # trace_info : t_16875
        lr=config.lr,                                                          # trace_info : t_16876
        min_lr=config.min_lr,                                                  # trace_info : t_16877
        decoupled_lr=config.decoupled_lr,                                      # trace_info : t_16878
        decoupled_min_lr=config.decoupled_min_lr,                              # trace_info : t_16879
    )

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}                                                     # trace_info : t_16893
    per_model_ep_buffers = {}                                                  # trace_info : t_16894
    for model_idx, model_chunk in enumerate(model_chunks):                     # trace_info : t_16895, t_16899
        if hasattr(model_chunk, 'buffers'):                                    # trace_info : t_16896
            per_model_buffers[model_idx] = model_chunk.buffers                 # trace_info : t_16897
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers# trace_info : t_16898

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))# trace_info : t_16900, t_16901, t_16902
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))# trace_info : t_16903, t_16904, t_16905

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())# trace_info : t_16906
    optimizers = [                                                             # trace_info : t_17678
        _get_megatron_optimizer_based_on_param_groups(                         # trace_info : t_16909, t_16922
            config,                                                            # trace_info : t_16910
            param_groups=dense_param_groups,                                   # trace_info : t_16911
            per_model_buffers=per_model_buffers,                               # trace_info : t_16912
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),# trace_info : t_16913
            data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),# trace_info : t_16917
            data_parallel_group_idx=model_parallel_rank,                       # trace_info : t_16921
        )
    ]
    if len(moe_param_groups) > 0:                                              # trace_info : t_17679
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

    if len(optimizers) == 1:                                                   # trace_info : t_17680
        return optimizers[0]                                                   # trace_info : t_17681

    return ChainedOptimizer(optimizers)
