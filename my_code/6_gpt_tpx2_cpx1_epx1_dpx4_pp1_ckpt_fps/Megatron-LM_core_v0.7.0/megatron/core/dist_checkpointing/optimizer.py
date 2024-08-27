# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Helpers for defining sharding for optimizer states based on existing sharding for model parameters. """

import logging
from copy import deepcopy
from dataclasses import replace
from itertools import chain
from typing import Dict, Iterable, List, Tuple, Union

logger = logging.getLogger(__name__)

import torch

from .dict_utils import nested_values
from .mapping import (
    LocalNonpersitentObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)
from .utils import extract_sharded_tensors_and_factories


def get_optim_param_to_id_map(optim_params_iter: Iterable[torch.nn.Parameter]) -> Dict[int, int]:
    param_mappings = {}                                                        # trace_info : t_29342, t_96935
    for i, param in enumerate(optim_params_iter):                              # trace_info : t_29343, t_29347, t_29350, t_29353, t_29356, ...
        if id(param) not in param_mappings:                                    # trace_info : t_29345, t_29348, t_29351, t_29354, t_29357, ...
            param_mappings[id(param)] = i                                      # trace_info : t_29346, t_29349, t_29352, t_29355, t_29358, ...
    return param_mappings                                                      # trace_info : t_29431, t_97024


def get_param_id_to_sharded_param_map(
    model_sharded_state_dict: ShardedStateDict, optim_params_iter: Iterable[torch.nn.Parameter]
) -> Dict[int, Union[ShardedTensor, ShardedTensorFactory]]:
    """ Generate mapping from optimizer state ids to model sharded parameters.

    Args:
        model_sharded_state_dict: sharded state dict with all model sharded tensors (can have any structure)
        optim_params_iter: iterable which iterates over model parameters tracked by the optimizer.
            The iteration must be in the same order as in the optimizer parameters.

    Returns:
        Dict[int, Union[ShardedTensor, ShardedTensorFactory]]: mapping from optimizer state ids
            to model sharded parameters.
    """
    model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(model_sharded_state_dict)# trace_info : t_29122, t_96715
    id_to_sharded_param_map = {}                                               # trace_info : t_29340, t_96933
    param_to_id_map = get_optim_param_to_id_map(optim_params_iter)             # trace_info : t_29341, t_96934
    for ten in nested_values(model_sharded_state_dict):                        # trace_info : t_29432, t_29443, t_29449, t_29455, t_29461, ...
        if id(ten.data) in param_to_id_map:                                    # trace_info : t_29441, t_29447, t_29453, t_29459, t_29465, ...
            id_to_sharded_param_map[param_to_id_map[id(ten.data)]] = ten       # trace_info : t_29442, t_29448, t_29454, t_29460, t_29466, ...
        else:
            logger.debug(f'{ten} is not tracked by the optimizer')

    if not id_to_sharded_param_map:                                            # trace_info : t_29608, t_97201
        logger.warning(
            "Sharded parameters mapping is empty. It means tensors in model state dict"
            " do not correspond to tensors in optimizer parameters map."
            " Make sure to call state_dict with `keep_vars=True`."
        )
    return id_to_sharded_param_map                                             # trace_info : t_29609, t_97202


def make_sharded_optimizer_tensor(
    model_param: Union[ShardedTensor, ShardedTensorFactory], optim_param: torch.Tensor, prefix: str
) -> Union[ShardedTensor, ShardedTensorFactory]:
    """ Build a ShardedTensor or ShardedTensorFactory for optimizer param based on model param

    Args:
        model_param (Union[ShardedTensor, ShardedTensorFactory]): model param
        optim_param (torch.Tensor): corresponding optimizer param
        prefix (str): optimizer prefix for the ShardedTensor or ShardedTensorFactory

    Returns:
        Union[ShardedTensor, ShardedTensorFactory]: wrapped optimizer parameter
    """
    if isinstance(model_param, ShardedTensorFactory):                          # trace_info : t_29618, t_29634, t_29650, t_29666, t_29682, ...
        return replace(model_param, key=f'{prefix}.{model_param.key}', data=optim_param)

    assert (
        tuple(optim_param.shape) == model_param.local_shape                    # trace_info : t_29619, t_29635, t_29651, t_29667, t_29683, ...
    ), f'Optimizer shape ({tuple(optim_param.shape)} does not match model shape ({model_param.local_shape})'
    return replace(                                                            # trace_info : t_29620, t_29622, t_29636, t_29638, t_29652, ...
        model_param, key=f'{prefix}.{model_param.key}', data=optim_param, dtype=optim_param.dtype# trace_info : t_29621, t_29637, t_29653, t_29669, t_29685, ...
    )


def optim_state_to_sharding_state(
    optim_state_dict: StateDict,
    id_to_sharded_param_map: Dict[int, ShardedTensor],
    exclude_keys: Tuple[str] = (),
):
    """ Turn optimizer state dict to sharded state dict based on model state dict *in-place*.

    Can be used to add sharding information to most common optimizer state dict.
    Creates separate ShardedTensors for each key in `optim_state_dict['state']`
    (e.g. for torch.optim.Adam there will be separate tensors for `exp_avg` and `exp_avg_sq`)

    Args:
        optim_state_dict (StateDict): optimizer state dict with
            state parameters under `state` key and group hyperparameters under `param_groups` -> `params` key.
        id_to_sharded_param_map (Dict[int, ShardedTensor]): mapping from optimizer param ids to model sharded tensors.
            Can be generated with `get_param_id_to_sharded_param_map` function
        exclude_keys (Tuple[str]): optimizer state keys to exclude from the final state dict.

    Returns:
        None: state dict is modified in place
    """
    sharded_state = {}                                                         # trace_info : t_30067, t_97660
    for param_id, param_state in optim_state_dict['state'].items():            # trace_info : t_30068, t_30115, t_30162, t_30209, t_30256, ...
        sharded_state[param_id] = {}                                           # trace_info : t_30069, t_30116, t_30163, t_30210, t_30257, ...
        for state_key, param in param_state.items():                           # trace_info : t_30070, t_30092, t_30114, t_30117, t_30139, ...
            if state_key in exclude_keys:                                      # trace_info : t_30071, t_30093, t_30118, t_30140, t_30165, ...
                continue
            if param_id in id_to_sharded_param_map:                            # trace_info : t_30072, t_30094, t_30119, t_30141, t_30166, ...
                sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(# trace_info : t_30073, t_30075, t_30095, t_30097, t_30120, ...
                    id_to_sharded_param_map[param_id], param, prefix=f'optimizer.state.{state_key}'# trace_info : t_30074, t_30096, t_30121, t_30143, t_30168, ...
                )
            else:
                raise ValueError(f'Param id {param_id} does not match any model sharded param')

    optim_state_dict['param_groups'] = deepcopy(optim_state_dict['param_groups'])# trace_info : t_31385, t_98978
    for group in optim_state_dict['param_groups']:                             # trace_info : t_31386, t_31389, t_31392, t_98979, t_98982, ...
        group['params'] = LocalNonpersitentObject(group['params'])             # trace_info : t_31387, t_31390, t_98980, t_98983
    optim_state_dict['state'] = sharded_state                                  # trace_info : t_31393, t_98986
