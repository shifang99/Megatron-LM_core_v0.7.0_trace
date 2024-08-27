# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utilities for transformer layers."""
from functools import lru_cache
from operator import itemgetter
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedStateDict, StateDict
from megatron.core.jit import jit_fuser
from megatron.core.utils import (
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)


def get_linear_layer(rows, columns, init_method, perform_initialization=True):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if perform_initialization:  # Take from modelparallel config
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


@lru_cache(maxsize=32)
def get_default_causal_mask(sq: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


@jit_fuser
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@jit_fuser
def erf_gelu(x):
    return (
        x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
    )


def make_sharded_tensors_for_checkpoint(
    state_dict: StateDict,
    prefix: str,
    tensor_parallel_layers_axis_map: Optional[Dict[str, int]] = None,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    extra_state_suffix: str = '_extra_state',
):
    """Wraps tensors from transformer layers with ShardedTensor or ShardedObject.

    For a given `state_dict`, wraps:
    - all _extra_states with ShardedObject
    - all tensors specified in tensor_parallel_layers_axis_map with TP and DP sharded ShardedTensor
    - other values with DP sharded ShardedTensor

    Args:
        state_dict (StateDict): state_dict to convert
        prefix (str): prefix appended to keys in final state dict
        tensor_parallel_layers_axis_map (Dict[str, int], optional): dict mapping layer
            names to the axis for TP sharding
        sharded_offsets (Iterable[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related), passed along to ShardedTensor
        extra_state_suffix (str, default = '_extra_state'): layers with this
            suffix will be wrapped with ShardedObject instead of ShardedTensor.

    """

    if tensor_parallel_layers_axis_map is None:                                # trace_info : t_25061, t_25078, t_25183, t_25263, t_25322, ...
        tensor_parallel_layers_axis_map = {}                                   # trace_info : t_25062, t_25079, t_25323, t_25504, t_25521, ...

    sharded_state_dict = {}                                                    # trace_info : t_25063, t_25080, t_25184, t_25264, t_25324, ...
    for layer_name in state_dict.keys():                                       # trace_info : t_25064, t_25081, t_25185, t_25251, t_25265, ...
        tensor = state_dict[layer_name]                                        # trace_info : t_25186, t_25338, t_25413, t_25572, t_25664, ...
        layer_key = f'{prefix}{layer_name}'                                    # trace_info : t_25187, t_25339, t_25414, t_25573, t_25665, ...

        if layer_name.endswith(extra_state_suffix):                            # trace_info : t_25188, t_25340, t_25415, t_25574, t_25666, ...
            sharded_state_dict[layer_key] = make_sharded_object_for_checkpoint(# trace_info : t_25742, t_25744, t_25981, t_25983, t_26462, ...
                tensor, layer_key, sharded_offsets                             # trace_info : t_25743, t_25982, t_26463, t_26680, t_27537, ...
            )

        elif layer_name in tensor_parallel_layers_axis_map:                    # trace_info : t_25189, t_25341, t_25416, t_25575, t_25667, ...
            tp_axis = tensor_parallel_layers_axis_map[layer_name]              # trace_info : t_25576, t_25798, t_25890, t_26279, t_26371, ...
            sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(# trace_info : t_25577, t_25579, t_25799, t_25801, t_25891, ...
                tensor, layer_key, tp_axis, prepend_offsets=sharded_offsets,   # trace_info : t_25578, t_25800, t_25892, t_26281, t_26373, ...
            )

        else:
            sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(# trace_info : t_25190, t_25192, t_25342, t_25344, t_25417, ...
                tensor, layer_key, prepend_offsets=sharded_offsets,            # trace_info : t_25191, t_25343, t_25418, t_25669, t_26108, ...
            )

    return sharded_state_dict                                                  # trace_info : t_25065, t_25082, t_25252, t_25266, t_25326, ...


def make_sharded_object_for_checkpoint(
    obj: Any,
    key: str,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    replica_id: Union[None, int, Tuple[int, ...]] = None,
    **kwargs,
):
    """ Helper for instantiating a non-sharded ShardedObject (replicated across TP and DP group).

    Args:
        obj (object): any object to be sharded
        key (str): unique identifier of the object
        sharded_offsets (Iterable[Tuple[int, int, int]]): offsets normally
            prepended to ShardedTensors, will be used as global offsets for
            ShardedObject
        replica_id (Union[None, int, Tuple[int, ...]]): replica id
    """
    if replica_id is None:                                                     # trace_info : t_25745, t_25984, t_26465, t_26682, t_27539, ...
        replica_id = (                                                         # trace_info : t_25761, t_26000, t_26481, t_26698, t_27555, ...
            0,                                                                 # trace_info : t_25746, t_25985, t_26466, t_26683, t_27540, ...
            parallel_state.get_tensor_model_parallel_rank(),                   # trace_info : t_25747, t_25986, t_26467, t_26684, t_27541, ...
            parallel_state.get_data_parallel_rank(with_context_parallel=True), # trace_info : t_25753, t_25992, t_26473, t_26690, t_27547, ...
        )

    return ShardedObject(key, obj, *_get_extra_state_offsets(sharded_offsets), replica_id, **kwargs)# trace_info : t_25762, t_26001, t_26482, t_26699, t_27556, ...


def _get_extra_state_offsets(
    sharded_offsets: Iterable[Tuple[int, int, int]]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ Turns ShardedTensor offsets into offsets suitable for ShardedObject. """
    if sharded_offsets:                                                        # trace_info : t_25763, t_26002, t_26483, t_26700, t_27557, ...
        sharded_offsets = sorted(sharded_offsets, key=itemgetter(0))  # sort by axis# trace_info : t_25764, t_26003, t_26484, t_26701, t_27558, ...
        axis, extra_state_offset, extra_state_shape = zip(*sharded_offsets)    # trace_info : t_25765, t_26004, t_26485, t_26702, t_27559, ...
        assert list(axis) == list(                                             # trace_info : t_25766, t_25768, t_26005, t_26007, t_26486, ...
            range(len(axis))                                                   # trace_info : t_25767, t_26006, t_26487, t_26704, t_27561, ...
        ), f'Expected contiguous axis for offsets: {sharded_offsets}'
    else:
        extra_state_shape = (1,)                                               # trace_info : t_29069, t_96662
        extra_state_offset = (0,)                                              # trace_info : t_29070, t_96663
    return extra_state_shape, extra_state_offset                               # trace_info : t_25769, t_26008, t_26489, t_26706, t_27563, ...


def sharded_state_dict_default(
    module: torch.nn.Module,
    prefix: str = '',
    sharded_offsets: Tuple[Tuple[int, int, int]] = (),
    metadata: Optional[dict] = None,
) -> ShardedStateDict:
    """Provides implementation for sharded_state_dict method for non-MegatronModules.

    Tries to call `module.sharded_state_dict` when possible,
    otherwise uses regular state dict and assumes tensors are replicated across TP and DP.

    `keep_vars=True` is passed to module.state_dict so that optimizer states
    can be sharded later on.

    Args:
        module (torch.nn.Module): module which sharded state dict we want to obtain
        prefix (str): prefix for the state dict keys
        sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
        metadata (dict, optional): metadata passed to module sharded_state_dict method

    Returns:
        dict: dictionary of state dict keys mapped to ShardedTensors
    """

    if hasattr(module, 'sharded_state_dict'):                                  # trace_info : t_25069, t_25086, t_25178, t_25258, t_25276, ...
        module_sharded_sd = module.sharded_state_dict(                         # trace_info : t_25070, t_25072, t_25087, t_25089, t_25277, ...
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata  # trace_info : t_25071, t_25088, t_25278, t_25496, t_25513, ...
        )
    else:
        module_sd = module.state_dict(prefix='', keep_vars=True)               # trace_info : t_25179, t_25259, t_25331, t_25529, t_25543, ...
        module_sharded_sd = make_sharded_tensors_for_checkpoint(               # trace_info : t_25180, t_25182, t_25260, t_25262, t_25332, ...
            module_sd, prefix, {}, sharded_offsets,                            # trace_info : t_25181, t_25261, t_25333, t_25531, t_25545, ...
        )
    return module_sharded_sd                                                   # trace_info : t_25173, t_25253, t_25267, t_25271, t_25489, ...
