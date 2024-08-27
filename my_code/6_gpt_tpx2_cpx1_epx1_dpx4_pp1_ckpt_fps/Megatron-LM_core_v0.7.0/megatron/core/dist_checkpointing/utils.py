# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Helpers for manipulating sharded tensors and sharded state dicts. """

from typing import Dict, Tuple

from .dict_utils import dict_list_map_inplace, extract_matching_values
from .mapping import (
    LocalNonpersitentObject,
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)


def extract_sharded_tensors(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """ Extract a dict consisting of only ShardedTensor objects from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor (keeping the original state dict structure)
            - state dict with all objects other than ShardedTensor (keeping the original state dict structure)
    """
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedTensor))


def extract_sharded_tensors_and_factories(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """ Extract a dict consisting of only ShardedTensor and ShardedTensorFactory objects from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor and ShardedTensorFactory objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor and ShardedTensorFactory (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(                                            # trace_info : t_29123, t_29125, t_96716, t_96718
        sharded_state_dict, lambda v: isinstance(v, (ShardedTensor, ShardedTensorFactory))# trace_info : t_29124, t_29133, t_29138, t_29143, t_29155, ...
    )


def extract_sharded_tensors_or_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """ Extract a dict consisting of only ShardedTensor, ShardedTensorFactory and LocalNonpersitentObject
    objects from a given state dict with any objects.

    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor, ShardedTensorFactory and LocalNonpersitentObject objects

    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor, ShardedTensorFactory and LocalNonpersitentObject (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict,
        lambda v: isinstance(v, (ShardedTensor, LocalNonpersitentObject, ShardedTensorFactory)),
    )


def extract_sharded_base(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedBase),)# trace_info : t_34390, t_34398, t_34403, t_34408, t_34420, ...


def extract_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    return extract_matching_values(                                            # trace_info : t_33003, t_33005, t_100370, t_100372
        sharded_state_dict, lambda v: isinstance(v, LocalNonpersitentObject),  # trace_info : t_33004, t_33013, t_33018, t_33023, t_33035, ...
    )


def add_prefix_for_sharding(sharded_state_dict: ShardedStateDict, prefix: str):
    """ Prepend a given prefix to all ShardedBase objects in a given state dict *in-place*.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict
        prefix (str): prefix to be prepended

    Returns:
        None: state dict is modified in-place
    """

    def add_prefix(t):
        if isinstance(t, ShardedBase):
            t.key = f'{prefix}{t.key}'
        return t

    dict_list_map_inplace(add_prefix, sharded_state_dict)


def replace_prefix_for_sharding(
    sharded_state_dict: ShardedStateDict, old_prefix: str, new_prefix: str
):
    """ Replaces the given prefix in *all* sharded keys in a given state dict.

    Errors out if some key does not begin with a given prefix.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to replace keys in
        old_prefix (str): prefix to be replaced in each key
        new_prefix (str): new prefix

    Returns:
        None: state dict is modified in place
    """

    def _replace_prefix(x):                                                    # trace_info : t_26929, t_28723, t_94522, t_96316
        if isinstance(x, (ShardedTensor, ShardedTensorFactory, ShardedObject)):# trace_info : t_26937, t_26946, t_26955, t_26964, t_26973, ...
            if not x.key.startswith(old_prefix):                               # trace_info : t_26938, t_26947, t_26956, t_26965, t_26974, ...
                raise ValueError(f'Expected {x.key} to begin with prefix {old_prefix}')
            x.key = f'{new_prefix}{x.key[len(old_prefix):]}'  # str.removeprefix in Python >= 3.9# trace_info : t_26939, t_26948, t_26957, t_26966, t_26975, ...
        return x                                                               # trace_info : t_26940, t_26949, t_26958, t_26967, t_26976, ...

    dict_list_map_inplace(_replace_prefix, sharded_state_dict)                 # trace_info : t_26930, t_28724, t_94523, t_96317


def apply_prefix_mapping(sharded_state_dict: ShardedStateDict, prefix_map: Dict[str, str]):
    """ Replaces prefixes *only in keys matching* with one of prefixes in the map.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to replace keys in
        prefix_map (Dict[str, str]): map of old->new prefixes. The first matching prefix for each key is used

    Returns:
        None: state dict is modified in place
    """

    def _replace_prefixes(x):                                                  # trace_info : t_26726, t_28520, t_94319, t_96113
        if not isinstance(x, (ShardedTensor, ShardedTensorFactory, ShardedObject)):# trace_info : t_26734, t_26746, t_26758, t_26770, t_26782, ...
            return x
        for old_prefix, new_prefix in prefix_map.items():                      # trace_info : t_26735, t_26747, t_26759, t_26761, t_26763, ...
            if x.key.startswith(old_prefix):                                   # trace_info : t_26736, t_26748, t_26760, t_26762, t_26772, ...
                x.key = (                                                      # trace_info : t_26738, t_26750, t_26836, t_26850, t_28532, ...
                    f'{new_prefix}{x.key[len(old_prefix):]}'  # str.removeprefix in Python >= 3.9# trace_info : t_26737, t_26749, t_26835, t_26849, t_28531, ...
                )
                break                                                          # trace_info : t_26739, t_26751, t_26837, t_26851, t_28533, ...
        return x                                                               # trace_info : t_26740, t_26752, t_26764, t_26776, t_26788, ...

    dict_list_map_inplace(_replace_prefixes, sharded_state_dict)               # trace_info : t_26727, t_28521, t_94320, t_96114
