# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Utilities for operating with dicts and lists.

All functions in this module handle nesting of dicts and lists.
Other objects (e.g. tuples) are treated as atomic leaf types that cannot be traversed.
"""

from collections import defaultdict
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch


def extract_matching_values(
    x: Union[dict, list], predicate: Callable[[Any], bool], return_lists_as_dicts: bool = False
) -> Tuple[Union[dict, list], Union[dict, list]]:
    """ Return matching and nonmatching values. Keeps hierarchy.

    Args:
        x (Union[dict, list]) : state dict to process. Top-level argument must be a dict or list
        predicate (object -> bool): determines matching values
        return_lists_as_dicts (bool): if True, matching lists will be turned
            into dicts, with keys indicating the indices of original elements.
            Useful for reconstructing the original hierarchy.
    """

    def _set_elem(target, k, v):                                               # trace_info : t_29126, t_29148, t_33006, t_33028, t_33220, ...
        if return_lists_as_dicts:                                              # trace_info : t_33943, t_33947, t_34025, t_34029, t_34091, ...
            target[k] = v
        else:
            target.append(v)                                                   # trace_info : t_33944, t_33948, t_34026, t_34030, t_34092, ...

    if isinstance(x, dict):                                                    # trace_info : t_29127, t_29149, t_33007, t_33029, t_33221, ...
        matching_vals = {}                                                     # trace_info : t_29128, t_29150, t_33008, t_33030, t_33222, ...
        nonmatching_vals = {}                                                  # trace_info : t_29129, t_29151, t_33009, t_33031, t_33223, ...
        for k, v in x.items():                                                 # trace_info : t_29130, t_29135, t_29140, t_29145, t_29152, ...
            if isinstance(v, (list, dict)):                                    # trace_info : t_29131, t_29136, t_29141, t_29146, t_29153, ...
                match, nonmatch = extract_matching_values(v, predicate, return_lists_as_dicts)# trace_info : t_29147, t_33027, t_33219, t_33226, t_33233, ...
                if match:                                                      # trace_info : t_29334, t_33214, t_33257, t_33279, t_33301, ...
                    matching_vals[k] = match                                   # trace_info : t_29335, t_34034, t_34040, t_34312, t_34600, ...
                if nonmatch or not v:                                          # trace_info : t_29336, t_33215, t_33258, t_33280, t_33302, ...
                    nonmatching_vals[k] = nonmatch                             # trace_info : t_29337, t_33216, t_33259, t_33281, t_33303, ...
            elif predicate(v):                                                 # trace_info : t_29132, t_29137, t_29142, t_29154, t_29159, ...
                matching_vals[k] = v                                           # trace_info : t_29156, t_29161, t_29166, t_29171, t_29176, ...
            else:
                nonmatching_vals[k] = v                                        # trace_info : t_29134, t_29139, t_29144, t_29186, t_29201, ...
    elif isinstance(x, list):                                                  # trace_info : t_33864, t_34075, t_34083, t_34168, t_35249, ...
        matching_vals = {} if return_lists_as_dicts else []                    # trace_info : t_33865, t_34076, t_34084, t_34169, t_35250, ...
        nonmatching_vals = {} if return_lists_as_dicts else []                 # trace_info : t_33866, t_34077, t_34085, t_34170, t_35251, ...
        for ind, v in enumerate(x):                                            # trace_info : t_33867, t_33949, t_34031, t_34078, t_34086, ...
            if isinstance(v, (list, dict)) and v:                              # trace_info : t_33868, t_33950, t_34079, t_34087, t_34094, ...
                match, nonmatch = extract_matching_values(v, predicate, return_lists_as_dicts)# trace_info : t_33869, t_33951, t_34080, t_34165, t_35254, ...
                if match:                                                      # trace_info : t_33941, t_34023, t_34158, t_34299, t_35321, ...
                    _set_elem(matching_vals, ind, match)                       # trace_info : t_33942, t_34024, t_35527, t_35668, t_101309, ...
                if nonmatch or not v:                                          # trace_info : t_33945, t_34027, t_34159, t_34300, t_35322, ...
                    _set_elem(nonmatching_vals, ind, nonmatch)                 # trace_info : t_33946, t_34028, t_34160, t_34301, t_35323, ...
            else:
                target = matching_vals if predicate(v) else nonmatching_vals   # trace_info : t_34088, t_34095, t_34102, t_34109, t_34116, ...
                _set_elem(target, ind, v)                                      # trace_info : t_34090, t_34097, t_34104, t_34111, t_34118, ...
    else:
        raise ValueError(f'Unexpected top-level object type: {type(x)}')
    return matching_vals, nonmatching_vals                                     # trace_info : t_29333, t_29339, t_33213, t_33256, t_33278, ...


def diff(x1: Any, x2: Any, prefix: Tuple = ()) -> Tuple[list, list, list]:
    """ Recursive diff of dicts.

    Args:
        x1 (object): left dict
        x2 (object): right dict
        prefix (tuple): tracks recursive calls. Used for reporting differing keys.

    Returns:
        Tuple[list, list, list]: tuple of:
            - only_left: Prefixes present only in left dict
            - only_right: Prefixes present only in right dict
            - mismatch: values present in both dicts but not equal across dicts.
                For tensors equality of all elems is checked.
                Each element is a tuple (prefix, type of left value, type of right value).
    """
    mismatch = []
    if isinstance(x1, dict) and isinstance(x2, dict):
        only_left = [prefix + (k,) for k in x1.keys() - x2.keys()]
        only_right = [prefix + (k,) for k in x2.keys() - x1.keys()]
        for k in x2.keys() & x1.keys():
            _left, _right, _mismatch = diff(x1[k], x2[k], prefix + (k,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    elif isinstance(x1, list) and isinstance(x2, list):
        only_left = list(range(len(x1) - 1, len(x2) - 1, -1))
        only_right = list(range(len(x1) - 1, len(x2) - 1, -1))
        for i, (v1, v2) in enumerate(zip(x1, x2)):
            _left, _right, _mismatch = diff(v1, v2, prefix + (i,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    else:
        only_left = []
        only_right = []
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            _is_mismatch = not torch.all(x1 == x2)
        else:
            try:
                _is_mismatch = bool(x1 != x2)
            except RuntimeError:
                _is_mismatch = True

        if _is_mismatch:
            mismatch.append((prefix, type(x1), type(x2)))

    return only_left, only_right, mismatch


def inspect_types(x: Any, prefix: Tuple = (), indent: int = 4):
    """ Helper to print types of (nested) dict values. """
    print_indent = lambda: print(' ' * indent * len(prefix), end='')
    if isinstance(x, dict):
        print()
        for k, v in x.items():
            print_indent()
            print(f'> {k}: ', end='')
            inspect_types(v, prefix + (k,), indent)
    elif isinstance(x, list):
        print()
        for i, v in enumerate(x):
            print_indent()
            print(f'- {i}: ', end='')
            inspect_types(v, prefix + (i,), indent)
    else:
        if isinstance(x, torch.Tensor):
            print(f'Tensor of shape {x.shape}')
        else:
            try:
                x_str = str(x)
            except:
                x_str = '<no string repr>'
            if len(x_str) > 30:
                x_str = x_str[:30] + '... (truncated)'
            print(f'[{type(x)}]: {x_str}')


def nested_values(x: Union[dict, list]):
    """ Returns iterator over (nested) values of a given dict or list. """
    x_iter = x.values() if isinstance(x, dict) else x                          # trace_info : t_29433, t_29437, t_35764, t_35768, t_35881, ...
    for v in x_iter:                                                           # trace_info : t_29434, t_29438, t_29444, t_29450, t_29456, ...
        if isinstance(v, (dict, list)):                                        # trace_info : t_29435, t_29439, t_29445, t_29451, t_29457, ...
            yield from nested_values(v)                                        # trace_info : t_29436, t_35767, t_35880, t_35884, t_35888, ...
        else:
            yield v                                                            # trace_info : t_29440, t_29446, t_29452, t_29458, t_29464, ...


def nested_items_iter(x: Union[dict, list]):
    """ Returns iterator over (nested) tuples (container, key, value) of a given dict or list. """
    x_iter = x.items() if isinstance(x, dict) else enumerate(x)
    for k, v in x_iter:
        if isinstance(v, (dict, list)):
            yield from nested_items_iter(v)
        else:
            yield x, k, v


def dict_map(f: Callable, d: dict):
    """ `map` equivalent for dicts. """
    for sub_d, k, v in nested_items_iter(d):
        sub_d[k] = f(v)


def dict_map_with_key(f: Callable, d: dict):
    """ `map` equivalent for dicts with a function that accepts tuple (key, value). """
    for sub_d, k, v in nested_items_iter(d):
        sub_d[k] = f(k, v)


def dict_list_map_inplace(f: Callable, x: Union[dict, list]):
    """ Maps dicts and lists *in-place* with a given function. """
    if isinstance(x, dict):                                                    # trace_info : t_26728, t_26731, t_26743, t_26755, t_26767, ...
        for k, v in x.items():                                                 # trace_info : t_26729, t_26741, t_26753, t_26765, t_26777, ...
            x[k] = dict_list_map_inplace(f, v)                                 # trace_info : t_26730, t_26742, t_26754, t_26766, t_26778, ...
    elif isinstance(x, list):                                                  # trace_info : t_26732, t_26744, t_26756, t_26768, t_26780, ...
        x[:] = (dict_list_map_inplace(f, v) for v in x)                        # trace_info : t_32501, t_32502, t_32597, t_32692, t_32726, ...
    else:
        return f(x)                                                            # trace_info : t_26733, t_26745, t_26757, t_26769, t_26781, ...
    return x                                                                   # trace_info : t_26926, t_27077, t_28720, t_28871, t_31953, ...


def dict_list_map_outplace(f: Callable, x: Union[dict, list]):
    """ Maps dicts and lists *out-of-place* with a given function. """
    if isinstance(x, dict):
        return {k: dict_list_map_outplace(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [dict_list_map_outplace(f, v) for v in x]
    else:
        return f(x)


def merge(x1: dict, x2: dict, key: Tuple[str, ...] = ()):
    """ Merges dicts and lists recursively. """
    if isinstance(x1, dict) and isinstance(x2, dict):
        for k, v2 in x2.items():
            if k not in x1:
                x1[k] = v2
            else:
                x1[k] = merge(x1[k], v2, key=key + (k,))
    elif isinstance(x1, list) and isinstance(x2, list):
        if len(x1) != len(x2):
            raise ValueError(
                f'Cannot merge two lists with different lengths ({len(x1)} and {len(x2)}, encountered at level {key})'
            )
        for i, v2 in enumerate(x2):
            x1[i] = merge(x1[i], v2, key=key + (i,))
    else:
        raise ValueError(
            f'Duplicate non-dict and non-list values encountered: `{x1}` and `{x2}` (at level {key})'
        )
    return x1


def map_reduce(
    xs: Iterable,
    key_fn: Callable = lambda x: x,
    value_fn: Callable = lambda x: x,
    reduce_fn: Callable = lambda x: x,
) -> dict:
    """ Simple map-reduce implementation following `more_itertools.map_reduce` interface. """
    res = defaultdict(list)
    for x in xs:
        res[key_fn(x)].append(value_fn(x))
    for k in res:
        res[k] = reduce_fn(res[k])
    return dict(res)
