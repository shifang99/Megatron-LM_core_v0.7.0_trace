# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import types                                                                   # trace_info : t_9651, t_9674, t_9693
from dataclasses import dataclass, field                                       # trace_info : t_9652, t_9675, t_9694
from typing import Tuple, Union                                                # trace_info : t_9654, t_9676, t_9696

import torch


@dataclass
class ModuleSpec:
    """This is a Module Specification dataclass.

    Specification defines the location of the module (to import dynamically)
    or the imported module itself. It also defines the params that need to be
    passed to initialize the module.

    Args:
        module (Union[Tuple, type]): A tuple describing the location of the
            module class e.g. `(module.location, ModuleClass)` or the imported
            module class itself e.g. `ModuleClass` (which is already imported
            using `from module.location import ModuleClass`).
        params (dict): A dictionary of params that need to be passed while init.

    """

    module: Union[Tuple, type]
    params: dict = field(default_factory=lambda: {})                           # trace_info : t_9653, t_9695
    submodules: type = None


def import_module(module_path: Tuple[str]):
    """Import a named object from a module in the context of this function.

    TODO: make this importer module more robust, at least make sure there
    are no side effects of using this as is
    """
    base_path, name = module_path
    try:
        module = __import__(base_path, globals(), locals(), [name])
    except ImportError as e:
        print(f"couldn't import module due to {e}")
        return None
    return vars(module)[name]


def get_module(spec_or_module: Union[ModuleSpec, type], **additional_kwargs):
    # If a module clas is already provided return it as is
    if isinstance(spec_or_module, (type, types.FunctionType)):
        return spec_or_module

    # If the module is provided instead of module path, then return it as is
    if isinstance(spec_or_module.module, (type, types.FunctionType)):
        return spec_or_module.module

    # Otherwise, return the dynamically imported module from the module path
    return import_module(spec_or_module.module)


def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):
    # If the passed `spec_or_module` is
    # a `Function`, then return it as it is
    # NOTE: to support an already initialized module add the following condition
    # `or isinstance(spec_or_module, torch.nn.Module)` to the following if check
    if isinstance(spec_or_module, types.FunctionType):                         # trace_info : t_9902, t_9952, t_9970, t_10031, t_10110, ...
        return spec_or_module                                                  # trace_info : t_10357, t_10678, t_11135, t_11456

    # If the passed `spec_or_module` is actually a spec (instance of
    # `ModuleSpec`) and it specifies a `Function` using its `module`
    # field, return the `Function` as it is
    if isinstance(spec_or_module, ModuleSpec) and isinstance(                  # trace_info : t_9903, t_9905, t_9953, t_9971, t_9973, ...
        spec_or_module.module, types.FunctionType                              # trace_info : t_9904, t_9972, t_10438, t_10683, t_10751, ...
    ):
        return spec_or_module.module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):                                       # trace_info : t_9906, t_9954, t_9974, t_10033, t_10112, ...
        module = spec_or_module                                                # trace_info : t_9955, t_10034, t_10113, t_10223, t_10321, ...
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):# trace_info : t_9907, t_9975, t_10441, t_10686, t_10754, ...
        module = spec_or_module.module                                         # trace_info : t_9908, t_9976, t_10442, t_10687, t_10755, ...
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module)

    # If the imported module is actually a `Function` return it as it is
    if isinstance(module, types.FunctionType):                                 # trace_info : t_9909, t_9956, t_9977, t_10035, t_10114, ...
        return module

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code

    # Add the `submodules` argument to the module init call if it exists in the
    # spec.
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:# trace_info : t_9910, t_9957, t_9978, t_10036, t_10115, ...
        kwargs["submodules"] = spec_or_module.submodules                       # trace_info : t_9911, t_9979, t_10445, t_10690, t_10758, ...

    try:                                                                       # trace_info : t_9912, t_9958, t_9980, t_10037, t_10116, ...
        return module(                                                         # trace_info : t_9913, t_9915, t_9917, t_9919, t_9959, ...
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs# trace_info : t_9914, t_9916, t_9918, t_9960, t_9962, ...
        )
    except Exception as e:
        # improve the error message since we hide the module name in the line above
        import sys

        tb = sys.exc_info()[2]
        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )
