# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import types                                                                   # trace_info : t_9707, t_9731, t_9753
from dataclasses import dataclass, field                                       # trace_info : t_9708, t_9732, t_9754
from typing import Tuple, Union                                                # trace_info : t_9710, t_9733, t_9756

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
    params: dict = field(default_factory=lambda: {})                           # trace_info : t_9709, t_9755
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
    if isinstance(spec_or_module, types.FunctionType):                         # trace_info : t_9962, t_10012, t_10051, t_10112, t_10200, ...
        return spec_or_module                                                  # trace_info : t_10533, t_10961, t_11535, t_11963

    # If the passed `spec_or_module` is actually a spec (instance of
    # `ModuleSpec`) and it specifies a `Function` using its `module`
    # field, return the `Function` as it is
    if isinstance(spec_or_module, ModuleSpec) and isinstance(                  # trace_info : t_9963, t_9965, t_10013, t_10052, t_10054, ...
        spec_or_module.module, types.FunctionType                              # trace_info : t_9964, t_10053, t_10635, t_10966, t_11055, ...
    ):
        return spec_or_module.module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):                                       # trace_info : t_9966, t_10014, t_10055, t_10114, t_10202, ...
        module = spec_or_module                                                # trace_info : t_10015, t_10115, t_10203, t_10344, t_10497, ...
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):# trace_info : t_9967, t_10056, t_10638, t_10969, t_11058, ...
        module = spec_or_module.module                                         # trace_info : t_9968, t_10057, t_10639, t_10970, t_11059, ...
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module)

    # If the imported module is actually a `Function` return it as it is
    if isinstance(module, types.FunctionType):                                 # trace_info : t_9969, t_10016, t_10058, t_10116, t_10204, ...
        return module

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code

    # Add the `submodules` argument to the module init call if it exists in the
    # spec.
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:# trace_info : t_9970, t_10017, t_10059, t_10117, t_10205, ...
        kwargs["submodules"] = spec_or_module.submodules                       # trace_info : t_9971, t_10060, t_10642, t_10973, t_11062, ...

    try:                                                                       # trace_info : t_9972, t_10018, t_10061, t_10118, t_10206, ...
        return module(                                                         # trace_info : t_9973, t_9975, t_9977, t_9979, t_10019, ...
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs# trace_info : t_9974, t_9976, t_9978, t_10020, t_10022, ...
        )
    except Exception as e:
        # improve the error message since we hide the module name in the line above
        import sys

        tb = sys.exc_info()[2]
        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )
