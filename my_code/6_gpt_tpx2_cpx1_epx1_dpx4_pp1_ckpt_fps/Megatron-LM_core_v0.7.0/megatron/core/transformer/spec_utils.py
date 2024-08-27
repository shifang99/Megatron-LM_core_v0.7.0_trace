# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import types                                                                   # trace_info : t_9370, t_9394, t_9416
from dataclasses import dataclass, field                                       # trace_info : t_9371, t_9395, t_9417
from typing import Tuple, Union                                                # trace_info : t_9373, t_9396, t_9419

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
    params: dict = field(default_factory=lambda: {})                           # trace_info : t_9372, t_9418
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
    if isinstance(spec_or_module, types.FunctionType):                         # trace_info : t_9625, t_9675, t_9714, t_9775, t_9863, ...
        return spec_or_module                                                  # trace_info : t_10196, t_10624, t_11198, t_11626

    # If the passed `spec_or_module` is actually a spec (instance of
    # `ModuleSpec`) and it specifies a `Function` using its `module`
    # field, return the `Function` as it is
    if isinstance(spec_or_module, ModuleSpec) and isinstance(                  # trace_info : t_9626, t_9628, t_9676, t_9715, t_9717, ...
        spec_or_module.module, types.FunctionType                              # trace_info : t_9627, t_9716, t_10298, t_10629, t_10718, ...
    ):
        return spec_or_module.module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):                                       # trace_info : t_9629, t_9677, t_9718, t_9777, t_9865, ...
        module = spec_or_module                                                # trace_info : t_9678, t_9778, t_9866, t_10007, t_10160, ...
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):# trace_info : t_9630, t_9719, t_10301, t_10632, t_10721, ...
        module = spec_or_module.module                                         # trace_info : t_9631, t_9720, t_10302, t_10633, t_10722, ...
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module)

    # If the imported module is actually a `Function` return it as it is
    if isinstance(module, types.FunctionType):                                 # trace_info : t_9632, t_9679, t_9721, t_9779, t_9867, ...
        return module

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code

    # Add the `submodules` argument to the module init call if it exists in the
    # spec.
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:# trace_info : t_9633, t_9680, t_9722, t_9780, t_9868, ...
        kwargs["submodules"] = spec_or_module.submodules                       # trace_info : t_9634, t_9723, t_10305, t_10636, t_10725, ...

    try:                                                                       # trace_info : t_9635, t_9681, t_9724, t_9781, t_9869, ...
        return module(                                                         # trace_info : t_9636, t_9638, t_9640, t_9642, t_9682, ...
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs# trace_info : t_9637, t_9639, t_9641, t_9683, t_9685, ...
        )
    except Exception as e:
        # improve the error message since we hide the module name in the line above
        import sys

        tb = sys.exc_info()[2]
        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )
