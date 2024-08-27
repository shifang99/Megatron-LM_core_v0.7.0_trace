# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import types                                                                   # trace_info : t_11208, t_11232, t_11254
from dataclasses import dataclass, field                                       # trace_info : t_11209, t_11233, t_11255
from typing import Tuple, Union                                                # trace_info : t_11211, t_11234, t_11257

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
    params: dict = field(default_factory=lambda: {})                           # trace_info : t_11210, t_11256
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
    if isinstance(spec_or_module, types.FunctionType):                         # trace_info : t_11463, t_11513, t_11552, t_11613, t_11701, ...
        return spec_or_module                                                  # trace_info : t_12034, t_12462, t_13036, t_13464

    # If the passed `spec_or_module` is actually a spec (instance of
    # `ModuleSpec`) and it specifies a `Function` using its `module`
    # field, return the `Function` as it is
    if isinstance(spec_or_module, ModuleSpec) and isinstance(                  # trace_info : t_11464, t_11466, t_11514, t_11553, t_11555, ...
        spec_or_module.module, types.FunctionType                              # trace_info : t_11465, t_11554, t_12136, t_12467, t_12556, ...
    ):
        return spec_or_module.module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):                                       # trace_info : t_11467, t_11515, t_11556, t_11615, t_11703, ...
        module = spec_or_module                                                # trace_info : t_11516, t_11616, t_11704, t_11845, t_11998, ...
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):# trace_info : t_11468, t_11557, t_12139, t_12470, t_12559, ...
        module = spec_or_module.module                                         # trace_info : t_11469, t_11558, t_12140, t_12471, t_12560, ...
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module)

    # If the imported module is actually a `Function` return it as it is
    if isinstance(module, types.FunctionType):                                 # trace_info : t_11470, t_11517, t_11559, t_11617, t_11705, ...
        return module

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code

    # Add the `submodules` argument to the module init call if it exists in the
    # spec.
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:# trace_info : t_11471, t_11518, t_11560, t_11618, t_11706, ...
        kwargs["submodules"] = spec_or_module.submodules                       # trace_info : t_11472, t_11561, t_12143, t_12474, t_12563, ...

    try:                                                                       # trace_info : t_11473, t_11519, t_11562, t_11619, t_11707, ...
        return module(                                                         # trace_info : t_11474, t_11476, t_11478, t_11480, t_11520, ...
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs# trace_info : t_11475, t_11477, t_11479, t_11521, t_11523, ...
        )
    except Exception as e:
        # improve the error message since we hide the module name in the line above
        import sys

        tb = sys.exc_info()[2]
        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )
