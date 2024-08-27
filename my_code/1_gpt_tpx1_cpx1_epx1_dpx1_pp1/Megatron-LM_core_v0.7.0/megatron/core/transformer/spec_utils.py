# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import types                                                                   # trace_info : t_6341, t_6365, t_6387
from dataclasses import dataclass, field                                       # trace_info : t_6342, t_6366, t_6388
from typing import Tuple, Union                                                # trace_info : t_6344, t_6367, t_6390

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
    params: dict = field(default_factory=lambda: {})                           # trace_info : t_6343, t_6389
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
    if isinstance(spec_or_module, types.FunctionType):                         # trace_info : t_6596, t_6646, t_6685, t_6746, t_6834, ...
        return spec_or_module                                                  # trace_info : t_7167, t_7595, t_8169, t_8597

    # If the passed `spec_or_module` is actually a spec (instance of
    # `ModuleSpec`) and it specifies a `Function` using its `module`
    # field, return the `Function` as it is
    if isinstance(spec_or_module, ModuleSpec) and isinstance(                  # trace_info : t_6597, t_6599, t_6647, t_6686, t_6688, ...
        spec_or_module.module, types.FunctionType                              # trace_info : t_6598, t_6687, t_7269, t_7600, t_7689, ...
    ):
        return spec_or_module.module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):                                       # trace_info : t_6600, t_6648, t_6689, t_6748, t_6836, ...
        module = spec_or_module                                                # trace_info : t_6649, t_6749, t_6837, t_6978, t_7131, ...
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):# trace_info : t_6601, t_6690, t_7272, t_7603, t_7692, ...
        module = spec_or_module.module                                         # trace_info : t_6602, t_6691, t_7273, t_7604, t_7693, ...
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module)

    # If the imported module is actually a `Function` return it as it is
    if isinstance(module, types.FunctionType):                                 # trace_info : t_6603, t_6650, t_6692, t_6750, t_6838, ...
        return module

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code

    # Add the `submodules` argument to the module init call if it exists in the
    # spec.
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:# trace_info : t_6604, t_6651, t_6693, t_6751, t_6839, ...
        kwargs["submodules"] = spec_or_module.submodules                       # trace_info : t_6605, t_6694, t_7276, t_7607, t_7696, ...

    try:                                                                       # trace_info : t_6606, t_6652, t_6695, t_6752, t_6840, ...
        return module(                                                         # trace_info : t_6607, t_6609, t_6611, t_6613, t_6653, ...
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs# trace_info : t_6608, t_6610, t_6612, t_6654, t_6656, ...
        )
    except Exception as e:
        # improve the error message since we hide the module name in the line above
        import sys

        tb = sys.exc_info()[2]
        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )
