# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import pathlib
import subprocess

from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(args):

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []                                                               # trace_info : t_8499
    _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(      # trace_info : t_8500, t_8502
        cpp_extension.CUDA_HOME                                                # trace_info : t_8501
    )
    if int(bare_metal_major) >= 11:                                            # trace_info : t_8512
        cc_flag.append('-gencode')                                             # trace_info : t_8513
        cc_flag.append('arch=compute_80,code=sm_80')                           # trace_info : t_8514
        if int(bare_metal_minor) >= 8:                                         # trace_info : t_8515
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()                         # trace_info : t_8516
    buildpath = srcpath / "build"                                              # trace_info : t_8517
    _create_build_dir(buildpath)                                               # trace_info : t_8518

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):           # trace_info : t_8523
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-gencode",
                "arch=compute_70,code=sm_70",
                "--use_fast_math",
            ]
            + extra_cuda_flags
            + cc_flag,
            verbose=(args.rank == 0),
        )


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(                                      # trace_info : t_8503, t_8505
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True                # trace_info : t_8504
    )
    output = raw_output.split()                                                # trace_info : t_8506
    release_idx = output.index("release") + 1                                  # trace_info : t_8507
    release = output[release_idx].split(".")                                   # trace_info : t_8508
    bare_metal_major = release[0]                                              # trace_info : t_8509
    bare_metal_minor = release[1][0]                                           # trace_info : t_8510

    return raw_output, bare_metal_major, bare_metal_minor                      # trace_info : t_8511


def _create_build_dir(buildpath):
    try:                                                                       # trace_info : t_8519
        os.mkdir(buildpath)                                                    # trace_info : t_8520
    except OSError:                                                            # trace_info : t_8521
        if not os.path.isdir(buildpath):                                       # trace_info : t_8522
            print(f"Creation of the build directory {buildpath} failed")
