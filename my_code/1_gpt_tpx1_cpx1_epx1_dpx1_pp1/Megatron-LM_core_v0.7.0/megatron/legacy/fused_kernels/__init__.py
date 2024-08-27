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
    cc_flag = []                                                               # trace_info : t_5550
    _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(      # trace_info : t_5551, t_5553
        cpp_extension.CUDA_HOME                                                # trace_info : t_5552
    )
    if int(bare_metal_major) >= 11:                                            # trace_info : t_5563
        cc_flag.append('-gencode')                                             # trace_info : t_5564
        cc_flag.append('arch=compute_80,code=sm_80')                           # trace_info : t_5565
        if int(bare_metal_minor) >= 8:                                         # trace_info : t_5566
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()                         # trace_info : t_5567
    buildpath = srcpath / "build"                                              # trace_info : t_5568
    _create_build_dir(buildpath)                                               # trace_info : t_5569

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):           # trace_info : t_5574
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
    raw_output = subprocess.check_output(                                      # trace_info : t_5554, t_5556
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True                # trace_info : t_5555
    )
    output = raw_output.split()                                                # trace_info : t_5557
    release_idx = output.index("release") + 1                                  # trace_info : t_5558
    release = output[release_idx].split(".")                                   # trace_info : t_5559
    bare_metal_major = release[0]                                              # trace_info : t_5560
    bare_metal_minor = release[1][0]                                           # trace_info : t_5561

    return raw_output, bare_metal_major, bare_metal_minor                      # trace_info : t_5562


def _create_build_dir(buildpath):
    try:                                                                       # trace_info : t_5570
        os.mkdir(buildpath)                                                    # trace_info : t_5571
    except OSError:                                                            # trace_info : t_5572
        if not os.path.isdir(buildpath):                                       # trace_info : t_5573
            print(f"Creation of the build directory {buildpath} failed")
