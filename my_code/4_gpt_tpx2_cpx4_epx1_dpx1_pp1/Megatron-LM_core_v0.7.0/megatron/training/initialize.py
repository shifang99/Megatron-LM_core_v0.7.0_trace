# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""

import random
import os
import time

import numpy as np
import torch
from datetime import timedelta

from megatron.legacy import fused_kernels
from megatron.training import get_adlr_autoresume
from megatron.training import get_args
from megatron.training import get_tensorboard_writer
from megatron.core import mpu, tensor_parallel
from megatron.training.arguments import parse_args, validate_args
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.legacy.model.transformer import bias_dropout_add_fused_train
from megatron.legacy.model.fused_bias_gelu import bias_gelu

def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:                                                      # trace_info : t_4
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."            # trace_info : t_5

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)                # trace_info : t_6

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):# trace_info : t_1093
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:                                              # trace_info : t_1094
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)                                     # trace_info : t_1095


    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)                                                 # trace_info : t_3099

    # torch.distributed initialization
    def finish_mpu_init():                                                     # trace_info : t_4313
        args = get_args()                                                      # trace_info : t_4321
        # Pytorch distributed.
        _initialize_distributed()                                              # trace_info : t_4325

        # Random seeds for reproducibility.
        if args.rank == 0:                                                     # trace_info : t_8701
            print("> setting random seeds to {} ...".format(args.seed))        # trace_info : t_8702
        _set_random_seed(args.seed, args.data_parallel_random_init)            # trace_info : t_8703

    if skip_mpu_initialization:                                                # trace_info : t_4314
        return None

    args = get_args()                                                          # trace_info : t_4315
    if args.lazy_mpu_init:                                                     # trace_info : t_4319
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()                                                      # trace_info : t_4320

        # Autoresume.
        _init_autoresume()                                                     # trace_info : t_8816

        # Compile dependencies.
        _compile_dependencies()                                                # trace_info : t_8820

        if args.tp_comm_overlap:                                               # trace_info : t_8893
           _initialize_tp_communicators()

        # No continuation function
        return None                                                            # trace_info : t_8894


def _compile_dependencies():

    args = get_args()                                                          # trace_info : t_8821

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8825
        start_time = time.time()                                               # trace_info : t_8826
        print("> compiling dataset index builder ...")                         # trace_info : t_8827
        from megatron.core.datasets.utils import compile_helpers               # trace_info : t_8828

        compile_helpers()                                                      # trace_info : t_8829
        print(                                                                 # trace_info : t_8834, t_8838
            ">>> done with dataset index builder. Compilation time: {:.3f} "   # trace_info : t_8835
            "seconds".format(time.time() - start_time),                        # trace_info : t_8836
            flush=True,                                                        # trace_info : t_8837
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length                                                  # trace_info : t_8839
    attn_batch_size = (                                                        # trace_info : t_8842
        args.num_attention_heads / args.tensor_model_parallel_size             # trace_info : t_8840
    ) * args.micro_batch_size                                                  # trace_info : t_8841
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (                                               # trace_info : t_8849
        seq_len > 16                                                           # trace_info : t_8843, t_8845, t_8847
        and seq_len <= 16384                                                   # trace_info : t_8844
        and seq_len % 4 == 0                                                   # trace_info : t_8846
        and attn_batch_size % 4 == 0                                           # trace_info : t_8848
    )
    # Print a warning.
    if not (                                                                   # trace_info : t_8851, t_8853, t_8855
        (args.fp16 or args.bf16)                                               # trace_info : t_8850
        and custom_kernel_constraint                                           # trace_info : t_8852
        and args.masked_softmax_fusion                                         # trace_info : t_8854
    ):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8856
        start_time = time.time()                                               # trace_info : t_8857
        print("> compiling and loading fused kernels ...", flush=True)         # trace_info : t_8858
        fused_kernels.load(args)                                               # trace_info : t_8859
        torch.distributed.barrier()                                            # trace_info : t_8885
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()                                                # trace_info : t_8886
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8887
        print(                                                                 # trace_info : t_8888, t_8892
            ">>> done with compiling and loading fused kernels. "              # trace_info : t_8889
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),# trace_info : t_8890
            flush=True,                                                        # trace_info : t_8891
        )

def _initialize_tp_communicators():
    """ initializing the communicators with user buffers for high-performance tensor-model-parallel 
        communication overlap """

    try:
       import yaml

       import transformer_engine
       from transformer_engine.pytorch import module as te_module

    except ImportError:
       raise RuntimeError("Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
             "'transformer_engine' packages") 

    args = get_args()

    if args.tp_comm_overlap_cfg is not None:
       with open(args.tp_comm_overlap_cfg,"r") as stream:    
          ub_cfgs = yaml.safe_load(stream)
    else:
       ub_cfgs = {}

    input_shape = [(args.seq_length * args.micro_batch_size) // args.context_parallel_size , args.hidden_size]

    #We create a MPI process group, which is needed to bootstrap the pipelined 
    #tensor-model-parallel communication overlap
    torch.distributed.new_group(backend='mpi')

    te_module.base.initialize_ub(shape = input_shape, tp_size = args.tensor_model_parallel_size, 
                                 use_fp8 = (args.fp8 is not None) , ub_cfgs = ub_cfgs,)

def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()                                                          # trace_info : t_4326

    device_count = torch.cuda.device_count()                                   # trace_info : t_4330
    if torch.distributed.is_initialized():                                     # trace_info : t_4331

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:                                                     # trace_info : t_4332
            print("> initializing torch distributed ...", flush=True)          # trace_info : t_4333
        # Manually set the device ids.
        if device_count > 0:                                                   # trace_info : t_4334
            device = args.rank % device_count                                  # trace_info : t_4335
            if args.local_rank is not None:                                    # trace_info : t_4336
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device                                       # trace_info : t_4337
            torch.cuda.set_device(device)                                      # trace_info : t_4338
        # Call the init process
        torch.distributed.init_process_group(                                  # trace_info : t_4339, t_4344
            backend=args.distributed_backend,                                  # trace_info : t_4340
            world_size=args.world_size,                                        # trace_info : t_4341
            rank=args.rank,                                                    # trace_info : t_4342
            timeout=timedelta(minutes=args.distributed_timeout_minutes),       # trace_info : t_4343
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:                                                       # trace_info : t_4345
        if mpu.model_parallel_is_initialized():                                # trace_info : t_4346
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(                                     # trace_info : t_4349, t_4359
                args.tensor_model_parallel_size,                               # trace_info : t_4350
                args.pipeline_model_parallel_size,                             # trace_info : t_4351
                args.virtual_pipeline_model_parallel_size,                     # trace_info : t_4352
                args.pipeline_model_parallel_split_rank,                       # trace_info : t_4353
                context_parallel_size=args.context_parallel_size,              # trace_info : t_4354
                expert_model_parallel_size=args.expert_model_parallel_size,    # trace_info : t_4355
                distributed_timeout_minutes=args.distributed_timeout_minutes,  # trace_info : t_4356
                nccl_communicator_config_path=args.nccl_communicator_config_path,# trace_info : t_4357
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',# trace_info : t_4358
            )
            if args.rank == 0:                                                 # trace_info : t_8681
                print(                                                         # trace_info : t_8682, t_8691
                    f"> initialized tensor model parallel with size "          # trace_info : t_8683, t_8690
                    f"{mpu.get_tensor_model_parallel_world_size()}"            # trace_info : t_8684
                )
                print(                                                         # trace_info : t_8692, t_8700
                    f"> initialized pipeline model parallel with size "        # trace_info : t_8693, t_8699
                    f"{mpu.get_pipeline_model_parallel_world_size()}"          # trace_info : t_8694
                )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()                                         # trace_info : t_8817
    if autoresume:                                                             # trace_info : t_8819
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:                                        # trace_info : t_8704
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())          # trace_info : t_8705
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:                                          # trace_info : t_8710
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)                                                      # trace_info : t_8711
        np.random.seed(seed)                                                   # trace_info : t_8712
        torch.manual_seed(seed)                                                # trace_info : t_8713
        if torch.cuda.device_count() > 0:                                      # trace_info : t_8714
            tensor_parallel.model_parallel_cuda_manual_seed(seed)              # trace_info : t_8715
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()                                                          # trace_info : t_17322
    writer = get_tensorboard_writer()                                          # trace_info : t_17326
    if writer:                                                                 # trace_info : t_17328
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])                         # trace_info : t_8905
    TORCH_MINOR = int(torch.__version__.split(".")[1])                         # trace_info : t_8906
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):          # trace_info : t_8907
        # nvfuser
        torch._C._jit_set_profiling_executor(True)                             # trace_info : t_8908
        torch._C._jit_set_profiling_mode(True)                                 # trace_info : t_8909
        torch._C._jit_override_can_fuse_on_cpu(False)                          # trace_info : t_8910
        torch._C._jit_override_can_fuse_on_gpu(False)                          # trace_info : t_8911
        torch._C._jit_set_texpr_fuser_enabled(False)                           # trace_info : t_8912
        torch._C._jit_set_nvfuser_enabled(True)                                # trace_info : t_8913
        torch._C._debug_set_autodiff_subgraph_inlining(False)                  # trace_info : t_8914
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()                                                     # trace_info : t_8915


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()                                                          # trace_info : t_8916
    if args.bf16:                                                              # trace_info : t_8920
        dtype = torch.bfloat16
    elif args.fp16:                                                            # trace_info : t_8921
        dtype = torch.float16                                                  # trace_info : t_8922
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(                                                         # trace_info : t_8923, t_8927
        args.ffn_hidden_size // args.tensor_model_parallel_size,               # trace_info : t_8924
        dtype=dtype,                                                           # trace_info : t_8925
        device="cuda",                                                         # trace_info : t_8926
    )
    input = torch.rand(                                                        # trace_info : t_8928, t_8935
        (                                                                      # trace_info : t_8932
            args.seq_length,                                                   # trace_info : t_8929
            args.micro_batch_size,                                             # trace_info : t_8930
            args.ffn_hidden_size // args.tensor_model_parallel_size,           # trace_info : t_8931
        ),
        dtype=dtype,                                                           # trace_info : t_8933
        device="cuda",                                                         # trace_info : t_8934
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):             # trace_info : t_8936, t_8954, t_8972
        bias.requires_grad, input.requires_grad = bias_grad, input_grad        # trace_info : t_8937, t_8955
        for _ in range(5):                                                     # trace_info : t_8938, t_8941, t_8944, t_8947, t_8950, ...
            output = bias_gelu(bias, input)                                    # trace_info : t_8939, t_8942, t_8945, t_8948, t_8951, ...
    del bias, input, output                                                    # trace_info : t_8973

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:                                                 # trace_info : t_8974
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length                                           # trace_info : t_8975
    input = torch.rand(                                                        # trace_info : t_8976, t_8980
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_8977
        dtype=dtype,                                                           # trace_info : t_8978
        device="cuda",                                                         # trace_info : t_8979
    )
    residual = torch.rand(                                                     # trace_info : t_8981, t_8985
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_8982
        dtype=dtype,                                                           # trace_info : t_8983
        device="cuda",                                                         # trace_info : t_8984
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(# trace_info : t_8986, t_8988
        residual                                                               # trace_info : t_8987
    )
    dropout_rate = 0.1                                                         # trace_info : t_8989
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(                           # trace_info : t_8990, t_8992, t_9012, t_9032
        [False, True], [True, True], [True, True]                              # trace_info : t_8991
    ):
        input.requires_grad = input_grad                                       # trace_info : t_8993, t_9013
        bias.requires_grad = bias_grad                                         # trace_info : t_8994, t_9014
        residual.requires_grad = residual_grad                                 # trace_info : t_8995, t_9015
        for _ in range(5):                                                     # trace_info : t_8996, t_8999, t_9002, t_9005, t_9008, ...
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)# trace_info : t_8997, t_9000, t_9003, t_9006, t_9009, ...
    del bias, input, residual, output                                          # trace_info : t_9033
    torch.cuda.empty_cache()                                                   # trace_info : t_9034
