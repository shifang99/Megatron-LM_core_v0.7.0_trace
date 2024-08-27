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
    def finish_mpu_init():                                                     # trace_info : t_5849
        args = get_args()                                                      # trace_info : t_5857
        # Pytorch distributed.
        _initialize_distributed()                                              # trace_info : t_5861

        # Random seeds for reproducibility.
        if args.rank == 0:                                                     # trace_info : t_10255
            print("> setting random seeds to {} ...".format(args.seed))        # trace_info : t_10256
        _set_random_seed(args.seed, args.data_parallel_random_init)            # trace_info : t_10257

    if skip_mpu_initialization:                                                # trace_info : t_5850
        return None

    args = get_args()                                                          # trace_info : t_5851
    if args.lazy_mpu_init:                                                     # trace_info : t_5855
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
        finish_mpu_init()                                                      # trace_info : t_5856

        # Autoresume.
        _init_autoresume()                                                     # trace_info : t_10370

        # Compile dependencies.
        _compile_dependencies()                                                # trace_info : t_10374

        if args.tp_comm_overlap:                                               # trace_info : t_10450
           _initialize_tp_communicators()

        # No continuation function
        return None                                                            # trace_info : t_10451


def _compile_dependencies():

    args = get_args()                                                          # trace_info : t_10375

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_10379
        start_time = time.time()                                               # trace_info : t_10380
        print("> compiling dataset index builder ...")                         # trace_info : t_10381
        from megatron.core.datasets.utils import compile_helpers               # trace_info : t_10382

        compile_helpers()                                                      # trace_info : t_10383
        print(                                                                 # trace_info : t_10388, t_10392
            ">>> done with dataset index builder. Compilation time: {:.3f} "   # trace_info : t_10389
            "seconds".format(time.time() - start_time),                        # trace_info : t_10390
            flush=True,                                                        # trace_info : t_10391
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length                                                  # trace_info : t_10393
    attn_batch_size = (                                                        # trace_info : t_10396
        args.num_attention_heads / args.tensor_model_parallel_size             # trace_info : t_10394
    ) * args.micro_batch_size                                                  # trace_info : t_10395
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (                                               # trace_info : t_10403
        seq_len > 16                                                           # trace_info : t_10397, t_10399, t_10401
        and seq_len <= 16384                                                   # trace_info : t_10398
        and seq_len % 4 == 0                                                   # trace_info : t_10400
        and attn_batch_size % 4 == 0                                           # trace_info : t_10402
    )
    # Print a warning.
    if not (                                                                   # trace_info : t_10405, t_10407
        (args.fp16 or args.bf16)                                               # trace_info : t_10404
        and custom_kernel_constraint                                           # trace_info : t_10406
        and args.masked_softmax_fusion
    ):
        if args.rank == 0:                                                     # trace_info : t_10408
            print(                                                             # trace_info : t_10409, t_10412
                "WARNING: constraints for invoking optimized"                  # trace_info : t_10410
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,                                                    # trace_info : t_10411
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_10413
        start_time = time.time()                                               # trace_info : t_10414
        print("> compiling and loading fused kernels ...", flush=True)         # trace_info : t_10415
        fused_kernels.load(args)                                               # trace_info : t_10416
        torch.distributed.barrier()                                            # trace_info : t_10442
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()                                                # trace_info : t_10443
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_10444
        print(                                                                 # trace_info : t_10445, t_10449
            ">>> done with compiling and loading fused kernels. "              # trace_info : t_10446
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),# trace_info : t_10447
            flush=True,                                                        # trace_info : t_10448
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
    args = get_args()                                                          # trace_info : t_5862

    device_count = torch.cuda.device_count()                                   # trace_info : t_5866
    if torch.distributed.is_initialized():                                     # trace_info : t_5867

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:                                                     # trace_info : t_5868
            print("> initializing torch distributed ...", flush=True)          # trace_info : t_5869
        # Manually set the device ids.
        if device_count > 0:                                                   # trace_info : t_5870
            device = args.rank % device_count                                  # trace_info : t_5871
            if args.local_rank is not None:                                    # trace_info : t_5872
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device                                       # trace_info : t_5873
            torch.cuda.set_device(device)                                      # trace_info : t_5874
        # Call the init process
        torch.distributed.init_process_group(                                  # trace_info : t_5875, t_5880
            backend=args.distributed_backend,                                  # trace_info : t_5876
            world_size=args.world_size,                                        # trace_info : t_5877
            rank=args.rank,                                                    # trace_info : t_5878
            timeout=timedelta(minutes=args.distributed_timeout_minutes),       # trace_info : t_5879
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:                                                       # trace_info : t_5881
        if mpu.model_parallel_is_initialized():                                # trace_info : t_5882
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(                                     # trace_info : t_5885, t_5895
                args.tensor_model_parallel_size,                               # trace_info : t_5886
                args.pipeline_model_parallel_size,                             # trace_info : t_5887
                args.virtual_pipeline_model_parallel_size,                     # trace_info : t_5888
                args.pipeline_model_parallel_split_rank,                       # trace_info : t_5889
                context_parallel_size=args.context_parallel_size,              # trace_info : t_5890
                expert_model_parallel_size=args.expert_model_parallel_size,    # trace_info : t_5891
                distributed_timeout_minutes=args.distributed_timeout_minutes,  # trace_info : t_5892
                nccl_communicator_config_path=args.nccl_communicator_config_path,# trace_info : t_5893
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',# trace_info : t_5894
            )
            if args.rank == 0:                                                 # trace_info : t_10235
                print(                                                         # trace_info : t_10236, t_10245
                    f"> initialized tensor model parallel with size "          # trace_info : t_10237, t_10244
                    f"{mpu.get_tensor_model_parallel_world_size()}"            # trace_info : t_10238
                )
                print(                                                         # trace_info : t_10246, t_10254
                    f"> initialized pipeline model parallel with size "        # trace_info : t_10247, t_10253
                    f"{mpu.get_pipeline_model_parallel_world_size()}"          # trace_info : t_10248
                )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()                                         # trace_info : t_10371
    if autoresume:                                                             # trace_info : t_10373
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:                                        # trace_info : t_10258
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())          # trace_info : t_10259
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:                                          # trace_info : t_10264
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)                                                      # trace_info : t_10265
        np.random.seed(seed)                                                   # trace_info : t_10266
        torch.manual_seed(seed)                                                # trace_info : t_10267
        if torch.cuda.device_count() > 0:                                      # trace_info : t_10268
            tensor_parallel.model_parallel_cuda_manual_seed(seed)              # trace_info : t_10269
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()                                                          # trace_info : t_19330
    writer = get_tensorboard_writer()                                          # trace_info : t_19334
    if writer:                                                                 # trace_info : t_19336
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])                         # trace_info : t_10462
    TORCH_MINOR = int(torch.__version__.split(".")[1])                         # trace_info : t_10463
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):          # trace_info : t_10464
        # nvfuser
        torch._C._jit_set_profiling_executor(True)                             # trace_info : t_10465
        torch._C._jit_set_profiling_mode(True)                                 # trace_info : t_10466
        torch._C._jit_override_can_fuse_on_cpu(False)                          # trace_info : t_10467
        torch._C._jit_override_can_fuse_on_gpu(False)                          # trace_info : t_10468
        torch._C._jit_set_texpr_fuser_enabled(False)                           # trace_info : t_10469
        torch._C._jit_set_nvfuser_enabled(True)                                # trace_info : t_10470
        torch._C._debug_set_autodiff_subgraph_inlining(False)                  # trace_info : t_10471
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()                                                     # trace_info : t_10472


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()                                                          # trace_info : t_10473
    if args.bf16:                                                              # trace_info : t_10477
        dtype = torch.bfloat16
    elif args.fp16:                                                            # trace_info : t_10478
        dtype = torch.float16                                                  # trace_info : t_10479
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(                                                         # trace_info : t_10480, t_10484
        args.ffn_hidden_size // args.tensor_model_parallel_size,               # trace_info : t_10481
        dtype=dtype,                                                           # trace_info : t_10482
        device="cuda",                                                         # trace_info : t_10483
    )
    input = torch.rand(                                                        # trace_info : t_10485, t_10492
        (                                                                      # trace_info : t_10489
            args.seq_length,                                                   # trace_info : t_10486
            args.micro_batch_size,                                             # trace_info : t_10487
            args.ffn_hidden_size // args.tensor_model_parallel_size,           # trace_info : t_10488
        ),
        dtype=dtype,                                                           # trace_info : t_10490
        device="cuda",                                                         # trace_info : t_10491
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):             # trace_info : t_10493, t_10511, t_10529
        bias.requires_grad, input.requires_grad = bias_grad, input_grad        # trace_info : t_10494, t_10512
        for _ in range(5):                                                     # trace_info : t_10495, t_10498, t_10501, t_10504, t_10507, ...
            output = bias_gelu(bias, input)                                    # trace_info : t_10496, t_10499, t_10502, t_10505, t_10508, ...
    del bias, input, output                                                    # trace_info : t_10530

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:                                                 # trace_info : t_10531
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length                                           # trace_info : t_10532
    input = torch.rand(                                                        # trace_info : t_10533, t_10537
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_10534
        dtype=dtype,                                                           # trace_info : t_10535
        device="cuda",                                                         # trace_info : t_10536
    )
    residual = torch.rand(                                                     # trace_info : t_10538, t_10542
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_10539
        dtype=dtype,                                                           # trace_info : t_10540
        device="cuda",                                                         # trace_info : t_10541
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(# trace_info : t_10543, t_10545
        residual                                                               # trace_info : t_10544
    )
    dropout_rate = 0.1                                                         # trace_info : t_10546
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(                           # trace_info : t_10547, t_10549, t_10569, t_10589
        [False, True], [True, True], [True, True]                              # trace_info : t_10548
    ):
        input.requires_grad = input_grad                                       # trace_info : t_10550, t_10570
        bias.requires_grad = bias_grad                                         # trace_info : t_10551, t_10571
        residual.requires_grad = residual_grad                                 # trace_info : t_10552, t_10572
        for _ in range(5):                                                     # trace_info : t_10553, t_10556, t_10559, t_10562, t_10565, ...
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)# trace_info : t_10554, t_10557, t_10560, t_10563, t_10566, ...
    del bias, input, residual, output                                          # trace_info : t_10590
    torch.cuda.empty_cache()                                                   # trace_info : t_10591
