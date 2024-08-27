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
    set_global_variables(args)                                                 # trace_info : t_3098

    # torch.distributed initialization
    def finish_mpu_init():                                                     # trace_info : t_4312
        args = get_args()                                                      # trace_info : t_4320
        # Pytorch distributed.
        _initialize_distributed()                                              # trace_info : t_4324

        # Random seeds for reproducibility.
        if args.rank == 0:                                                     # trace_info : t_8337
            print("> setting random seeds to {} ...".format(args.seed))        # trace_info : t_8338
        _set_random_seed(args.seed, args.data_parallel_random_init)            # trace_info : t_8339

    if skip_mpu_initialization:                                                # trace_info : t_4313
        return None

    args = get_args()                                                          # trace_info : t_4314
    if args.lazy_mpu_init:                                                     # trace_info : t_4318
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
        finish_mpu_init()                                                      # trace_info : t_4319

        # Autoresume.
        _init_autoresume()                                                     # trace_info : t_8452

        # Compile dependencies.
        _compile_dependencies()                                                # trace_info : t_8456

        if args.tp_comm_overlap:                                               # trace_info : t_8532
           _initialize_tp_communicators()

        # No continuation function
        return None                                                            # trace_info : t_8533


def _compile_dependencies():

    args = get_args()                                                          # trace_info : t_8457

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8461
        start_time = time.time()                                               # trace_info : t_8462
        print("> compiling dataset index builder ...")                         # trace_info : t_8463
        from megatron.core.datasets.utils import compile_helpers               # trace_info : t_8464

        compile_helpers()                                                      # trace_info : t_8465
        print(                                                                 # trace_info : t_8470, t_8474
            ">>> done with dataset index builder. Compilation time: {:.3f} "   # trace_info : t_8471
            "seconds".format(time.time() - start_time),                        # trace_info : t_8472
            flush=True,                                                        # trace_info : t_8473
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length                                                  # trace_info : t_8475
    attn_batch_size = (                                                        # trace_info : t_8478
        args.num_attention_heads / args.tensor_model_parallel_size             # trace_info : t_8476
    ) * args.micro_batch_size                                                  # trace_info : t_8477
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (                                               # trace_info : t_8485
        seq_len > 16                                                           # trace_info : t_8479, t_8481, t_8483
        and seq_len <= 16384                                                   # trace_info : t_8480
        and seq_len % 4 == 0                                                   # trace_info : t_8482
        and attn_batch_size % 4 == 0                                           # trace_info : t_8484
    )
    # Print a warning.
    if not (                                                                   # trace_info : t_8487, t_8489
        (args.fp16 or args.bf16)                                               # trace_info : t_8486, t_8488
        and custom_kernel_constraint
        and args.masked_softmax_fusion
    ):
        if args.rank == 0:                                                     # trace_info : t_8490
            print(                                                             # trace_info : t_8491, t_8494
                "WARNING: constraints for invoking optimized"                  # trace_info : t_8492
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,                                                    # trace_info : t_8493
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8495
        start_time = time.time()                                               # trace_info : t_8496
        print("> compiling and loading fused kernels ...", flush=True)         # trace_info : t_8497
        fused_kernels.load(args)                                               # trace_info : t_8498
        torch.distributed.barrier()                                            # trace_info : t_8524
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()                                                # trace_info : t_8525
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8526
        print(                                                                 # trace_info : t_8527, t_8531
            ">>> done with compiling and loading fused kernels. "              # trace_info : t_8528
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),# trace_info : t_8529
            flush=True,                                                        # trace_info : t_8530
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
    args = get_args()                                                          # trace_info : t_4325

    device_count = torch.cuda.device_count()                                   # trace_info : t_4329
    if torch.distributed.is_initialized():                                     # trace_info : t_4330

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:                                                     # trace_info : t_4331
            print("> initializing torch distributed ...", flush=True)          # trace_info : t_4332
        # Manually set the device ids.
        if device_count > 0:                                                   # trace_info : t_4333
            device = args.rank % device_count                                  # trace_info : t_4334
            if args.local_rank is not None:                                    # trace_info : t_4335
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device                                       # trace_info : t_4336
            torch.cuda.set_device(device)                                      # trace_info : t_4337
        # Call the init process
        torch.distributed.init_process_group(                                  # trace_info : t_4338, t_4343
            backend=args.distributed_backend,                                  # trace_info : t_4339
            world_size=args.world_size,                                        # trace_info : t_4340
            rank=args.rank,                                                    # trace_info : t_4341
            timeout=timedelta(minutes=args.distributed_timeout_minutes),       # trace_info : t_4342
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:                                                       # trace_info : t_4344
        if mpu.model_parallel_is_initialized():                                # trace_info : t_4345
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(                                     # trace_info : t_4348, t_4358
                args.tensor_model_parallel_size,                               # trace_info : t_4349
                args.pipeline_model_parallel_size,                             # trace_info : t_4350
                args.virtual_pipeline_model_parallel_size,                     # trace_info : t_4351
                args.pipeline_model_parallel_split_rank,                       # trace_info : t_4352
                context_parallel_size=args.context_parallel_size,              # trace_info : t_4353
                expert_model_parallel_size=args.expert_model_parallel_size,    # trace_info : t_4354
                distributed_timeout_minutes=args.distributed_timeout_minutes,  # trace_info : t_4355
                nccl_communicator_config_path=args.nccl_communicator_config_path,# trace_info : t_4356
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',# trace_info : t_4357
            )
            if args.rank == 0:                                                 # trace_info : t_8317
                print(                                                         # trace_info : t_8318, t_8327
                    f"> initialized tensor model parallel with size "          # trace_info : t_8319, t_8326
                    f"{mpu.get_tensor_model_parallel_world_size()}"            # trace_info : t_8320
                )
                print(                                                         # trace_info : t_8328, t_8336
                    f"> initialized pipeline model parallel with size "        # trace_info : t_8329, t_8335
                    f"{mpu.get_pipeline_model_parallel_world_size()}"          # trace_info : t_8330
                )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()                                         # trace_info : t_8453
    if autoresume:                                                             # trace_info : t_8455
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:                                        # trace_info : t_8340
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())          # trace_info : t_8341
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:                                          # trace_info : t_8346
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)                                                      # trace_info : t_8347
        np.random.seed(seed)                                                   # trace_info : t_8348
        torch.manual_seed(seed)                                                # trace_info : t_8349
        if torch.cuda.device_count() > 0:                                      # trace_info : t_8350
            tensor_parallel.model_parallel_cuda_manual_seed(seed)              # trace_info : t_8351
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()                                                          # trace_info : t_17531
    writer = get_tensorboard_writer()                                          # trace_info : t_17535
    if writer:                                                                 # trace_info : t_17537
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])                         # trace_info : t_8544
    TORCH_MINOR = int(torch.__version__.split(".")[1])                         # trace_info : t_8545
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):          # trace_info : t_8546
        # nvfuser
        torch._C._jit_set_profiling_executor(True)                             # trace_info : t_8547
        torch._C._jit_set_profiling_mode(True)                                 # trace_info : t_8548
        torch._C._jit_override_can_fuse_on_cpu(False)                          # trace_info : t_8549
        torch._C._jit_override_can_fuse_on_gpu(False)                          # trace_info : t_8550
        torch._C._jit_set_texpr_fuser_enabled(False)                           # trace_info : t_8551
        torch._C._jit_set_nvfuser_enabled(True)                                # trace_info : t_8552
        torch._C._debug_set_autodiff_subgraph_inlining(False)                  # trace_info : t_8553
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()                                                     # trace_info : t_8554


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()                                                          # trace_info : t_8555
    if args.bf16:                                                              # trace_info : t_8559
        dtype = torch.bfloat16
    elif args.fp16:                                                            # trace_info : t_8560
        dtype = torch.float16
    else:
        dtype = torch.float32                                                  # trace_info : t_8561

    # Warmup fused bias+gelu
    bias = torch.rand(                                                         # trace_info : t_8562, t_8566
        args.ffn_hidden_size // args.tensor_model_parallel_size,               # trace_info : t_8563
        dtype=dtype,                                                           # trace_info : t_8564
        device="cuda",                                                         # trace_info : t_8565
    )
    input = torch.rand(                                                        # trace_info : t_8567, t_8574
        (                                                                      # trace_info : t_8571
            args.seq_length,                                                   # trace_info : t_8568
            args.micro_batch_size,                                             # trace_info : t_8569
            args.ffn_hidden_size // args.tensor_model_parallel_size,           # trace_info : t_8570
        ),
        dtype=dtype,                                                           # trace_info : t_8572
        device="cuda",                                                         # trace_info : t_8573
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):             # trace_info : t_8575, t_8593, t_8611
        bias.requires_grad, input.requires_grad = bias_grad, input_grad        # trace_info : t_8576, t_8594
        for _ in range(5):                                                     # trace_info : t_8577, t_8580, t_8583, t_8586, t_8589, ...
            output = bias_gelu(bias, input)                                    # trace_info : t_8578, t_8581, t_8584, t_8587, t_8590, ...
    del bias, input, output                                                    # trace_info : t_8612

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:                                                 # trace_info : t_8613
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()# trace_info : t_8614
    else:
        seq_length = args.seq_length
    input = torch.rand(                                                        # trace_info : t_8620, t_8624
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_8621
        dtype=dtype,                                                           # trace_info : t_8622
        device="cuda",                                                         # trace_info : t_8623
    )
    residual = torch.rand(                                                     # trace_info : t_8625, t_8629
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_8626
        dtype=dtype,                                                           # trace_info : t_8627
        device="cuda",                                                         # trace_info : t_8628
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(# trace_info : t_8630, t_8632
        residual                                                               # trace_info : t_8631
    )
    dropout_rate = 0.1                                                         # trace_info : t_8633
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(                           # trace_info : t_8634, t_8636, t_8656, t_8676
        [False, True], [True, True], [True, True]                              # trace_info : t_8635
    ):
        input.requires_grad = input_grad                                       # trace_info : t_8637, t_8657
        bias.requires_grad = bias_grad                                         # trace_info : t_8638, t_8658
        residual.requires_grad = residual_grad                                 # trace_info : t_8639, t_8659
        for _ in range(5):                                                     # trace_info : t_8640, t_8643, t_8646, t_8649, t_8652, ...
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)# trace_info : t_8641, t_8644, t_8647, t_8650, t_8653, ...
    del bias, input, residual, output                                          # trace_info : t_8677
    torch.cuda.empty_cache()                                                   # trace_info : t_8678
