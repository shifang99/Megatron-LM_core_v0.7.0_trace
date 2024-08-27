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
    set_global_variables(args)                                                 # trace_info : t_3100

    # torch.distributed initialization
    def finish_mpu_init():                                                     # trace_info : t_4314
        args = get_args()                                                      # trace_info : t_4322
        # Pytorch distributed.
        _initialize_distributed()                                              # trace_info : t_4326

        # Random seeds for reproducibility.
        if args.rank == 0:                                                     # trace_info : t_8420
            print("> setting random seeds to {} ...".format(args.seed))        # trace_info : t_8421
        _set_random_seed(args.seed, args.data_parallel_random_init)            # trace_info : t_8422

    if skip_mpu_initialization:                                                # trace_info : t_4315
        return None

    args = get_args()                                                          # trace_info : t_4316
    if args.lazy_mpu_init:                                                     # trace_info : t_4320
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
        finish_mpu_init()                                                      # trace_info : t_4321

        # Autoresume.
        _init_autoresume()                                                     # trace_info : t_8535

        # Compile dependencies.
        _compile_dependencies()                                                # trace_info : t_8539

        if args.tp_comm_overlap:                                               # trace_info : t_8612
           _initialize_tp_communicators()

        # No continuation function
        return None                                                            # trace_info : t_8613


def _compile_dependencies():

    args = get_args()                                                          # trace_info : t_8540

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8544
        start_time = time.time()                                               # trace_info : t_8545
        print("> compiling dataset index builder ...")                         # trace_info : t_8546
        from megatron.core.datasets.utils import compile_helpers               # trace_info : t_8547

        compile_helpers()                                                      # trace_info : t_8548
        print(                                                                 # trace_info : t_8553, t_8557
            ">>> done with dataset index builder. Compilation time: {:.3f} "   # trace_info : t_8554
            "seconds".format(time.time() - start_time),                        # trace_info : t_8555
            flush=True,                                                        # trace_info : t_8556
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length                                                  # trace_info : t_8558
    attn_batch_size = (                                                        # trace_info : t_8561
        args.num_attention_heads / args.tensor_model_parallel_size             # trace_info : t_8559
    ) * args.micro_batch_size                                                  # trace_info : t_8560
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (                                               # trace_info : t_8568
        seq_len > 16                                                           # trace_info : t_8562, t_8564, t_8566
        and seq_len <= 16384                                                   # trace_info : t_8563
        and seq_len % 4 == 0                                                   # trace_info : t_8565
        and attn_batch_size % 4 == 0                                           # trace_info : t_8567
    )
    # Print a warning.
    if not (                                                                   # trace_info : t_8570, t_8572, t_8574
        (args.fp16 or args.bf16)                                               # trace_info : t_8569
        and custom_kernel_constraint                                           # trace_info : t_8571
        and args.masked_softmax_fusion                                         # trace_info : t_8573
    ):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8575
        start_time = time.time()                                               # trace_info : t_8576
        print("> compiling and loading fused kernels ...", flush=True)         # trace_info : t_8577
        fused_kernels.load(args)                                               # trace_info : t_8578
        torch.distributed.barrier()                                            # trace_info : t_8604
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()                                                # trace_info : t_8605
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_8606
        print(                                                                 # trace_info : t_8607, t_8611
            ">>> done with compiling and loading fused kernels. "              # trace_info : t_8608
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),# trace_info : t_8609
            flush=True,                                                        # trace_info : t_8610
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
    args = get_args()                                                          # trace_info : t_4327

    device_count = torch.cuda.device_count()                                   # trace_info : t_4331
    if torch.distributed.is_initialized():                                     # trace_info : t_4332

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:                                                     # trace_info : t_4333
            print("> initializing torch distributed ...", flush=True)          # trace_info : t_4334
        # Manually set the device ids.
        if device_count > 0:                                                   # trace_info : t_4335
            device = args.rank % device_count                                  # trace_info : t_4336
            if args.local_rank is not None:                                    # trace_info : t_4337
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device                                       # trace_info : t_4338
            torch.cuda.set_device(device)                                      # trace_info : t_4339
        # Call the init process
        torch.distributed.init_process_group(                                  # trace_info : t_4340, t_4345
            backend=args.distributed_backend,                                  # trace_info : t_4341
            world_size=args.world_size,                                        # trace_info : t_4342
            rank=args.rank,                                                    # trace_info : t_4343
            timeout=timedelta(minutes=args.distributed_timeout_minutes),       # trace_info : t_4344
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:                                                       # trace_info : t_4346
        if mpu.model_parallel_is_initialized():                                # trace_info : t_4347
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(                                     # trace_info : t_4350, t_4360
                args.tensor_model_parallel_size,                               # trace_info : t_4351
                args.pipeline_model_parallel_size,                             # trace_info : t_4352
                args.virtual_pipeline_model_parallel_size,                     # trace_info : t_4353
                args.pipeline_model_parallel_split_rank,                       # trace_info : t_4354
                context_parallel_size=args.context_parallel_size,              # trace_info : t_4355
                expert_model_parallel_size=args.expert_model_parallel_size,    # trace_info : t_4356
                distributed_timeout_minutes=args.distributed_timeout_minutes,  # trace_info : t_4357
                nccl_communicator_config_path=args.nccl_communicator_config_path,# trace_info : t_4358
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',# trace_info : t_4359
            )
            if args.rank == 0:                                                 # trace_info : t_8400
                print(                                                         # trace_info : t_8401, t_8410
                    f"> initialized tensor model parallel with size "          # trace_info : t_8402, t_8409
                    f"{mpu.get_tensor_model_parallel_world_size()}"            # trace_info : t_8403
                )
                print(                                                         # trace_info : t_8411, t_8419
                    f"> initialized pipeline model parallel with size "        # trace_info : t_8412, t_8418
                    f"{mpu.get_pipeline_model_parallel_world_size()}"          # trace_info : t_8413
                )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()                                         # trace_info : t_8536
    if autoresume:                                                             # trace_info : t_8538
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:                                        # trace_info : t_8423
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())          # trace_info : t_8424
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:                                          # trace_info : t_8429
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)                                                      # trace_info : t_8430
        np.random.seed(seed)                                                   # trace_info : t_8431
        torch.manual_seed(seed)                                                # trace_info : t_8432
        if torch.cuda.device_count() > 0:                                      # trace_info : t_8433
            tensor_parallel.model_parallel_cuda_manual_seed(seed)              # trace_info : t_8434
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()                                                          # trace_info : t_17600
    writer = get_tensorboard_writer()                                          # trace_info : t_17604
    if writer:                                                                 # trace_info : t_17606
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])                         # trace_info : t_8624
    TORCH_MINOR = int(torch.__version__.split(".")[1])                         # trace_info : t_8625
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):          # trace_info : t_8626
        # nvfuser
        torch._C._jit_set_profiling_executor(True)                             # trace_info : t_8627
        torch._C._jit_set_profiling_mode(True)                                 # trace_info : t_8628
        torch._C._jit_override_can_fuse_on_cpu(False)                          # trace_info : t_8629
        torch._C._jit_override_can_fuse_on_gpu(False)                          # trace_info : t_8630
        torch._C._jit_set_texpr_fuser_enabled(False)                           # trace_info : t_8631
        torch._C._jit_set_nvfuser_enabled(True)                                # trace_info : t_8632
        torch._C._debug_set_autodiff_subgraph_inlining(False)                  # trace_info : t_8633
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()                                                     # trace_info : t_8634


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()                                                          # trace_info : t_8635
    if args.bf16:                                                              # trace_info : t_8639
        dtype = torch.bfloat16
    elif args.fp16:                                                            # trace_info : t_8640
        dtype = torch.float16                                                  # trace_info : t_8641
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(                                                         # trace_info : t_8642, t_8646
        args.ffn_hidden_size // args.tensor_model_parallel_size,               # trace_info : t_8643
        dtype=dtype,                                                           # trace_info : t_8644
        device="cuda",                                                         # trace_info : t_8645
    )
    input = torch.rand(                                                        # trace_info : t_8647, t_8654
        (                                                                      # trace_info : t_8651
            args.seq_length,                                                   # trace_info : t_8648
            args.micro_batch_size,                                             # trace_info : t_8649
            args.ffn_hidden_size // args.tensor_model_parallel_size,           # trace_info : t_8650
        ),
        dtype=dtype,                                                           # trace_info : t_8652
        device="cuda",                                                         # trace_info : t_8653
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):             # trace_info : t_8655, t_8673, t_8691
        bias.requires_grad, input.requires_grad = bias_grad, input_grad        # trace_info : t_8656, t_8674
        for _ in range(5):                                                     # trace_info : t_8657, t_8660, t_8663, t_8666, t_8669, ...
            output = bias_gelu(bias, input)                                    # trace_info : t_8658, t_8661, t_8664, t_8667, t_8670, ...
    del bias, input, output                                                    # trace_info : t_8692

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:                                                 # trace_info : t_8693
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length                                           # trace_info : t_8694
    input = torch.rand(                                                        # trace_info : t_8695, t_8699
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_8696
        dtype=dtype,                                                           # trace_info : t_8697
        device="cuda",                                                         # trace_info : t_8698
    )
    residual = torch.rand(                                                     # trace_info : t_8700, t_8704
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_8701
        dtype=dtype,                                                           # trace_info : t_8702
        device="cuda",                                                         # trace_info : t_8703
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(# trace_info : t_8705, t_8707
        residual                                                               # trace_info : t_8706
    )
    dropout_rate = 0.1                                                         # trace_info : t_8708
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(                           # trace_info : t_8709, t_8711, t_8731, t_8751
        [False, True], [True, True], [True, True]                              # trace_info : t_8710
    ):
        input.requires_grad = input_grad                                       # trace_info : t_8712, t_8732
        bias.requires_grad = bias_grad                                         # trace_info : t_8713, t_8733
        residual.requires_grad = residual_grad                                 # trace_info : t_8714, t_8734
        for _ in range(5):                                                     # trace_info : t_8715, t_8718, t_8721, t_8724, t_8727, ...
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)# trace_info : t_8716, t_8719, t_8722, t_8725, t_8728, ...
    del bias, input, residual, output                                          # trace_info : t_8752
    torch.cuda.empty_cache()                                                   # trace_info : t_8753
