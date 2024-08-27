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
    def finish_mpu_init():                                                     # trace_info : t_4058
        args = get_args()                                                      # trace_info : t_4066
        # Pytorch distributed.
        _initialize_distributed()                                              # trace_info : t_4070

        # Random seeds for reproducibility.
        if args.rank == 0:                                                     # trace_info : t_5391
            print("> setting random seeds to {} ...".format(args.seed))        # trace_info : t_5392
        _set_random_seed(args.seed, args.data_parallel_random_init)            # trace_info : t_5393

    if skip_mpu_initialization:                                                # trace_info : t_4059
        return None

    args = get_args()                                                          # trace_info : t_4060
    if args.lazy_mpu_init:                                                     # trace_info : t_4064
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
        finish_mpu_init()                                                      # trace_info : t_4065

        # Autoresume.
        _init_autoresume()                                                     # trace_info : t_5506

        # Compile dependencies.
        _compile_dependencies()                                                # trace_info : t_5510

        if args.tp_comm_overlap:                                               # trace_info : t_5583
           _initialize_tp_communicators()

        # No continuation function
        return None                                                            # trace_info : t_5584


def _compile_dependencies():

    args = get_args()                                                          # trace_info : t_5511

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_5515
        start_time = time.time()                                               # trace_info : t_5516
        print("> compiling dataset index builder ...")                         # trace_info : t_5517
        from megatron.core.datasets.utils import compile_helpers               # trace_info : t_5518

        compile_helpers()                                                      # trace_info : t_5519
        print(                                                                 # trace_info : t_5524, t_5528
            ">>> done with dataset index builder. Compilation time: {:.3f} "   # trace_info : t_5525
            "seconds".format(time.time() - start_time),                        # trace_info : t_5526
            flush=True,                                                        # trace_info : t_5527
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length                                                  # trace_info : t_5529
    attn_batch_size = (                                                        # trace_info : t_5532
        args.num_attention_heads / args.tensor_model_parallel_size             # trace_info : t_5530
    ) * args.micro_batch_size                                                  # trace_info : t_5531
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (                                               # trace_info : t_5539
        seq_len > 16                                                           # trace_info : t_5533, t_5535, t_5537
        and seq_len <= 16384                                                   # trace_info : t_5534
        and seq_len % 4 == 0                                                   # trace_info : t_5536
        and attn_batch_size % 4 == 0                                           # trace_info : t_5538
    )
    # Print a warning.
    if not (                                                                   # trace_info : t_5541, t_5543, t_5545
        (args.fp16 or args.bf16)                                               # trace_info : t_5540
        and custom_kernel_constraint                                           # trace_info : t_5542
        and args.masked_softmax_fusion                                         # trace_info : t_5544
    ):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_5546
        start_time = time.time()                                               # trace_info : t_5547
        print("> compiling and loading fused kernels ...", flush=True)         # trace_info : t_5548
        fused_kernels.load(args)                                               # trace_info : t_5549
        torch.distributed.barrier()                                            # trace_info : t_5575
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()                                                # trace_info : t_5576
    if torch.distributed.get_rank() == 0:                                      # trace_info : t_5577
        print(                                                                 # trace_info : t_5578, t_5582
            ">>> done with compiling and loading fused kernels. "              # trace_info : t_5579
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),# trace_info : t_5580
            flush=True,                                                        # trace_info : t_5581
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
    args = get_args()                                                          # trace_info : t_4071

    device_count = torch.cuda.device_count()                                   # trace_info : t_4075
    if torch.distributed.is_initialized():                                     # trace_info : t_4076

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:                                                     # trace_info : t_4077
            print("> initializing torch distributed ...", flush=True)          # trace_info : t_4078
        # Manually set the device ids.
        if device_count > 0:                                                   # trace_info : t_4079
            device = args.rank % device_count                                  # trace_info : t_4080
            if args.local_rank is not None:                                    # trace_info : t_4081
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device                                       # trace_info : t_4082
            torch.cuda.set_device(device)                                      # trace_info : t_4083
        # Call the init process
        torch.distributed.init_process_group(                                  # trace_info : t_4084, t_4089
            backend=args.distributed_backend,                                  # trace_info : t_4085
            world_size=args.world_size,                                        # trace_info : t_4086
            rank=args.rank,                                                    # trace_info : t_4087
            timeout=timedelta(minutes=args.distributed_timeout_minutes),       # trace_info : t_4088
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:                                                       # trace_info : t_4090
        if mpu.model_parallel_is_initialized():                                # trace_info : t_4091
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(                                     # trace_info : t_4094, t_4104
                args.tensor_model_parallel_size,                               # trace_info : t_4095
                args.pipeline_model_parallel_size,                             # trace_info : t_4096
                args.virtual_pipeline_model_parallel_size,                     # trace_info : t_4097
                args.pipeline_model_parallel_split_rank,                       # trace_info : t_4098
                context_parallel_size=args.context_parallel_size,              # trace_info : t_4099
                expert_model_parallel_size=args.expert_model_parallel_size,    # trace_info : t_4100
                distributed_timeout_minutes=args.distributed_timeout_minutes,  # trace_info : t_4101
                nccl_communicator_config_path=args.nccl_communicator_config_path,# trace_info : t_4102
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',# trace_info : t_4103
            )
            if args.rank == 0:                                                 # trace_info : t_5371
                print(                                                         # trace_info : t_5372, t_5381
                    f"> initialized tensor model parallel with size "          # trace_info : t_5373, t_5380
                    f"{mpu.get_tensor_model_parallel_world_size()}"            # trace_info : t_5374
                )
                print(                                                         # trace_info : t_5382, t_5390
                    f"> initialized pipeline model parallel with size "        # trace_info : t_5383, t_5389
                    f"{mpu.get_pipeline_model_parallel_world_size()}"          # trace_info : t_5384
                )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()                                         # trace_info : t_5507
    if autoresume:                                                             # trace_info : t_5509
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:                                        # trace_info : t_5394
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())          # trace_info : t_5395
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:                                          # trace_info : t_5400
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)                                                      # trace_info : t_5401
        np.random.seed(seed)                                                   # trace_info : t_5402
        torch.manual_seed(seed)                                                # trace_info : t_5403
        if torch.cuda.device_count() > 0:                                      # trace_info : t_5404
            tensor_parallel.model_parallel_cuda_manual_seed(seed)              # trace_info : t_5405
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()                                                          # trace_info : t_14464
    writer = get_tensorboard_writer()                                          # trace_info : t_14468
    if writer:                                                                 # trace_info : t_14470
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])                         # trace_info : t_5595
    TORCH_MINOR = int(torch.__version__.split(".")[1])                         # trace_info : t_5596
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):          # trace_info : t_5597
        # nvfuser
        torch._C._jit_set_profiling_executor(True)                             # trace_info : t_5598
        torch._C._jit_set_profiling_mode(True)                                 # trace_info : t_5599
        torch._C._jit_override_can_fuse_on_cpu(False)                          # trace_info : t_5600
        torch._C._jit_override_can_fuse_on_gpu(False)                          # trace_info : t_5601
        torch._C._jit_set_texpr_fuser_enabled(False)                           # trace_info : t_5602
        torch._C._jit_set_nvfuser_enabled(True)                                # trace_info : t_5603
        torch._C._debug_set_autodiff_subgraph_inlining(False)                  # trace_info : t_5604
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()                                                     # trace_info : t_5605


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()                                                          # trace_info : t_5606
    if args.bf16:                                                              # trace_info : t_5610
        dtype = torch.bfloat16
    elif args.fp16:                                                            # trace_info : t_5611
        dtype = torch.float16                                                  # trace_info : t_5612
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(                                                         # trace_info : t_5613, t_5617
        args.ffn_hidden_size // args.tensor_model_parallel_size,               # trace_info : t_5614
        dtype=dtype,                                                           # trace_info : t_5615
        device="cuda",                                                         # trace_info : t_5616
    )
    input = torch.rand(                                                        # trace_info : t_5618, t_5625
        (                                                                      # trace_info : t_5622
            args.seq_length,                                                   # trace_info : t_5619
            args.micro_batch_size,                                             # trace_info : t_5620
            args.ffn_hidden_size // args.tensor_model_parallel_size,           # trace_info : t_5621
        ),
        dtype=dtype,                                                           # trace_info : t_5623
        device="cuda",                                                         # trace_info : t_5624
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):             # trace_info : t_5626, t_5644, t_5662
        bias.requires_grad, input.requires_grad = bias_grad, input_grad        # trace_info : t_5627, t_5645
        for _ in range(5):                                                     # trace_info : t_5628, t_5631, t_5634, t_5637, t_5640, ...
            output = bias_gelu(bias, input)                                    # trace_info : t_5629, t_5632, t_5635, t_5638, t_5641, ...
    del bias, input, output                                                    # trace_info : t_5663

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:                                                 # trace_info : t_5664
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length                                           # trace_info : t_5665
    input = torch.rand(                                                        # trace_info : t_5666, t_5670
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_5667
        dtype=dtype,                                                           # trace_info : t_5668
        device="cuda",                                                         # trace_info : t_5669
    )
    residual = torch.rand(                                                     # trace_info : t_5671, t_5675
        (seq_length, args.micro_batch_size, args.hidden_size),                 # trace_info : t_5672
        dtype=dtype,                                                           # trace_info : t_5673
        device="cuda",                                                         # trace_info : t_5674
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(# trace_info : t_5676, t_5678
        residual                                                               # trace_info : t_5677
    )
    dropout_rate = 0.1                                                         # trace_info : t_5679
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(                           # trace_info : t_5680, t_5682, t_5702, t_5722
        [False, True], [True, True], [True, True]                              # trace_info : t_5681
    ):
        input.requires_grad = input_grad                                       # trace_info : t_5683, t_5703
        bias.requires_grad = bias_grad                                         # trace_info : t_5684, t_5704
        residual.requires_grad = residual_grad                                 # trace_info : t_5685, t_5705
        for _ in range(5):                                                     # trace_info : t_5686, t_5689, t_5692, t_5695, t_5698, ...
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)# trace_info : t_5687, t_5690, t_5693, t_5696, t_5699, ...
    del bias, input, residual, output                                          # trace_info : t_5723
    torch.cuda.empty_cache()                                                   # trace_info : t_5724
