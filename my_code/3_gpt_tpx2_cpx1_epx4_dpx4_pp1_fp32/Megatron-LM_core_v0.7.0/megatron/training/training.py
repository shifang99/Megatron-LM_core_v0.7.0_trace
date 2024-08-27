# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import dataclasses
from datetime import datetime
import gc
import logging
import math
import os
import sys
from .log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config, StragglerDetector
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.pipeline_parallel import get_forward_backward_func
from .async_utils import maybe_finalize_async_save
from .utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    append_to_progress_log,
)
from .global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)


stimer = StragglerDetector()

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()                                                # trace_info : t_8694, t_15886, t_17407, t_17574, t_30594
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                    # trace_info : t_8695, t_15887, t_17408, t_17575, t_30595
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))            # trace_info : t_8696, t_15888, t_17409, t_17576, t_30596


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads        # trace_info : t_21727, t_26072, t_30417
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size# trace_info : t_21728, t_26073, t_30418
    # Group Query Attention.
    if not args.group_query_attention:                                         # trace_info : t_21729, t_26074, t_30419
        args.num_query_groups = args.num_attention_heads                       # trace_info : t_21730, t_26075, t_30420
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk# trace_info : t_21731, t_26076, t_30421
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1                      # trace_info : t_21732, t_26077, t_30422
    return (                                                                   # trace_info : t_21760, t_26105, t_30450
        12                                                                     # trace_info : t_21733, t_21735, t_21737, t_21739, t_21741, ...
        * batch_size                                                           # trace_info : t_21734, t_26079, t_30424
        * args.seq_length                                                      # trace_info : t_21736, t_26081, t_30426
        * args.num_layers                                                      # trace_info : t_21738, t_26083, t_30428
        * args.hidden_size                                                     # trace_info : t_21740, t_26085, t_30430
        * args.hidden_size                                                     # trace_info : t_21742, t_26087, t_30432
        * (
            # Attention.
            (                                                                  # trace_info : t_21756, t_21758, t_26101, t_26103, t_30446, ...
                (                                                              # trace_info : t_21750, t_26095, t_30440
                    1                                                          # trace_info : t_21744, t_21746, t_21748, t_26089, t_26091, ...
                    + (args.num_query_groups / args.num_attention_heads)       # trace_info : t_21745, t_26090, t_30435
                    + (args.seq_length / args.hidden_size)                     # trace_info : t_21747, t_26092, t_30437
                ) * query_projection_to_hidden_size_ratio                      # trace_info : t_21749, t_26094, t_30439
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)                      # trace_info : t_21751, t_21753, t_21755, t_26096, t_26098, ...
                * num_experts_routed_to                                        # trace_info : t_21752, t_26097, t_30442
                * gated_linear_multiplier                                      # trace_info : t_21754, t_26099, t_30444
            )
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))# trace_info : t_21757, t_26102, t_30447
        )
    )


def get_start_time_from_progress_log():
    """
    Gets start time of earliest job with same world size. Also returns the number
    of floating-point operations completed in last saved checkpoint.
    """
    args = get_args()
    assert args.save is not None
    progress_log_filename = os.path.join(args.save, "progress.txt")

    # start_time is time when job with same world size started.
    # start_num_floating_point_operations is the number of floating-point operations
    # completed when this job started.
    # latest_num_floating_point_operations is the number of floating-point operations
    # completed in most recent saved checkpoint.
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0

    def _get_field(string, type):
        return type(string.split(': ')[1])

    with open(progress_log_filename, 'r') as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split('\t')
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = \
                    _get_field(line_tokens[7], float)
            if world_size_in_line != args.world_size:
                # Re-start search if we see a different world size.
                start_time = None
                start_num_floating_point_operations = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = \
                        latest_num_floating_point_operations
    assert start_time is not None and start_num_floating_point_operations is not None, \
        "Should have seen at least one 'Starting job' entry with same world_size"
    return datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), \
        start_num_floating_point_operations


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,               # trace_info : t_1, t_3
                        args_defaults=args_defaults)                           # trace_info : t_2

    args = get_args()                                                          # trace_info : t_8534
    timers = get_timers()                                                      # trace_info : t_8538

    if args.log_progress:                                                      # trace_info : t_8542
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()                                                   # trace_info : t_8543

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],                      # trace_info : t_8679, t_8682
                                     dtype=torch.double,                       # trace_info : t_8680
                                     device='cuda')                            # trace_info : t_8681
    torch.distributed.all_reduce(start_time_tensor,                            # trace_info : t_8683, t_8685
                                 op=torch.distributed.ReduceOp.MIN)            # trace_info : t_8684
    _TRAIN_START_TIME = start_time_tensor.item()                               # trace_info : t_8686
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(       # trace_info : t_8687, t_8689
        time.time() - _TRAIN_START_TIME))                                      # trace_info : t_8688
    print_datetime('after megatron is initialized')                            # trace_info : t_8693

    args = get_args()                                                          # trace_info : t_8700
    timers = get_timers()                                                      # trace_info : t_8704

    one_logger = get_one_logger()                                              # trace_info : t_8708
    if one_logger:                                                             # trace_info : t_8710
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)       # trace_info : t_8711
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(         # trace_info : t_8732, t_8734
        model_provider, model_type)                                            # trace_info : t_8733

    timers('model-and-optimizer-setup').stop()                                 # trace_info : t_15874
    print_datetime('after model, optimizer, and learning rate '                # trace_info : t_15885
                   'scheduler are built')
    config = get_model_config(model[0])                                        # trace_info : t_15892

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(        # trace_info : t_15901, t_15917
        barrier=True)                                                          # trace_info : t_15916
    if args.virtual_pipeline_model_parallel_size is not None:                  # trace_info : t_15924
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \         # trace_info : t_17394
            = build_train_valid_test_data_iterators(                           # trace_info : t_15925, t_15927
                train_valid_test_dataset_provider)                             # trace_info : t_15926
    timers('train/valid/test-data-iterators-setup').stop()                     # trace_info : t_17395
    print_datetime('after dataloaders are built')                              # trace_info : t_17406

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = {}                                                 # trace_info : t_17413

    # Print setup timing.
    print_rank_0('done with setup ...')                                        # trace_info : t_17414
    timers.log(['model-and-optimizer-setup',                                   # trace_info : t_17418, t_17420, t_17422
                'train/valid/test-data-iterators-setup'], barrier=True)        # trace_info : t_17419, t_17421

    if not args.skip_train:                                                    # trace_info : t_17508
        print_rank_0('training ...')                                           # trace_info : t_17509

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:        # trace_info : t_17513
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0                                                          # trace_info : t_17514
        if args.do_train and args.train_iters > 0:                             # trace_info : t_17515
            iteration, num_floating_point_operations_so_far = train(           # trace_info : t_17516, t_17521
                forward_step_func,                                             # trace_info : t_17517
                model, optimizer, opt_param_scheduler,                         # trace_info : t_17518
                train_data_iterator, valid_data_iterator,                      # trace_info : t_17519
                process_non_loss_data_func, config, checkpointing_context)     # trace_info : t_17520

        print_datetime('after training is done')                               # trace_info : t_30593

        if args.save and iteration != 0 and iteration % args.save_interval != 0:# trace_info : t_30600
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:                                                          # trace_info : t_30601
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:                                                           # trace_info : t_30602
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    maybe_finalize_async_save(blocking=True)                                   # trace_info : t_30603



def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()                                                          # trace_info : t_8744
    args.model_type = model_type                                               # trace_info : t_8748

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \                  # trace_info : t_8749
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()                            # trace_info : t_8754
        post_process = mpu.is_pipeline_last_stage()                            # trace_info : t_8763
        add_encoder = True                                                     # trace_info : t_8778
        add_decoder = True                                                     # trace_info : t_8779
        if model_type == ModelType.encoder_and_decoder:                        # trace_info : t_8780
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(                                       # trace_info : t_8781, t_8784
                pre_process=pre_process,                                       # trace_info : t_8782
                post_process=post_process                                      # trace_info : t_8783
            )
        model.model_type = model_type                                          # trace_info : t_11925

    if not isinstance(model, list):                                            # trace_info : t_11926
        model = [model]                                                        # trace_info : t_11927

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:                                                 # trace_info : t_11928, t_12371
        for param in model_module.parameters():                                # trace_info : t_11929, t_11942, t_11958, t_11974, t_11990, ...
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)# trace_info : t_11930, t_11943, t_11959, t_11975, t_11991, ...

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:                                      # trace_info : t_12372
        print(' > number of parameters on (tensor, pipeline) '                 # trace_info : t_12380, t_12398
              'model parallel rank ({}, {}): {}'.format(                       # trace_info : t_12381, t_12396
            mpu.get_tensor_model_parallel_rank(),                              # trace_info : t_12382
            mpu.get_pipeline_model_parallel_rank(),                            # trace_info : t_12388
            sum([sum([p.nelement() for p in model_module.parameters()])        # trace_info : t_12393, t_12395
                 for model_module in model])), flush=True)                     # trace_info : t_12394, t_12397

    # GPU allocation.
    for model_module in model:                                                 # trace_info : t_12399, t_12401
        model_module.cuda(torch.cuda.current_device())                         # trace_info : t_12400

    # Fp16 conversion.
    if args.fp16 or args.bf16:                                                 # trace_info : t_12402
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:                                                          # trace_info : t_12403
        config = get_model_config(model[0])                                    # trace_info : t_12404
        ddp_config = DistributedDataParallelConfig(                            # trace_info : t_12413, t_12419
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,       # trace_info : t_12414
            overlap_grad_reduce=args.overlap_grad_reduce,                      # trace_info : t_12415
            use_distributed_optimizer=args.use_distributed_optimizer,          # trace_info : t_12416
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,         # trace_info : t_12417
            bucket_size=args.ddp_bucket_size)                                  # trace_info : t_12418
        model = [DDP(config,                                                   # trace_info : t_12425, t_12427
                     ddp_config,
                     model_chunk,
                     data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                     expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]       # trace_info : t_12426

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:                                     # trace_info : t_14959
            for model_module in model:
                model_module.broadcast_params()

    return model                                                               # trace_info : t_14960


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()                                                          # trace_info : t_15738

    # Iteration-based training.
    if args.train_iters:                                                       # trace_info : t_15742
        if args.lr_decay_iters is None:                                        # trace_info : t_15743
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size          # trace_info : t_15744
        wd_incr_steps = args.train_iters * args.global_batch_size              # trace_info : t_15745
        if args.lr_warmup_fraction is not None:                                # trace_info : t_15746
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps         # trace_info : t_15747
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(                             # trace_info : t_15748, t_15762
        optimizer,                                                             # trace_info : t_15749
        init_lr=args.lr_warmup_init,                                           # trace_info : t_15750
        max_lr=args.lr,                                                        # trace_info : t_15751
        min_lr=args.min_lr,                                                    # trace_info : t_15752
        lr_warmup_steps=lr_warmup_steps,                                       # trace_info : t_15753
        lr_decay_steps=lr_decay_steps,                                         # trace_info : t_15754
        lr_decay_style=args.lr_decay_style,                                    # trace_info : t_15755
        start_wd=args.start_weight_decay,                                      # trace_info : t_15756
        end_wd=args.end_weight_decay,                                          # trace_info : t_15757
        wd_incr_steps=wd_incr_steps,                                           # trace_info : t_15758
        wd_incr_style=args.weight_decay_incr_style,                            # trace_info : t_15759
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,# trace_info : t_15760
        override_opt_param_scheduler=args.override_opt_param_scheduler)        # trace_info : t_15761

    return opt_param_scheduler                                                 # trace_info : t_15866


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()                                                          # trace_info : t_8735
    timers = get_timers()                                                      # trace_info : t_8739

    model = get_model(model_provider_func, model_type)                         # trace_info : t_8743
    unwrapped_model = unwrap_model(model)                                      # trace_info : t_14961

    kwargs = {}                                                                # trace_info : t_14973
    for f in dataclasses.fields(OptimizerConfig):                              # trace_info : t_14974, t_14977, t_14980, t_14983, t_14986, ...
        if hasattr(args, f.name):                                              # trace_info : t_14975, t_14978, t_14981, t_14984, t_14987, ...
            kwargs[f.name] = getattr(args, f.name)                             # trace_info : t_14976, t_14979, t_14982, t_14985, t_14988, ...
    config = OptimizerConfig(**kwargs)                                         # trace_info : t_15049
    config.timers = timers                                                     # trace_info : t_15075
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,        # trace_info : t_15076, t_15078
                                       scale_lr_cond, lr_mult)                 # trace_info : t_15077
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)             # trace_info : t_15737

    if args.load is not None or args.pretrained_checkpoint is not None:        # trace_info : t_15867
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0                                                     # trace_info : t_15868
        args.num_floating_point_operations_so_far = 0                          # trace_info : t_15869

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \                     # trace_info : t_15870, t_15872
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):          # trace_info : t_15871
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler                               # trace_info : t_15873



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()                                                          # trace_info : t_17624, t_21917, t_26262
    timers = get_timers()                                                      # trace_info : t_17628, t_21921, t_26266

    # Set grad to zero.
    for model_chunk in model:                                                  # trace_info : t_17632, t_17746, t_21925, t_22039, t_26270, ...
        model_chunk.zero_grad_buffer()                                         # trace_info : t_17633, t_21926, t_26271
    optimizer.zero_grad()                                                      # trace_info : t_17747, t_22040, t_26385

    # Forward pass.
    forward_backward_func = get_forward_backward_func()                        # trace_info : t_17827, t_22180, t_26525
    losses_reduced = forward_backward_func(                                    # trace_info : t_17836, t_17847, t_22189, t_22200, t_26534, ...
        forward_step_func=forward_step_func,                                   # trace_info : t_17837, t_22190, t_26535
        data_iterator=data_iterator,                                           # trace_info : t_17838, t_22191, t_26536
        model=model,                                                           # trace_info : t_17839, t_22192, t_26537
        num_microbatches=get_num_microbatches(),                               # trace_info : t_17840, t_22193, t_26538
        seq_length=args.seq_length,                                            # trace_info : t_17843, t_22196, t_26541
        micro_batch_size=args.micro_batch_size,                                # trace_info : t_17844, t_22197, t_26542
        decoder_seq_length=args.decoder_seq_length,                            # trace_info : t_17845, t_22198, t_26543
        forward_only=False)                                                    # trace_info : t_17846, t_22199, t_26544

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:                                    # trace_info : t_20589, t_24934, t_29279
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_20590, t_24935, t_29280
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)  # trace_info : t_20591, t_24936, t_29281
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()         # trace_info : t_20598, t_24943, t_29288
    timers('optimizer').stop()                                                 # trace_info : t_21589, t_25934, t_30279

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_21597, t_25942, t_30287
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:                                                      # trace_info : t_21598, t_25943, t_30288
        increment = get_num_microbatches() * \                                 # trace_info : t_21599, t_21603, t_21605, t_25944, t_25948, ...
                    args.micro_batch_size * \                                  # trace_info : t_21602, t_25947, t_30292
                    args.data_parallel_size                                    # trace_info : t_21604, t_25949, t_30294
        opt_param_scheduler.step(increment=increment)                          # trace_info : t_21606, t_25951, t_30296
        skipped_iter = 0                                                       # trace_info : t_21683, t_26028, t_30373
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:                                    # trace_info : t_21684, t_26029, t_30374
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):                        # trace_info : t_21685, t_26030, t_30375
        # Average loss across microbatches.
        loss_reduced = {}                                                      # trace_info : t_21696, t_26041, t_30386
        for key in losses_reduced[0].keys():                                   # trace_info : t_21697, t_21707, t_26042, t_26052, t_30387, ...
            numerator = 0                                                      # trace_info : t_21698, t_26043, t_30388
            denominator = 0                                                    # trace_info : t_21699, t_26044, t_30389
            for x in losses_reduced:                                           # trace_info : t_21700, t_21705, t_26045, t_26050, t_30390, ...
                val = x[key]                                                   # trace_info : t_21701, t_26046, t_30391
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):            # trace_info : t_21702, t_26047, t_30392
                    numerator += val[0]                                        # trace_info : t_21703, t_26048, t_30393
                    denominator += val[1]                                      # trace_info : t_21704, t_26049, t_30394
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator                        # trace_info : t_21706, t_26051, t_30396
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad        # trace_info : t_21708, t_26053, t_30398
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()                                                          # trace_info : t_21800, t_26145, t_30490
    timers = get_timers()                                                      # trace_info : t_21804, t_26149, t_30494
    writer = get_tensorboard_writer()                                          # trace_info : t_21808, t_26153, t_30498
    wandb_writer = get_wandb_writer()                                          # trace_info : t_21810, t_26155, t_30500
    one_logger = get_one_logger()                                              # trace_info : t_21812, t_26157, t_30502

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'                                 # trace_info : t_21814, t_26159, t_30504
    skipped_iters_key = 'skipped iterations'                                   # trace_info : t_21815, t_26160, t_30505
    nan_iters_key = 'nan iterations'                                           # trace_info : t_21816, t_26161, t_30506
    # Advanced iterations.
    if not skipped_iter:                                                       # trace_info : t_21817, t_26162, t_30507
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(             # trace_info : t_21818, t_21820, t_21822, t_26163, t_26165, ...
            advanced_iters_key, 0) + 1                                         # trace_info : t_21819, t_21821, t_26164, t_26166, t_30509, ...
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(                  # trace_info : t_21823, t_21825, t_21827, t_26168, t_26170, ...
        skipped_iters_key, 0) + skipped_iter                                   # trace_info : t_21824, t_21826, t_26169, t_26171, t_30514, ...
    # Update losses and set nan iterations
    got_nan = False                                                            # trace_info : t_21828, t_26173, t_30518
    for key in loss_dict:                                                      # trace_info : t_21829, t_21836, t_26174, t_26181, t_30519, ...
        if not skipped_iter:                                                   # trace_info : t_21830, t_26175, t_30520
            total_loss_dict[key] = total_loss_dict.get(                        # trace_info : t_21831, t_21833, t_21835, t_26176, t_26178, ...
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]# trace_info : t_21832, t_21834, t_26177, t_26179, t_30522, ...
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(                      # trace_info : t_21837, t_21839, t_21841, t_26182, t_26184, ...
        nan_iters_key, 0) + int(got_nan)                                       # trace_info : t_21838, t_21840, t_26183, t_26185, t_30528, ...

    # Logging.
    timers_to_log = [                                                          # trace_info : t_21842, t_26187, t_30532
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \           # trace_info : t_21843, t_21847, t_26188, t_26192, t_30533, ...
        get_num_microbatches()                                                 # trace_info : t_21844, t_26189, t_30534

    # Track app tag & app tag ID
    if one_logger:                                                             # trace_info : t_21848, t_26193, t_30538
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \                 # trace_info : t_21849, t_21851, t_26194, t_26196, t_30539, ...
                       total_loss_dict[skipped_iters_key]                      # trace_info : t_21850, t_26195, t_30540

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \                                    # trace_info : t_21852, t_26197, t_30542
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):            # trace_info : t_21853, t_26198, t_30543
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            if args.decoupled_lr is not None:
                writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:                                           # trace_info : t_21854, t_26199, t_30544
        moe_loss_scale = 1 / get_num_microbatches()                            # trace_info : t_21855, t_26200, t_30545
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)# trace_info : t_21858, t_26203, t_30548

    if iteration % args.log_interval == 0:                                     # trace_info : t_21871, t_26216, t_30561
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)
        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        assert learning_rate is not None
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            assert decoupled_learning_rate is None
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag                                                  # trace_info : t_21872, t_26217, t_30562


def compute_throughputs_and_append_to_progress_log(iteration,
                                                   num_floating_point_operations_so_far):
    args = get_args()
    if args.save is None:
        return

    # Compute job throughput.
    # args.num_floating_point_operations_so_far keeps track of floating-point operations
    # completed at the start of job.
    global _TRAIN_START_TIME
    job_throughput = \
        (num_floating_point_operations_so_far -
         args.num_floating_point_operations_so_far) / (
            (time.time() - _TRAIN_START_TIME) * 10**12 * args.world_size)

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = \
        (num_floating_point_operations_so_far -
         start_num_floating_point_operations) / (
            elapsed_time * 10**12 * args.world_size)

    tokens_so_far = args.consumed_train_samples * args.seq_length
    saved_ckpt_prefix = 'Saving async checkpoint' if args.async_save else 'Saved checkpoint'
    append_to_progress_log(f"{saved_ckpt_prefix}\tIteration: {iteration}\t"
                           f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
                           f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
                           f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
                           f"Tokens (in billions): {tokens_so_far / 10**9:.2f}")


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler,
                             num_floating_point_operations_so_far, checkpointing_context):
    args = get_args()
    timers = get_timers()
    # Extra barrier is added to make sure all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far, checkpointing_context)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])

    if args.log_progress:
        compute_throughputs_and_append_to_progress_log(iteration,
                                                       num_floating_point_operations_so_far)


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config, checkpointing_context):
    """Train the model function."""
    args = get_args()                                                          # trace_info : t_17522
    timers = get_timers()                                                      # trace_info : t_17526

    # Write args to tensorboard
    write_args_to_tensorboard()                                                # trace_info : t_17530

    # Turn on training mode which enables dropout.
    for model_module in model:                                                 # trace_info : t_17538, t_17540
        model_module.train()                                                   # trace_info : t_17539

    # Tracking loss.
    total_loss_dict = {}                                                       # trace_info : t_17541

    # Iterations.
    iteration = args.iteration                                                 # trace_info : t_17542
    one_logger = get_one_logger()                                              # trace_info : t_17543
    if one_logger:                                                             # trace_info : t_17545
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics({
            'train_samples_start': args.consumed_train_samples,
            'train_iterations_start': iteration,
            'train_samples_target': train_samples_target,
            'train_iterations_target': args.train_iters,
        })

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far# trace_info : t_17546

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss                              # trace_info : t_17547
    config.timers = timers                                                     # trace_info : t_17548
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:                 # trace_info : t_17549
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.delay_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:                  # trace_info : t_17550
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads                    # trace_info : t_17551

    timers('interval-time', log_level=0).start(barrier=True)                   # trace_info : t_17552
    print_datetime('before the start of training step')                        # trace_info : t_17573
    report_memory_flag = True                                                  # trace_info : t_17580
    exit = False                                                               # trace_info : t_17581

    if args.manual_gc:                                                         # trace_info : t_17582
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:                                                     # trace_info : t_17583
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    total_flops = 0.0                                                          # trace_info : t_17584

    num_microbatches = get_num_microbatches()                                  # trace_info : t_17585
    eval_duration = 0.0                                                        # trace_info : t_17588
    eval_iterations = 0                                                        # trace_info : t_17589
    def track_e2e_metrics():                                                   # trace_info : t_17590
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:                                                         # trace_info : t_30576
            train_duration = timers('interval-time').active_time()  # overall_elapsed
            train_samples = args.consumed_train_samples - train_samples_start
            train_iterations = iteration - iteration_start
            train_iterations_time_msecs_avg = (train_duration * 1000.0) / train_iterations
            if eval_iterations:
                validation_iterations_time_msecs_avg = (eval_duration * 1000.0) / eval_iterations
            else:
                validation_iterations_time_msecs_avg = None

            one_logger.log_metrics({
                'train_iterations_end': iteration,
                'train_samples_end': args.consumed_train_samples,
                'train_iterations': train_iterations,
                'train_samples': train_samples,
                'train_iterations_time_msecs_avg': train_iterations_time_msecs_avg,
                'validation_iterations_time_msecs_avg': validation_iterations_time_msecs_avg
            })

    while iteration < args.train_iters:                                        # trace_info : t_17591, t_21884, t_26229, t_30574
        if args.profile and \                                                  # trace_info : t_17592, t_21885, t_26230
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        maybe_finalize_async_save(False)                                       # trace_info : t_17593, t_21886, t_26231

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)# trace_info : t_17600, t_21893, t_26238
        if get_num_microbatches() != num_microbatches and iteration != 0:      # trace_info : t_17605, t_21898, t_26243
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
        num_microbatches = get_num_microbatches()                              # trace_info : t_17608, t_21901, t_26246
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)# trace_info : t_17611, t_21904, t_26249

        args.curr_iteration = iteration                                        # trace_info : t_17616, t_21909, t_26254
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \              # trace_info : t_21709, t_26054, t_30399
            train_step(forward_step_func,                                      # trace_info : t_17617, t_17623, t_21910, t_21916, t_26255, ...
                       train_data_iterator,                                    # trace_info : t_17618, t_21911, t_26256
                       model,                                                  # trace_info : t_17619, t_21912, t_26257
                       optimizer,                                              # trace_info : t_17620, t_21913, t_26258
                       opt_param_scheduler,                                    # trace_info : t_17621, t_21914, t_26259
                       config)                                                 # trace_info : t_17622, t_21915, t_26260
        iteration += 1                                                         # trace_info : t_21710, t_26055, t_30400
        batch_size = mpu.get_data_parallel_world_size() * \                    # trace_info : t_21711, t_21720, t_21724, t_26056, t_26065, ...
                     args.micro_batch_size * \                                 # trace_info : t_21719, t_26064, t_30409
                     get_num_microbatches()                                    # trace_info : t_21721, t_26066, t_30411
        args.consumed_train_samples += batch_size                              # trace_info : t_21725, t_26070, t_30415
        num_fp_ops = num_floating_point_operations(args, batch_size)           # trace_info : t_21726, t_26071, t_30416
        num_floating_point_operations_so_far += num_fp_ops                     # trace_info : t_21761, t_26106, t_30451
        total_flops += num_fp_ops                                              # trace_info : t_21762, t_26107, t_30452

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()                         # trace_info : t_21763, t_26108, t_30453
        params_norm = None                                                     # trace_info : t_21766, t_26111, t_30456
        if args.log_params_norm:                                               # trace_info : t_21767, t_26112, t_30457
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:                                 # trace_info : t_21768, t_26113, t_30458
            track_e2e_metrics()

        learning_rate = None                                                   # trace_info : t_21769, t_26114, t_30459
        decoupled_learning_rate = None                                         # trace_info : t_21770, t_26115, t_30460
        for param_group in optimizer.param_groups:                             # trace_info : t_21771, t_21783, t_21786, t_21789, t_21792, ...
            if param_group['is_decoupled_lr']:                                 # trace_info : t_21781, t_21784, t_21787, t_21790, t_26126, ...
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']                              # trace_info : t_21782, t_21785, t_21788, t_21791, t_26127, ...
        report_memory_flag = training_log(loss_dict, total_loss_dict,          # trace_info : t_21793, t_21799, t_26138, t_26144, t_30483, ...
                                          learning_rate,                       # trace_info : t_21794, t_26139, t_30484
                                          decoupled_learning_rate,             # trace_info : t_21795, t_26140, t_30485
                                          iteration, loss_scale,               # trace_info : t_21796, t_26141, t_30486
                                          report_memory_flag, skipped_iter,    # trace_info : t_21797, t_26142, t_30487
                                          grad_norm, params_norm, num_zeros_in_grad)# trace_info : t_21798, t_26143, t_30488
        # StragglerDetector
        if iteration % args.log_interval == 0 and args.log_straggler:          # trace_info : t_21873, t_26218, t_30563
            stimer.report(total_flops, args.log_interval)
            total_flops = 0.0

        if args.check_weight_hash_across_dp_replicas_interval is not None and \# trace_info : t_21874, t_26219, t_30564
                iteration % args.check_weight_hash_across_dp_replicas_interval == 0:
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.disable_pre_hook()
            assert check_param_hashes_across_dp_replicas(model), \
                "Parameter hashes not matching across DP replicas"
            torch.distributed.barrier()
            print_rank_0(f">>> Weight hashes match after {iteration} iterations...")
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.enable_pre_hook()

        # Autoresume
        if args.adlr_autoresume and \                                          # trace_info : t_21875, t_26220, t_30565
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \    # trace_info : t_21876, t_26221, t_30566
           args.do_valid:
            timers('interval-time').stop()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.disable_pre_hook()
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = 'iteration {}'.format(iteration)
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, False)
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += args.eval_iters
            timers('eval-time').stop()
            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.enable_pre_hook()
            timers('interval-time', log_level=0).start(barrier=True)

        # Checkpointing
        saved_checkpoint = False                                               # trace_info : t_21877, t_26222, t_30567
        if args.exit_signal_handler:                                           # trace_info : t_21878, t_26223, t_30568
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \                              # trace_info : t_21879, t_26224, t_30569
           iteration % args.save_interval == 0:
            timers('interval-time').stop()
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
            saved_checkpoint = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Exiting based on duration
        if args.exit_duration_in_mins:                                         # trace_info : t_21880, t_26225, t_30570
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int, device='cuda')
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler,
                                             num_floating_point_operations_so_far,
                                             checkpointing_context)
                print_datetime('exiting program after {} minutes'.format(train_time))
                exit = True
                break

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:         # trace_info : t_21881, t_26226, t_30571
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.profile and \                                                  # trace_info : t_21882, t_26227, t_30572
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()

        if args.manual_gc:                                                     # trace_info : t_21883, t_26228, t_30573
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    track_e2e_metrics()                                                        # trace_info : t_30575

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()                                          # trace_info : t_30577
    if writer:                                                                 # trace_info : t_30579
        writer.flush()
    wandb_writer = get_wandb_writer()                                          # trace_info : t_30580
    if wandb_writer:                                                           # trace_info : t_30582
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:           # trace_info : t_30583
        optimizer.disable_pre_hook()

    maybe_finalize_async_save(True)                                            # trace_info : t_30584

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:                                                                   # trace_info : t_30591
        sys.exit()

    return iteration, num_floating_point_operations_so_far                     # trace_info : t_30592


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if key not in total_loss_dict:
                            total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                        val = loss_dict[key]
                        if isinstance(val, tuple) or isinstance(val, list):
                            total_loss_dict[key][0] += val[0]
                            total_loss_dict[key][1] += val[1]
                        else:
                            total_loss_dict[key][0] += val
                            total_loss_dict[key][1] += 1

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = numerator / denominator

    timers('evaluate').stop()
    timers.log(['evaluate'])

    return total_loss_dict, collected_non_loss_data, False

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose)
    # Timelimit hit during evaluation
    if timelimit:
        return
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)
            if wandb_writer and is_last_rank():
                wandb_writer.log({
                    '{} validation'.format(key): total_loss_dict[key].item()},
                    iteration)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args()                                                          # trace_info : t_15952

    # Number of train/valid/test samples.
    if args.train_samples:                                                     # trace_info : t_15956
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size              # trace_info : t_15957
    eval_iters = (args.train_iters // args.eval_interval + 1) * \              # trace_info : t_15958, t_15960
                 args.eval_iters                                               # trace_info : t_15959
    test_iters = args.eval_iters                                               # trace_info : t_15961

    return (                                                                   # trace_info : t_15965
        train_samples,                                                         # trace_info : t_15962
        eval_iters * args.global_batch_size,                                   # trace_info : t_15963
        test_iters * args.global_batch_size,                                   # trace_info : t_15964
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()          # trace_info : t_15951
    print_rank_0(' > datasets target sizes (minimum size):')                   # trace_info : t_15966
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0])) # trace_info : t_15970
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1])) # trace_info : t_15974
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2])) # trace_info : t_15978
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)# trace_info : t_15982


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()                                                          # trace_info : t_15935

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None) # trace_info : t_15939

    print_rank_0('> building train, validation, and test datasets ...')        # trace_info : t_15940

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:                # trace_info : t_15944
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:                # trace_info : t_15945
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)# trace_info : t_15946

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:            # trace_info : t_15947

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(         # trace_info : t_15948, t_15950
            build_train_valid_test_datasets_provider)                          # trace_info : t_15949
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(                      # trace_info : t_16997, t_16999
            train_ds, args.consumed_train_samples)                             # trace_info : t_16998
        if args.skip_train:                                                    # trace_info : t_17046
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(                  # trace_info : t_17047, t_17049
                valid_ds, args.consumed_valid_samples)                         # trace_info : t_17048
        test_dataloader = build_pretraining_data_loader(test_ds, 0)            # trace_info : t_17096

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0       # trace_info : t_17143
        do_valid = valid_dataloader is not None and args.eval_iters > 0        # trace_info : t_17144
        do_test = test_dataloader is not None and args.eval_iters > 0          # trace_info : t_17145
        flags = torch.tensor(                                                  # trace_info : t_17146, t_17149
            [int(do_train), int(do_valid), int(do_test)],                      # trace_info : t_17147
            dtype=torch.long, device='cuda')                                   # trace_info : t_17148
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)                                      # trace_info : t_17150

    args.do_train = getattr(args, "do_train", False) or flags[0].item()        # trace_info : t_17151
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()        # trace_info : t_17152
    args.do_test = getattr(args, "do_test", False) or flags[2].item()          # trace_info : t_17153

    return train_dataloader, valid_dataloader, test_dataloader                 # trace_info : t_17154


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()                                                          # trace_info : t_15928

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \                    # trace_info : t_17155
        build_train_valid_test_data_loaders(                                   # trace_info : t_15932, t_15934
            build_train_valid_test_datasets_provider)                          # trace_info : t_15933

    # Build iterators.
    dl_type = args.dataloader_type                                             # trace_info : t_17156
    assert dl_type in ['single', 'cyclic', 'external']                         # trace_info : t_17157

    def _get_iterator(dataloader_type, dataloader):                            # trace_info : t_17158
        """Return dataset iterator."""
        if dataloader_type == "single":                                        # trace_info : t_17161, t_17239, t_17317
            return iter(dataloader)                                            # trace_info : t_17162, t_17240, t_17318
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:                                           # trace_info : t_17159
        train_data_iterator = _get_iterator(dl_type, train_dataloader)         # trace_info : t_17160
    else:
        train_data_iterator = None

    if valid_dataloader is not None:                                           # trace_info : t_17237
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)         # trace_info : t_17238
    else:
        valid_data_iterator = None

    if test_dataloader is not None:                                            # trace_info : t_17315
        test_data_iterator = _get_iterator(dl_type, test_dataloader)           # trace_info : t_17316
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator        # trace_info : t_17393
