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
    torch.distributed.barrier()                                                # trace_info : t_5740, t_12926, t_14339, t_14507, t_25407
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                    # trace_info : t_5741, t_12927, t_14340, t_14508, t_25408
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))            # trace_info : t_5742, t_12928, t_14341, t_14509, t_25409


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads        # trace_info : t_17981, t_21620, t_25259
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size# trace_info : t_17982, t_21621, t_25260
    # Group Query Attention.
    if not args.group_query_attention:                                         # trace_info : t_17983, t_21622, t_25261
        args.num_query_groups = args.num_attention_heads                       # trace_info : t_17984, t_21623, t_25262
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk# trace_info : t_17985, t_21624, t_25263
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1                      # trace_info : t_17986, t_21625, t_25264
    return (                                                                   # trace_info : t_18014, t_21653, t_25292
        12                                                                     # trace_info : t_17987, t_17989, t_17991, t_17993, t_17995, ...
        * batch_size                                                           # trace_info : t_17988, t_21627, t_25266
        * args.seq_length                                                      # trace_info : t_17990, t_21629, t_25268
        * args.num_layers                                                      # trace_info : t_17992, t_21631, t_25270
        * args.hidden_size                                                     # trace_info : t_17994, t_21633, t_25272
        * args.hidden_size                                                     # trace_info : t_17996, t_21635, t_25274
        * (
            # Attention.
            (                                                                  # trace_info : t_18010, t_18012, t_21649, t_21651, t_25288, ...
                (                                                              # trace_info : t_18004, t_21643, t_25282
                    1                                                          # trace_info : t_17998, t_18000, t_18002, t_21637, t_21639, ...
                    + (args.num_query_groups / args.num_attention_heads)       # trace_info : t_17999, t_21638, t_25277
                    + (args.seq_length / args.hidden_size)                     # trace_info : t_18001, t_21640, t_25279
                ) * query_projection_to_hidden_size_ratio                      # trace_info : t_18003, t_21642, t_25281
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)                      # trace_info : t_18005, t_18007, t_18009, t_21644, t_21646, ...
                * num_experts_routed_to                                        # trace_info : t_18006, t_21645, t_25284
                * gated_linear_multiplier                                      # trace_info : t_18008, t_21647, t_25286
            )
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))# trace_info : t_18011, t_21650, t_25289
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

    args = get_args()                                                          # trace_info : t_5585
    timers = get_timers()                                                      # trace_info : t_5589

    if args.log_progress:                                                      # trace_info : t_5593
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()                                                   # trace_info : t_5594

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],                      # trace_info : t_5725, t_5728
                                     dtype=torch.double,                       # trace_info : t_5726
                                     device='cuda')                            # trace_info : t_5727
    torch.distributed.all_reduce(start_time_tensor,                            # trace_info : t_5729, t_5731
                                 op=torch.distributed.ReduceOp.MIN)            # trace_info : t_5730
    _TRAIN_START_TIME = start_time_tensor.item()                               # trace_info : t_5732
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(       # trace_info : t_5733, t_5735
        time.time() - _TRAIN_START_TIME))                                      # trace_info : t_5734
    print_datetime('after megatron is initialized')                            # trace_info : t_5739

    args = get_args()                                                          # trace_info : t_5746
    timers = get_timers()                                                      # trace_info : t_5750

    one_logger = get_one_logger()                                              # trace_info : t_5754
    if one_logger:                                                             # trace_info : t_5756
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)       # trace_info : t_5757
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(         # trace_info : t_5778, t_5780
        model_provider, model_type)                                            # trace_info : t_5779

    timers('model-and-optimizer-setup').stop()                                 # trace_info : t_12914
    print_datetime('after model, optimizer, and learning rate '                # trace_info : t_12925
                   'scheduler are built')
    config = get_model_config(model[0])                                        # trace_info : t_12932

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(        # trace_info : t_12941, t_12957
        barrier=True)                                                          # trace_info : t_12956
    if args.virtual_pipeline_model_parallel_size is not None:                  # trace_info : t_12964
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
        train_data_iterator, valid_data_iterator, test_data_iterator \         # trace_info : t_14326
            = build_train_valid_test_data_iterators(                           # trace_info : t_12965, t_12967
                train_valid_test_dataset_provider)                             # trace_info : t_12966
    timers('train/valid/test-data-iterators-setup').stop()                     # trace_info : t_14327
    print_datetime('after dataloaders are built')                              # trace_info : t_14338

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = {}                                                 # trace_info : t_14345

    # Print setup timing.
    print_rank_0('done with setup ...')                                        # trace_info : t_14346
    timers.log(['model-and-optimizer-setup',                                   # trace_info : t_14350, t_14352, t_14354
                'train/valid/test-data-iterators-setup'], barrier=True)        # trace_info : t_14351, t_14353

    if not args.skip_train:                                                    # trace_info : t_14441
        print_rank_0('training ...')                                           # trace_info : t_14442

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:        # trace_info : t_14446
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0                                                          # trace_info : t_14447
        if args.do_train and args.train_iters > 0:                             # trace_info : t_14448
            iteration, num_floating_point_operations_so_far = train(           # trace_info : t_14449, t_14454
                forward_step_func,                                             # trace_info : t_14450
                model, optimizer, opt_param_scheduler,                         # trace_info : t_14451
                train_data_iterator, valid_data_iterator,                      # trace_info : t_14452
                process_non_loss_data_func, config, checkpointing_context)     # trace_info : t_14453

        print_datetime('after training is done')                               # trace_info : t_25406

        if args.save and iteration != 0 and iteration % args.save_interval != 0:# trace_info : t_25413
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:                                                          # trace_info : t_25414
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:                                                           # trace_info : t_25415
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    maybe_finalize_async_save(blocking=True)                                   # trace_info : t_25416



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
    args = get_args()                                                          # trace_info : t_5790
    args.model_type = model_type                                               # trace_info : t_5794

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \                  # trace_info : t_5795
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
        pre_process = mpu.is_pipeline_first_stage()                            # trace_info : t_5800
        post_process = mpu.is_pipeline_last_stage()                            # trace_info : t_5809
        add_encoder = True                                                     # trace_info : t_5824
        add_decoder = True                                                     # trace_info : t_5825
        if model_type == ModelType.encoder_and_decoder:                        # trace_info : t_5826
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
            model = model_provider_func(                                       # trace_info : t_5827, t_5830
                pre_process=pre_process,                                       # trace_info : t_5828
                post_process=post_process                                      # trace_info : t_5829
            )
        model.model_type = model_type                                          # trace_info : t_8695

    if not isinstance(model, list):                                            # trace_info : t_8696
        model = [model]                                                        # trace_info : t_8697

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:                                                 # trace_info : t_8698, t_9109
        for param in model_module.parameters():                                # trace_info : t_8699, t_8712, t_8728, t_8744, t_8760, ...
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)# trace_info : t_8700, t_8713, t_8729, t_8745, t_8761, ...

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:                                      # trace_info : t_9110
        print(' > number of parameters on (tensor, pipeline) '                 # trace_info : t_9118, t_9136
              'model parallel rank ({}, {}): {}'.format(                       # trace_info : t_9119, t_9134
            mpu.get_tensor_model_parallel_rank(),                              # trace_info : t_9120
            mpu.get_pipeline_model_parallel_rank(),                            # trace_info : t_9126
            sum([sum([p.nelement() for p in model_module.parameters()])        # trace_info : t_9131, t_9133
                 for model_module in model])), flush=True)                     # trace_info : t_9132, t_9135

    # GPU allocation.
    for model_module in model:                                                 # trace_info : t_9137, t_9139
        model_module.cuda(torch.cuda.current_device())                         # trace_info : t_9138

    # Fp16 conversion.
    if args.fp16 or args.bf16:                                                 # trace_info : t_9140
        model = [Float16Module(model_module, args) for model_module in model]  # trace_info : t_9141

    if wrap_with_ddp:                                                          # trace_info : t_9150
        config = get_model_config(model[0])                                    # trace_info : t_9151
        ddp_config = DistributedDataParallelConfig(                            # trace_info : t_9164, t_9170
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,       # trace_info : t_9165
            overlap_grad_reduce=args.overlap_grad_reduce,                      # trace_info : t_9166
            use_distributed_optimizer=args.use_distributed_optimizer,          # trace_info : t_9167
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,         # trace_info : t_9168
            bucket_size=args.ddp_bucket_size)                                  # trace_info : t_9169
        model = [DDP(config,                                                   # trace_info : t_9176, t_9178
                     ddp_config,
                     model_chunk,
                     data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                     expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]       # trace_info : t_9177

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:                                     # trace_info : t_11414
            for model_module in model:
                model_module.broadcast_params()

    return model                                                               # trace_info : t_11415


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()                                                          # trace_info : t_12816

    # Iteration-based training.
    if args.train_iters:                                                       # trace_info : t_12820
        if args.lr_decay_iters is None:                                        # trace_info : t_12821
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size          # trace_info : t_12822
        wd_incr_steps = args.train_iters * args.global_batch_size              # trace_info : t_12823
        if args.lr_warmup_fraction is not None:                                # trace_info : t_12824
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps         # trace_info : t_12825
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

    opt_param_scheduler = OptimizerParamScheduler(                             # trace_info : t_12826, t_12840
        optimizer,                                                             # trace_info : t_12827
        init_lr=args.lr_warmup_init,                                           # trace_info : t_12828
        max_lr=args.lr,                                                        # trace_info : t_12829
        min_lr=args.min_lr,                                                    # trace_info : t_12830
        lr_warmup_steps=lr_warmup_steps,                                       # trace_info : t_12831
        lr_decay_steps=lr_decay_steps,                                         # trace_info : t_12832
        lr_decay_style=args.lr_decay_style,                                    # trace_info : t_12833
        start_wd=args.start_weight_decay,                                      # trace_info : t_12834
        end_wd=args.end_weight_decay,                                          # trace_info : t_12835
        wd_incr_steps=wd_incr_steps,                                           # trace_info : t_12836
        wd_incr_style=args.weight_decay_incr_style,                            # trace_info : t_12837
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,# trace_info : t_12838
        override_opt_param_scheduler=args.override_opt_param_scheduler)        # trace_info : t_12839

    return opt_param_scheduler                                                 # trace_info : t_12906


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()                                                          # trace_info : t_5781
    timers = get_timers()                                                      # trace_info : t_5785

    model = get_model(model_provider_func, model_type)                         # trace_info : t_5789
    unwrapped_model = unwrap_model(model)                                      # trace_info : t_11416

    kwargs = {}                                                                # trace_info : t_11430
    for f in dataclasses.fields(OptimizerConfig):                              # trace_info : t_11431, t_11434, t_11437, t_11440, t_11443, ...
        if hasattr(args, f.name):                                              # trace_info : t_11432, t_11435, t_11438, t_11441, t_11444, ...
            kwargs[f.name] = getattr(args, f.name)                             # trace_info : t_11433, t_11436, t_11439, t_11442, t_11445, ...
    config = OptimizerConfig(**kwargs)                                         # trace_info : t_11506
    config.timers = timers                                                     # trace_info : t_11532
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,        # trace_info : t_11533, t_11535
                                       scale_lr_cond, lr_mult)                 # trace_info : t_11534
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)             # trace_info : t_12815

    if args.load is not None or args.pretrained_checkpoint is not None:        # trace_info : t_12907
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0                                                     # trace_info : t_12908
        args.num_floating_point_operations_so_far = 0                          # trace_info : t_12909

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \                     # trace_info : t_12910, t_12912
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):          # trace_info : t_12911
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler                               # trace_info : t_12913



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()                                                          # trace_info : t_14557, t_18142, t_21781
    timers = get_timers()                                                      # trace_info : t_14561, t_18146, t_21785

    # Set grad to zero.
    for model_chunk in model:                                                  # trace_info : t_14565, t_14663, t_18150, t_18248, t_21789, ...
        model_chunk.zero_grad_buffer()                                         # trace_info : t_14566, t_18151, t_21790
    optimizer.zero_grad()                                                      # trace_info : t_14664, t_18249, t_21888

    # Forward pass.
    forward_backward_func = get_forward_backward_func()                        # trace_info : t_14798, t_18439, t_22078
    losses_reduced = forward_backward_func(                                    # trace_info : t_14807, t_14818, t_18448, t_18459, t_22087, ...
        forward_step_func=forward_step_func,                                   # trace_info : t_14808, t_18449, t_22088
        data_iterator=data_iterator,                                           # trace_info : t_14809, t_18450, t_22089
        model=model,                                                           # trace_info : t_14810, t_18451, t_22090
        num_microbatches=get_num_microbatches(),                               # trace_info : t_14811, t_18452, t_22091
        seq_length=args.seq_length,                                            # trace_info : t_14814, t_18455, t_22094
        micro_batch_size=args.micro_batch_size,                                # trace_info : t_14815, t_18456, t_22095
        decoder_seq_length=args.decoder_seq_length,                            # trace_info : t_14816, t_18457, t_22096
        forward_only=False)                                                    # trace_info : t_14817, t_18458, t_22097

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:                                    # trace_info : t_16719, t_20358, t_23997
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_16720, t_20359, t_23998
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)  # trace_info : t_16721, t_20360, t_23999
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()         # trace_info : t_16728, t_20367, t_24006
    timers('optimizer').stop()                                                 # trace_info : t_17881, t_21520, t_25159

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_17889, t_21528, t_25167
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:                                                      # trace_info : t_17890, t_21529, t_25168
        increment = get_num_microbatches() * \                                 # trace_info : t_17891, t_17895, t_17897, t_21530, t_21534, ...
                    args.micro_batch_size * \                                  # trace_info : t_17894, t_21533, t_25172
                    args.data_parallel_size                                    # trace_info : t_17896, t_21535, t_25174
        opt_param_scheduler.step(increment=increment)                          # trace_info : t_17898, t_21537, t_25176
        skipped_iter = 0                                                       # trace_info : t_17937, t_21576, t_25215
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:                                    # trace_info : t_17938, t_21577, t_25216
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):                        # trace_info : t_17939, t_21578, t_25217
        # Average loss across microbatches.
        loss_reduced = {}                                                      # trace_info : t_17950, t_21589, t_25228
        for key in losses_reduced[0].keys():                                   # trace_info : t_17951, t_17961, t_21590, t_21600, t_25229, ...
            numerator = 0                                                      # trace_info : t_17952, t_21591, t_25230
            denominator = 0                                                    # trace_info : t_17953, t_21592, t_25231
            for x in losses_reduced:                                           # trace_info : t_17954, t_17959, t_21593, t_21598, t_25232, ...
                val = x[key]                                                   # trace_info : t_17955, t_21594, t_25233
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):            # trace_info : t_17956, t_21595, t_25234
                    numerator += val[0]                                        # trace_info : t_17957, t_21596, t_25235
                    denominator += val[1]                                      # trace_info : t_17958, t_21597, t_25236
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator                        # trace_info : t_17960, t_21599, t_25238
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad        # trace_info : t_17962, t_21601, t_25240
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()                                                          # trace_info : t_18041, t_21680, t_25319
    timers = get_timers()                                                      # trace_info : t_18045, t_21684, t_25323
    writer = get_tensorboard_writer()                                          # trace_info : t_18049, t_21688, t_25327
    wandb_writer = get_wandb_writer()                                          # trace_info : t_18051, t_21690, t_25329
    one_logger = get_one_logger()                                              # trace_info : t_18053, t_21692, t_25331

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'                                 # trace_info : t_18055, t_21694, t_25333
    skipped_iters_key = 'skipped iterations'                                   # trace_info : t_18056, t_21695, t_25334
    nan_iters_key = 'nan iterations'                                           # trace_info : t_18057, t_21696, t_25335
    # Advanced iterations.
    if not skipped_iter:                                                       # trace_info : t_18058, t_21697, t_25336
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(             # trace_info : t_18059, t_18061, t_18063, t_21698, t_21700, ...
            advanced_iters_key, 0) + 1                                         # trace_info : t_18060, t_18062, t_21699, t_21701, t_25338, ...
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(                  # trace_info : t_18064, t_18066, t_18068, t_21703, t_21705, ...
        skipped_iters_key, 0) + skipped_iter                                   # trace_info : t_18065, t_18067, t_21704, t_21706, t_25343, ...
    # Update losses and set nan iterations
    got_nan = False                                                            # trace_info : t_18069, t_21708, t_25347
    for key in loss_dict:                                                      # trace_info : t_18070, t_18077, t_21709, t_21716, t_25348, ...
        if not skipped_iter:                                                   # trace_info : t_18071, t_21710, t_25349
            total_loss_dict[key] = total_loss_dict.get(                        # trace_info : t_18072, t_18074, t_18076, t_21711, t_21713, ...
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]# trace_info : t_18073, t_18075, t_21712, t_21714, t_25351, ...
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(                      # trace_info : t_18078, t_18080, t_18082, t_21717, t_21719, ...
        nan_iters_key, 0) + int(got_nan)                                       # trace_info : t_18079, t_18081, t_21718, t_21720, t_25357, ...

    # Logging.
    timers_to_log = [                                                          # trace_info : t_18083, t_21722, t_25361
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
    batch_size = args.micro_batch_size * args.data_parallel_size * \           # trace_info : t_18084, t_18088, t_21723, t_21727, t_25362, ...
        get_num_microbatches()                                                 # trace_info : t_18085, t_21724, t_25363

    # Track app tag & app tag ID
    if one_logger:                                                             # trace_info : t_18089, t_21728, t_25367
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \                 # trace_info : t_18090, t_18092, t_21729, t_21731, t_25368, ...
                       total_loss_dict[skipped_iters_key]                      # trace_info : t_18091, t_21730, t_25369

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \                                    # trace_info : t_18093, t_21732, t_25371
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):            # trace_info : t_18094, t_21733, t_25372
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
    if args.num_experts is not None:                                           # trace_info : t_18095, t_21734, t_25373
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:                                     # trace_info : t_18096, t_21735, t_25374
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

    return report_memory_flag                                                  # trace_info : t_18097, t_21736, t_25375


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
    args = get_args()                                                          # trace_info : t_14455
    timers = get_timers()                                                      # trace_info : t_14459

    # Write args to tensorboard
    write_args_to_tensorboard()                                                # trace_info : t_14463

    # Turn on training mode which enables dropout.
    for model_module in model:                                                 # trace_info : t_14471, t_14473
        model_module.train()                                                   # trace_info : t_14472

    # Tracking loss.
    total_loss_dict = {}                                                       # trace_info : t_14474

    # Iterations.
    iteration = args.iteration                                                 # trace_info : t_14475
    one_logger = get_one_logger()                                              # trace_info : t_14476
    if one_logger:                                                             # trace_info : t_14478
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics({
            'train_samples_start': args.consumed_train_samples,
            'train_iterations_start': iteration,
            'train_samples_target': train_samples_target,
            'train_iterations_target': args.train_iters,
        })

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far# trace_info : t_14479

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss                              # trace_info : t_14480
    config.timers = timers                                                     # trace_info : t_14481
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:                 # trace_info : t_14482
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
    if args.overlap_param_gather and args.delay_param_gather:                  # trace_info : t_14483
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads                    # trace_info : t_14484

    timers('interval-time', log_level=0).start(barrier=True)                   # trace_info : t_14485
    print_datetime('before the start of training step')                        # trace_info : t_14506
    report_memory_flag = True                                                  # trace_info : t_14513
    exit = False                                                               # trace_info : t_14514

    if args.manual_gc:                                                         # trace_info : t_14515
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:                                                     # trace_info : t_14516
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    total_flops = 0.0                                                          # trace_info : t_14517

    num_microbatches = get_num_microbatches()                                  # trace_info : t_14518
    eval_duration = 0.0                                                        # trace_info : t_14521
    eval_iterations = 0                                                        # trace_info : t_14522
    def track_e2e_metrics():                                                   # trace_info : t_14523
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:                                                         # trace_info : t_25389
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

    while iteration < args.train_iters:                                        # trace_info : t_14524, t_18109, t_21748, t_25387
        if args.profile and \                                                  # trace_info : t_14525, t_18110, t_21749
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        maybe_finalize_async_save(False)                                       # trace_info : t_14526, t_18111, t_21750

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)# trace_info : t_14533, t_18118, t_21757
        if get_num_microbatches() != num_microbatches and iteration != 0:      # trace_info : t_14538, t_18123, t_21762
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
        num_microbatches = get_num_microbatches()                              # trace_info : t_14541, t_18126, t_21765
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)# trace_info : t_14544, t_18129, t_21768

        args.curr_iteration = iteration                                        # trace_info : t_14549, t_18134, t_21773
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \              # trace_info : t_17963, t_21602, t_25241
            train_step(forward_step_func,                                      # trace_info : t_14550, t_14556, t_18135, t_18141, t_21774, ...
                       train_data_iterator,                                    # trace_info : t_14551, t_18136, t_21775
                       model,                                                  # trace_info : t_14552, t_18137, t_21776
                       optimizer,                                              # trace_info : t_14553, t_18138, t_21777
                       opt_param_scheduler,                                    # trace_info : t_14554, t_18139, t_21778
                       config)                                                 # trace_info : t_14555, t_18140, t_21779
        iteration += 1                                                         # trace_info : t_17964, t_21603, t_25242
        batch_size = mpu.get_data_parallel_world_size() * \                    # trace_info : t_17965, t_17974, t_17978, t_21604, t_21613, ...
                     args.micro_batch_size * \                                 # trace_info : t_17973, t_21612, t_25251
                     get_num_microbatches()                                    # trace_info : t_17975, t_21614, t_25253
        args.consumed_train_samples += batch_size                              # trace_info : t_17979, t_21618, t_25257
        num_fp_ops = num_floating_point_operations(args, batch_size)           # trace_info : t_17980, t_21619, t_25258
        num_floating_point_operations_so_far += num_fp_ops                     # trace_info : t_18015, t_21654, t_25293
        total_flops += num_fp_ops                                              # trace_info : t_18016, t_21655, t_25294

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()                         # trace_info : t_18017, t_21656, t_25295
        params_norm = None                                                     # trace_info : t_18021, t_21660, t_25299
        if args.log_params_norm:                                               # trace_info : t_18022, t_21661, t_25300
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:                                 # trace_info : t_18023, t_21662, t_25301
            track_e2e_metrics()

        learning_rate = None                                                   # trace_info : t_18024, t_21663, t_25302
        decoupled_learning_rate = None                                         # trace_info : t_18025, t_21664, t_25303
        for param_group in optimizer.param_groups:                             # trace_info : t_18026, t_18030, t_18033, t_21665, t_21669, ...
            if param_group['is_decoupled_lr']:                                 # trace_info : t_18028, t_18031, t_21667, t_21670, t_25306, ...
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']                              # trace_info : t_18029, t_18032, t_21668, t_21671, t_25307, ...
        report_memory_flag = training_log(loss_dict, total_loss_dict,          # trace_info : t_18034, t_18040, t_21673, t_21679, t_25312, ...
                                          learning_rate,                       # trace_info : t_18035, t_21674, t_25313
                                          decoupled_learning_rate,             # trace_info : t_18036, t_21675, t_25314
                                          iteration, loss_scale,               # trace_info : t_18037, t_21676, t_25315
                                          report_memory_flag, skipped_iter,    # trace_info : t_18038, t_21677, t_25316
                                          grad_norm, params_norm, num_zeros_in_grad)# trace_info : t_18039, t_21678, t_25317
        # StragglerDetector
        if iteration % args.log_interval == 0 and args.log_straggler:          # trace_info : t_18098, t_21737, t_25376
            stimer.report(total_flops, args.log_interval)
            total_flops = 0.0

        if args.check_weight_hash_across_dp_replicas_interval is not None and \# trace_info : t_18099, t_21738, t_25377
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
        if args.adlr_autoresume and \                                          # trace_info : t_18100, t_21739, t_25378
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \    # trace_info : t_18101, t_21740, t_25379
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
        saved_checkpoint = False                                               # trace_info : t_18102, t_21741, t_25380
        if args.exit_signal_handler:                                           # trace_info : t_18103, t_21742, t_25381
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \                              # trace_info : t_18104, t_21743, t_25382
           iteration % args.save_interval == 0:
            timers('interval-time').stop()
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
            saved_checkpoint = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Exiting based on duration
        if args.exit_duration_in_mins:                                         # trace_info : t_18105, t_21744, t_25383
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
        if args.exit_interval and iteration % args.exit_interval == 0:         # trace_info : t_18106, t_21745, t_25384
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.profile and \                                                  # trace_info : t_18107, t_21746, t_25385
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()

        if args.manual_gc:                                                     # trace_info : t_18108, t_21747, t_25386
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    track_e2e_metrics()                                                        # trace_info : t_25388

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()                                          # trace_info : t_25390
    if writer:                                                                 # trace_info : t_25392
        writer.flush()
    wandb_writer = get_wandb_writer()                                          # trace_info : t_25393
    if wandb_writer:                                                           # trace_info : t_25395
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:           # trace_info : t_25396
        optimizer.disable_pre_hook()

    maybe_finalize_async_save(True)                                            # trace_info : t_25397

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:                                                                   # trace_info : t_25404
        sys.exit()

    return iteration, num_floating_point_operations_so_far                     # trace_info : t_25405


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

    args = get_args()                                                          # trace_info : t_12992

    # Number of train/valid/test samples.
    if args.train_samples:                                                     # trace_info : t_12996
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size              # trace_info : t_12997
    eval_iters = (args.train_iters // args.eval_interval + 1) * \              # trace_info : t_12998, t_13000
                 args.eval_iters                                               # trace_info : t_12999
    test_iters = args.eval_iters                                               # trace_info : t_13001

    return (                                                                   # trace_info : t_13005
        train_samples,                                                         # trace_info : t_13002
        eval_iters * args.global_batch_size,                                   # trace_info : t_13003
        test_iters * args.global_batch_size,                                   # trace_info : t_13004
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()          # trace_info : t_12991
    print_rank_0(' > datasets target sizes (minimum size):')                   # trace_info : t_13006
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0])) # trace_info : t_13010
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1])) # trace_info : t_13014
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2])) # trace_info : t_13018
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)# trace_info : t_13022


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()                                                          # trace_info : t_12975

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None) # trace_info : t_12979

    print_rank_0('> building train, validation, and test datasets ...')        # trace_info : t_12980

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:                # trace_info : t_12984
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:                # trace_info : t_12985
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)# trace_info : t_12986

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:            # trace_info : t_12987

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(         # trace_info : t_12988, t_12990
            build_train_valid_test_datasets_provider)                          # trace_info : t_12989
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(                      # trace_info : t_14037, t_14039
            train_ds, args.consumed_train_samples)                             # trace_info : t_14038
        if args.skip_train:                                                    # trace_info : t_14086
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(                  # trace_info : t_14087, t_14089
                valid_ds, args.consumed_valid_samples)                         # trace_info : t_14088
        test_dataloader = build_pretraining_data_loader(test_ds, 0)            # trace_info : t_14136

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0       # trace_info : t_14183
        do_valid = valid_dataloader is not None and args.eval_iters > 0        # trace_info : t_14184
        do_test = test_dataloader is not None and args.eval_iters > 0          # trace_info : t_14185
        flags = torch.tensor(                                                  # trace_info : t_14186, t_14189
            [int(do_train), int(do_valid), int(do_test)],                      # trace_info : t_14187
            dtype=torch.long, device='cuda')                                   # trace_info : t_14188
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)                                      # trace_info : t_14190

    args.do_train = getattr(args, "do_train", False) or flags[0].item()        # trace_info : t_14191
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()        # trace_info : t_14192
    args.do_test = getattr(args, "do_test", False) or flags[2].item()          # trace_info : t_14193

    return train_dataloader, valid_dataloader, test_dataloader                 # trace_info : t_14194


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()                                                          # trace_info : t_12968

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \                    # trace_info : t_14195
        build_train_valid_test_data_loaders(                                   # trace_info : t_12972, t_12974
            build_train_valid_test_datasets_provider)                          # trace_info : t_12973

    # Build iterators.
    dl_type = args.dataloader_type                                             # trace_info : t_14196
    assert dl_type in ['single', 'cyclic', 'external']                         # trace_info : t_14197

    def _get_iterator(dataloader_type, dataloader):                            # trace_info : t_14198
        """Return dataset iterator."""
        if dataloader_type == "single":                                        # trace_info : t_14201, t_14243, t_14285
            return iter(dataloader)                                            # trace_info : t_14202, t_14244, t_14286
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:                                           # trace_info : t_14199
        train_data_iterator = _get_iterator(dl_type, train_dataloader)         # trace_info : t_14200
    else:
        train_data_iterator = None

    if valid_dataloader is not None:                                           # trace_info : t_14241
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)         # trace_info : t_14242
    else:
        valid_data_iterator = None

    if test_dataloader is not None:                                            # trace_info : t_14283
        test_data_iterator = _get_iterator(dl_type, test_dataloader)           # trace_info : t_14284
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator        # trace_info : t_14325
