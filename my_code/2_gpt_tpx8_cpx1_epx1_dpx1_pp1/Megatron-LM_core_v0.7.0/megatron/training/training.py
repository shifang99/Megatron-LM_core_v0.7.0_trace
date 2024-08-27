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
    torch.distributed.barrier()                                                # trace_info : t_10607, t_17793, t_19206, t_19373, t_30186
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                    # trace_info : t_10608, t_17794, t_19207, t_19374, t_30187
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))            # trace_info : t_10609, t_17795, t_19208, t_19375, t_30188


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads        # trace_info : t_22818, t_26428, t_30038
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size# trace_info : t_22819, t_26429, t_30039
    # Group Query Attention.
    if not args.group_query_attention:                                         # trace_info : t_22820, t_26430, t_30040
        args.num_query_groups = args.num_attention_heads                       # trace_info : t_22821, t_26431, t_30041
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk# trace_info : t_22822, t_26432, t_30042
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1                      # trace_info : t_22823, t_26433, t_30043
    return (                                                                   # trace_info : t_22851, t_26461, t_30071
        12                                                                     # trace_info : t_22824, t_22826, t_22828, t_22830, t_22832, ...
        * batch_size                                                           # trace_info : t_22825, t_26435, t_30045
        * args.seq_length                                                      # trace_info : t_22827, t_26437, t_30047
        * args.num_layers                                                      # trace_info : t_22829, t_26439, t_30049
        * args.hidden_size                                                     # trace_info : t_22831, t_26441, t_30051
        * args.hidden_size                                                     # trace_info : t_22833, t_26443, t_30053
        * (
            # Attention.
            (                                                                  # trace_info : t_22847, t_22849, t_26457, t_26459, t_30067, ...
                (                                                              # trace_info : t_22841, t_26451, t_30061
                    1                                                          # trace_info : t_22835, t_22837, t_22839, t_26445, t_26447, ...
                    + (args.num_query_groups / args.num_attention_heads)       # trace_info : t_22836, t_26446, t_30056
                    + (args.seq_length / args.hidden_size)                     # trace_info : t_22838, t_26448, t_30058
                ) * query_projection_to_hidden_size_ratio                      # trace_info : t_22840, t_26450, t_30060
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)                      # trace_info : t_22842, t_22844, t_22846, t_26452, t_26454, ...
                * num_experts_routed_to                                        # trace_info : t_22843, t_26453, t_30063
                * gated_linear_multiplier                                      # trace_info : t_22845, t_26455, t_30065
            )
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))# trace_info : t_22848, t_26458, t_30068
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

    args = get_args()                                                          # trace_info : t_10452
    timers = get_timers()                                                      # trace_info : t_10456

    if args.log_progress:                                                      # trace_info : t_10460
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()                                                   # trace_info : t_10461

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],                      # trace_info : t_10592, t_10595
                                     dtype=torch.double,                       # trace_info : t_10593
                                     device='cuda')                            # trace_info : t_10594
    torch.distributed.all_reduce(start_time_tensor,                            # trace_info : t_10596, t_10598
                                 op=torch.distributed.ReduceOp.MIN)            # trace_info : t_10597
    _TRAIN_START_TIME = start_time_tensor.item()                               # trace_info : t_10599
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(       # trace_info : t_10600, t_10602
        time.time() - _TRAIN_START_TIME))                                      # trace_info : t_10601
    print_datetime('after megatron is initialized')                            # trace_info : t_10606

    args = get_args()                                                          # trace_info : t_10613
    timers = get_timers()                                                      # trace_info : t_10617

    one_logger = get_one_logger()                                              # trace_info : t_10621
    if one_logger:                                                             # trace_info : t_10623
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)       # trace_info : t_10624
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(         # trace_info : t_10645, t_10647
        model_provider, model_type)                                            # trace_info : t_10646

    timers('model-and-optimizer-setup').stop()                                 # trace_info : t_17781
    print_datetime('after model, optimizer, and learning rate '                # trace_info : t_17792
                   'scheduler are built')
    config = get_model_config(model[0])                                        # trace_info : t_17799

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(        # trace_info : t_17808, t_17824
        barrier=True)                                                          # trace_info : t_17823
    if args.virtual_pipeline_model_parallel_size is not None:                  # trace_info : t_17831
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
        train_data_iterator, valid_data_iterator, test_data_iterator \         # trace_info : t_19193
            = build_train_valid_test_data_iterators(                           # trace_info : t_17832, t_17834
                train_valid_test_dataset_provider)                             # trace_info : t_17833
    timers('train/valid/test-data-iterators-setup').stop()                     # trace_info : t_19194
    print_datetime('after dataloaders are built')                              # trace_info : t_19205

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = {}                                                 # trace_info : t_19212

    # Print setup timing.
    print_rank_0('done with setup ...')                                        # trace_info : t_19213
    timers.log(['model-and-optimizer-setup',                                   # trace_info : t_19217, t_19219, t_19221
                'train/valid/test-data-iterators-setup'], barrier=True)        # trace_info : t_19218, t_19220

    if not args.skip_train:                                                    # trace_info : t_19307
        print_rank_0('training ...')                                           # trace_info : t_19308

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:        # trace_info : t_19312
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0                                                          # trace_info : t_19313
        if args.do_train and args.train_iters > 0:                             # trace_info : t_19314
            iteration, num_floating_point_operations_so_far = train(           # trace_info : t_19315, t_19320
                forward_step_func,                                             # trace_info : t_19316
                model, optimizer, opt_param_scheduler,                         # trace_info : t_19317
                train_data_iterator, valid_data_iterator,                      # trace_info : t_19318
                process_non_loss_data_func, config, checkpointing_context)     # trace_info : t_19319

        print_datetime('after training is done')                               # trace_info : t_30185

        if args.save and iteration != 0 and iteration % args.save_interval != 0:# trace_info : t_30192
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:                                                          # trace_info : t_30193
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:                                                           # trace_info : t_30194
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    maybe_finalize_async_save(blocking=True)                                   # trace_info : t_30195



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
    args = get_args()                                                          # trace_info : t_10657
    args.model_type = model_type                                               # trace_info : t_10661

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \                  # trace_info : t_10662
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
        pre_process = mpu.is_pipeline_first_stage()                            # trace_info : t_10667
        post_process = mpu.is_pipeline_last_stage()                            # trace_info : t_10676
        add_encoder = True                                                     # trace_info : t_10691
        add_decoder = True                                                     # trace_info : t_10692
        if model_type == ModelType.encoder_and_decoder:                        # trace_info : t_10693
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
            model = model_provider_func(                                       # trace_info : t_10694, t_10697
                pre_process=pre_process,                                       # trace_info : t_10695
                post_process=post_process                                      # trace_info : t_10696
            )
        model.model_type = model_type                                          # trace_info : t_13562

    if not isinstance(model, list):                                            # trace_info : t_13563
        model = [model]                                                        # trace_info : t_13564

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:                                                 # trace_info : t_13565, t_13976
        for param in model_module.parameters():                                # trace_info : t_13566, t_13579, t_13595, t_13611, t_13627, ...
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)# trace_info : t_13567, t_13580, t_13596, t_13612, t_13628, ...

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:                                      # trace_info : t_13977
        print(' > number of parameters on (tensor, pipeline) '                 # trace_info : t_13985, t_14003
              'model parallel rank ({}, {}): {}'.format(                       # trace_info : t_13986, t_14001
            mpu.get_tensor_model_parallel_rank(),                              # trace_info : t_13987
            mpu.get_pipeline_model_parallel_rank(),                            # trace_info : t_13993
            sum([sum([p.nelement() for p in model_module.parameters()])        # trace_info : t_13998, t_14000
                 for model_module in model])), flush=True)                     # trace_info : t_13999, t_14002

    # GPU allocation.
    for model_module in model:                                                 # trace_info : t_14004, t_14006
        model_module.cuda(torch.cuda.current_device())                         # trace_info : t_14005

    # Fp16 conversion.
    if args.fp16 or args.bf16:                                                 # trace_info : t_14007
        model = [Float16Module(model_module, args) for model_module in model]  # trace_info : t_14008

    if wrap_with_ddp:                                                          # trace_info : t_14017
        config = get_model_config(model[0])                                    # trace_info : t_14018
        ddp_config = DistributedDataParallelConfig(                            # trace_info : t_14031, t_14037
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,       # trace_info : t_14032
            overlap_grad_reduce=args.overlap_grad_reduce,                      # trace_info : t_14033
            use_distributed_optimizer=args.use_distributed_optimizer,          # trace_info : t_14034
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,         # trace_info : t_14035
            bucket_size=args.ddp_bucket_size)                                  # trace_info : t_14036
        model = [DDP(config,                                                   # trace_info : t_14043, t_14045
                     ddp_config,
                     model_chunk,
                     data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                     expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]       # trace_info : t_14044

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:                                     # trace_info : t_16281
            for model_module in model:
                model_module.broadcast_params()

    return model                                                               # trace_info : t_16282


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()                                                          # trace_info : t_17683

    # Iteration-based training.
    if args.train_iters:                                                       # trace_info : t_17687
        if args.lr_decay_iters is None:                                        # trace_info : t_17688
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size          # trace_info : t_17689
        wd_incr_steps = args.train_iters * args.global_batch_size              # trace_info : t_17690
        if args.lr_warmup_fraction is not None:                                # trace_info : t_17691
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps         # trace_info : t_17692
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

    opt_param_scheduler = OptimizerParamScheduler(                             # trace_info : t_17693, t_17707
        optimizer,                                                             # trace_info : t_17694
        init_lr=args.lr_warmup_init,                                           # trace_info : t_17695
        max_lr=args.lr,                                                        # trace_info : t_17696
        min_lr=args.min_lr,                                                    # trace_info : t_17697
        lr_warmup_steps=lr_warmup_steps,                                       # trace_info : t_17698
        lr_decay_steps=lr_decay_steps,                                         # trace_info : t_17699
        lr_decay_style=args.lr_decay_style,                                    # trace_info : t_17700
        start_wd=args.start_weight_decay,                                      # trace_info : t_17701
        end_wd=args.end_weight_decay,                                          # trace_info : t_17702
        wd_incr_steps=wd_incr_steps,                                           # trace_info : t_17703
        wd_incr_style=args.weight_decay_incr_style,                            # trace_info : t_17704
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,# trace_info : t_17705
        override_opt_param_scheduler=args.override_opt_param_scheduler)        # trace_info : t_17706

    return opt_param_scheduler                                                 # trace_info : t_17773


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()                                                          # trace_info : t_10648
    timers = get_timers()                                                      # trace_info : t_10652

    model = get_model(model_provider_func, model_type)                         # trace_info : t_10656
    unwrapped_model = unwrap_model(model)                                      # trace_info : t_16283

    kwargs = {}                                                                # trace_info : t_16297
    for f in dataclasses.fields(OptimizerConfig):                              # trace_info : t_16298, t_16301, t_16304, t_16307, t_16310, ...
        if hasattr(args, f.name):                                              # trace_info : t_16299, t_16302, t_16305, t_16308, t_16311, ...
            kwargs[f.name] = getattr(args, f.name)                             # trace_info : t_16300, t_16303, t_16306, t_16309, t_16312, ...
    config = OptimizerConfig(**kwargs)                                         # trace_info : t_16373
    config.timers = timers                                                     # trace_info : t_16399
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,        # trace_info : t_16400, t_16402
                                       scale_lr_cond, lr_mult)                 # trace_info : t_16401
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)             # trace_info : t_17682

    if args.load is not None or args.pretrained_checkpoint is not None:        # trace_info : t_17774
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0                                                     # trace_info : t_17775
        args.num_floating_point_operations_so_far = 0                          # trace_info : t_17776

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \                     # trace_info : t_17777, t_17779
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):          # trace_info : t_17778
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler                               # trace_info : t_17780



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()                                                          # trace_info : t_19423, t_22979, t_26589
    timers = get_timers()                                                      # trace_info : t_19427, t_22983, t_26593

    # Set grad to zero.
    for model_chunk in model:                                                  # trace_info : t_19431, t_19529, t_22987, t_23085, t_26597, ...
        model_chunk.zero_grad_buffer()                                         # trace_info : t_19432, t_22988, t_26598
    optimizer.zero_grad()                                                      # trace_info : t_19530, t_23086, t_26696

    # Forward pass.
    forward_backward_func = get_forward_backward_func()                        # trace_info : t_19664, t_23276, t_26886
    losses_reduced = forward_backward_func(                                    # trace_info : t_19673, t_19684, t_23285, t_23296, t_26895, ...
        forward_step_func=forward_step_func,                                   # trace_info : t_19674, t_23286, t_26896
        data_iterator=data_iterator,                                           # trace_info : t_19675, t_23287, t_26897
        model=model,                                                           # trace_info : t_19676, t_23288, t_26898
        num_microbatches=get_num_microbatches(),                               # trace_info : t_19677, t_23289, t_26899
        seq_length=args.seq_length,                                            # trace_info : t_19680, t_23292, t_26902
        micro_batch_size=args.micro_batch_size,                                # trace_info : t_19681, t_23293, t_26903
        decoder_seq_length=args.decoder_seq_length,                            # trace_info : t_19682, t_23294, t_26904
        forward_only=False)                                                    # trace_info : t_19683, t_23295, t_26905

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:                                    # trace_info : t_21556, t_25166, t_28776
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_21557, t_25167, t_28777
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)  # trace_info : t_21558, t_25168, t_28778
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()         # trace_info : t_21565, t_25175, t_28785
    timers('optimizer').stop()                                                 # trace_info : t_22718, t_26328, t_29938

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_22726, t_26336, t_29946
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:                                                      # trace_info : t_22727, t_26337, t_29947
        increment = get_num_microbatches() * \                                 # trace_info : t_22728, t_22732, t_22734, t_26338, t_26342, ...
                    args.micro_batch_size * \                                  # trace_info : t_22731, t_26341, t_29951
                    args.data_parallel_size                                    # trace_info : t_22733, t_26343, t_29953
        opt_param_scheduler.step(increment=increment)                          # trace_info : t_22735, t_26345, t_29955
        skipped_iter = 0                                                       # trace_info : t_22774, t_26384, t_29994
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:                                    # trace_info : t_22775, t_26385, t_29995
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):                        # trace_info : t_22776, t_26386, t_29996
        # Average loss across microbatches.
        loss_reduced = {}                                                      # trace_info : t_22787, t_26397, t_30007
        for key in losses_reduced[0].keys():                                   # trace_info : t_22788, t_22798, t_26398, t_26408, t_30008, ...
            numerator = 0                                                      # trace_info : t_22789, t_26399, t_30009
            denominator = 0                                                    # trace_info : t_22790, t_26400, t_30010
            for x in losses_reduced:                                           # trace_info : t_22791, t_22796, t_26401, t_26406, t_30011, ...
                val = x[key]                                                   # trace_info : t_22792, t_26402, t_30012
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):            # trace_info : t_22793, t_26403, t_30013
                    numerator += val[0]                                        # trace_info : t_22794, t_26404, t_30014
                    denominator += val[1]                                      # trace_info : t_22795, t_26405, t_30015
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator                        # trace_info : t_22797, t_26407, t_30017
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad        # trace_info : t_22799, t_26409, t_30019
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()                                                          # trace_info : t_22878, t_26488, t_30098
    timers = get_timers()                                                      # trace_info : t_22882, t_26492, t_30102
    writer = get_tensorboard_writer()                                          # trace_info : t_22886, t_26496, t_30106
    wandb_writer = get_wandb_writer()                                          # trace_info : t_22888, t_26498, t_30108
    one_logger = get_one_logger()                                              # trace_info : t_22890, t_26500, t_30110

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'                                 # trace_info : t_22892, t_26502, t_30112
    skipped_iters_key = 'skipped iterations'                                   # trace_info : t_22893, t_26503, t_30113
    nan_iters_key = 'nan iterations'                                           # trace_info : t_22894, t_26504, t_30114
    # Advanced iterations.
    if not skipped_iter:                                                       # trace_info : t_22895, t_26505, t_30115
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(             # trace_info : t_22896, t_22898, t_22900, t_26506, t_26508, ...
            advanced_iters_key, 0) + 1                                         # trace_info : t_22897, t_22899, t_26507, t_26509, t_30117, ...
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(                  # trace_info : t_22901, t_22903, t_22905, t_26511, t_26513, ...
        skipped_iters_key, 0) + skipped_iter                                   # trace_info : t_22902, t_22904, t_26512, t_26514, t_30122, ...
    # Update losses and set nan iterations
    got_nan = False                                                            # trace_info : t_22906, t_26516, t_30126
    for key in loss_dict:                                                      # trace_info : t_22907, t_22914, t_26517, t_26524, t_30127, ...
        if not skipped_iter:                                                   # trace_info : t_22908, t_26518, t_30128
            total_loss_dict[key] = total_loss_dict.get(                        # trace_info : t_22909, t_22911, t_22913, t_26519, t_26521, ...
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]# trace_info : t_22910, t_22912, t_26520, t_26522, t_30130, ...
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(                      # trace_info : t_22915, t_22917, t_22919, t_26525, t_26527, ...
        nan_iters_key, 0) + int(got_nan)                                       # trace_info : t_22916, t_22918, t_26526, t_26528, t_30136, ...

    # Logging.
    timers_to_log = [                                                          # trace_info : t_22920, t_26530, t_30140
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
    batch_size = args.micro_batch_size * args.data_parallel_size * \           # trace_info : t_22921, t_22925, t_26531, t_26535, t_30141, ...
        get_num_microbatches()                                                 # trace_info : t_22922, t_26532, t_30142

    # Track app tag & app tag ID
    if one_logger:                                                             # trace_info : t_22926, t_26536, t_30146
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \                 # trace_info : t_22927, t_22929, t_26537, t_26539, t_30147, ...
                       total_loss_dict[skipped_iters_key]                      # trace_info : t_22928, t_26538, t_30148

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \                                    # trace_info : t_22930, t_26540, t_30150
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):            # trace_info : t_22931, t_26541, t_30151
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
    if args.num_experts is not None:                                           # trace_info : t_22932, t_26542, t_30152
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:                                     # trace_info : t_22933, t_26543, t_30153
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

    return report_memory_flag                                                  # trace_info : t_22934, t_26544, t_30154


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
    args = get_args()                                                          # trace_info : t_19321
    timers = get_timers()                                                      # trace_info : t_19325

    # Write args to tensorboard
    write_args_to_tensorboard()                                                # trace_info : t_19329

    # Turn on training mode which enables dropout.
    for model_module in model:                                                 # trace_info : t_19337, t_19339
        model_module.train()                                                   # trace_info : t_19338

    # Tracking loss.
    total_loss_dict = {}                                                       # trace_info : t_19340

    # Iterations.
    iteration = args.iteration                                                 # trace_info : t_19341
    one_logger = get_one_logger()                                              # trace_info : t_19342
    if one_logger:                                                             # trace_info : t_19344
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics({
            'train_samples_start': args.consumed_train_samples,
            'train_iterations_start': iteration,
            'train_samples_target': train_samples_target,
            'train_iterations_target': args.train_iters,
        })

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far# trace_info : t_19345

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss                              # trace_info : t_19346
    config.timers = timers                                                     # trace_info : t_19347
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:                 # trace_info : t_19348
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
    if args.overlap_param_gather and args.delay_param_gather:                  # trace_info : t_19349
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads                    # trace_info : t_19350

    timers('interval-time', log_level=0).start(barrier=True)                   # trace_info : t_19351
    print_datetime('before the start of training step')                        # trace_info : t_19372
    report_memory_flag = True                                                  # trace_info : t_19379
    exit = False                                                               # trace_info : t_19380

    if args.manual_gc:                                                         # trace_info : t_19381
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:                                                     # trace_info : t_19382
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    total_flops = 0.0                                                          # trace_info : t_19383

    num_microbatches = get_num_microbatches()                                  # trace_info : t_19384
    eval_duration = 0.0                                                        # trace_info : t_19387
    eval_iterations = 0                                                        # trace_info : t_19388
    def track_e2e_metrics():                                                   # trace_info : t_19389
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:                                                         # trace_info : t_30168
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

    while iteration < args.train_iters:                                        # trace_info : t_19390, t_22946, t_26556, t_30166
        if args.profile and \                                                  # trace_info : t_19391, t_22947, t_26557
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        maybe_finalize_async_save(False)                                       # trace_info : t_19392, t_22948, t_26558

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)# trace_info : t_19399, t_22955, t_26565
        if get_num_microbatches() != num_microbatches and iteration != 0:      # trace_info : t_19404, t_22960, t_26570
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
        num_microbatches = get_num_microbatches()                              # trace_info : t_19407, t_22963, t_26573
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)# trace_info : t_19410, t_22966, t_26576

        args.curr_iteration = iteration                                        # trace_info : t_19415, t_22971, t_26581
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \              # trace_info : t_22800, t_26410, t_30020
            train_step(forward_step_func,                                      # trace_info : t_19416, t_19422, t_22972, t_22978, t_26582, ...
                       train_data_iterator,                                    # trace_info : t_19417, t_22973, t_26583
                       model,                                                  # trace_info : t_19418, t_22974, t_26584
                       optimizer,                                              # trace_info : t_19419, t_22975, t_26585
                       opt_param_scheduler,                                    # trace_info : t_19420, t_22976, t_26586
                       config)                                                 # trace_info : t_19421, t_22977, t_26587
        iteration += 1                                                         # trace_info : t_22801, t_26411, t_30021
        batch_size = mpu.get_data_parallel_world_size() * \                    # trace_info : t_22802, t_22811, t_22815, t_26412, t_26421, ...
                     args.micro_batch_size * \                                 # trace_info : t_22810, t_26420, t_30030
                     get_num_microbatches()                                    # trace_info : t_22812, t_26422, t_30032
        args.consumed_train_samples += batch_size                              # trace_info : t_22816, t_26426, t_30036
        num_fp_ops = num_floating_point_operations(args, batch_size)           # trace_info : t_22817, t_26427, t_30037
        num_floating_point_operations_so_far += num_fp_ops                     # trace_info : t_22852, t_26462, t_30072
        total_flops += num_fp_ops                                              # trace_info : t_22853, t_26463, t_30073

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()                         # trace_info : t_22854, t_26464, t_30074
        params_norm = None                                                     # trace_info : t_22858, t_26468, t_30078
        if args.log_params_norm:                                               # trace_info : t_22859, t_26469, t_30079
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:                                 # trace_info : t_22860, t_26470, t_30080
            track_e2e_metrics()

        learning_rate = None                                                   # trace_info : t_22861, t_26471, t_30081
        decoupled_learning_rate = None                                         # trace_info : t_22862, t_26472, t_30082
        for param_group in optimizer.param_groups:                             # trace_info : t_22863, t_22867, t_22870, t_26473, t_26477, ...
            if param_group['is_decoupled_lr']:                                 # trace_info : t_22865, t_22868, t_26475, t_26478, t_30085, ...
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']                              # trace_info : t_22866, t_22869, t_26476, t_26479, t_30086, ...
        report_memory_flag = training_log(loss_dict, total_loss_dict,          # trace_info : t_22871, t_22877, t_26481, t_26487, t_30091, ...
                                          learning_rate,                       # trace_info : t_22872, t_26482, t_30092
                                          decoupled_learning_rate,             # trace_info : t_22873, t_26483, t_30093
                                          iteration, loss_scale,               # trace_info : t_22874, t_26484, t_30094
                                          report_memory_flag, skipped_iter,    # trace_info : t_22875, t_26485, t_30095
                                          grad_norm, params_norm, num_zeros_in_grad)# trace_info : t_22876, t_26486, t_30096
        # StragglerDetector
        if iteration % args.log_interval == 0 and args.log_straggler:          # trace_info : t_22935, t_26545, t_30155
            stimer.report(total_flops, args.log_interval)
            total_flops = 0.0

        if args.check_weight_hash_across_dp_replicas_interval is not None and \# trace_info : t_22936, t_26546, t_30156
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
        if args.adlr_autoresume and \                                          # trace_info : t_22937, t_26547, t_30157
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \    # trace_info : t_22938, t_26548, t_30158
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
        saved_checkpoint = False                                               # trace_info : t_22939, t_26549, t_30159
        if args.exit_signal_handler:                                           # trace_info : t_22940, t_26550, t_30160
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \                              # trace_info : t_22941, t_26551, t_30161
           iteration % args.save_interval == 0:
            timers('interval-time').stop()
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
            saved_checkpoint = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Exiting based on duration
        if args.exit_duration_in_mins:                                         # trace_info : t_22942, t_26552, t_30162
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
        if args.exit_interval and iteration % args.exit_interval == 0:         # trace_info : t_22943, t_26553, t_30163
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.profile and \                                                  # trace_info : t_22944, t_26554, t_30164
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()

        if args.manual_gc:                                                     # trace_info : t_22945, t_26555, t_30165
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    track_e2e_metrics()                                                        # trace_info : t_30167

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()                                          # trace_info : t_30169
    if writer:                                                                 # trace_info : t_30171
        writer.flush()
    wandb_writer = get_wandb_writer()                                          # trace_info : t_30172
    if wandb_writer:                                                           # trace_info : t_30174
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:           # trace_info : t_30175
        optimizer.disable_pre_hook()

    maybe_finalize_async_save(True)                                            # trace_info : t_30176

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:                                                                   # trace_info : t_30183
        sys.exit()

    return iteration, num_floating_point_operations_so_far                     # trace_info : t_30184


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

    args = get_args()                                                          # trace_info : t_17859

    # Number of train/valid/test samples.
    if args.train_samples:                                                     # trace_info : t_17863
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size              # trace_info : t_17864
    eval_iters = (args.train_iters // args.eval_interval + 1) * \              # trace_info : t_17865, t_17867
                 args.eval_iters                                               # trace_info : t_17866
    test_iters = args.eval_iters                                               # trace_info : t_17868

    return (                                                                   # trace_info : t_17872
        train_samples,                                                         # trace_info : t_17869
        eval_iters * args.global_batch_size,                                   # trace_info : t_17870
        test_iters * args.global_batch_size,                                   # trace_info : t_17871
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()          # trace_info : t_17858
    print_rank_0(' > datasets target sizes (minimum size):')                   # trace_info : t_17873
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0])) # trace_info : t_17877
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1])) # trace_info : t_17881
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2])) # trace_info : t_17885
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)# trace_info : t_17889


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()                                                          # trace_info : t_17842

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None) # trace_info : t_17846

    print_rank_0('> building train, validation, and test datasets ...')        # trace_info : t_17847

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:                # trace_info : t_17851
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:                # trace_info : t_17852
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)# trace_info : t_17853

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:            # trace_info : t_17854

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(         # trace_info : t_17855, t_17857
            build_train_valid_test_datasets_provider)                          # trace_info : t_17856
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(                      # trace_info : t_18904, t_18906
            train_ds, args.consumed_train_samples)                             # trace_info : t_18905
        if args.skip_train:                                                    # trace_info : t_18953
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(                  # trace_info : t_18954, t_18956
                valid_ds, args.consumed_valid_samples)                         # trace_info : t_18955
        test_dataloader = build_pretraining_data_loader(test_ds, 0)            # trace_info : t_19003

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0       # trace_info : t_19050
        do_valid = valid_dataloader is not None and args.eval_iters > 0        # trace_info : t_19051
        do_test = test_dataloader is not None and args.eval_iters > 0          # trace_info : t_19052
        flags = torch.tensor(                                                  # trace_info : t_19053, t_19056
            [int(do_train), int(do_valid), int(do_test)],                      # trace_info : t_19054
            dtype=torch.long, device='cuda')                                   # trace_info : t_19055
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)                                      # trace_info : t_19057

    args.do_train = getattr(args, "do_train", False) or flags[0].item()        # trace_info : t_19058
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()        # trace_info : t_19059
    args.do_test = getattr(args, "do_test", False) or flags[2].item()          # trace_info : t_19060

    return train_dataloader, valid_dataloader, test_dataloader                 # trace_info : t_19061


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()                                                          # trace_info : t_17835

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \                    # trace_info : t_19062
        build_train_valid_test_data_loaders(                                   # trace_info : t_17839, t_17841
            build_train_valid_test_datasets_provider)                          # trace_info : t_17840

    # Build iterators.
    dl_type = args.dataloader_type                                             # trace_info : t_19063
    assert dl_type in ['single', 'cyclic', 'external']                         # trace_info : t_19064

    def _get_iterator(dataloader_type, dataloader):                            # trace_info : t_19065
        """Return dataset iterator."""
        if dataloader_type == "single":                                        # trace_info : t_19068, t_19110, t_19152
            return iter(dataloader)                                            # trace_info : t_19069, t_19111, t_19153
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:                                           # trace_info : t_19066
        train_data_iterator = _get_iterator(dl_type, train_dataloader)         # trace_info : t_19067
    else:
        train_data_iterator = None

    if valid_dataloader is not None:                                           # trace_info : t_19108
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)         # trace_info : t_19109
    else:
        valid_data_iterator = None

    if test_dataloader is not None:                                            # trace_info : t_19150
        test_data_iterator = _get_iterator(dl_type, test_dataloader)           # trace_info : t_19151
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator        # trace_info : t_19192
