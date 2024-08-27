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
    torch.distributed.barrier()                                                # trace_info : t_9050, t_15785, t_17198, t_17365, t_26904
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                    # trace_info : t_9051, t_15786, t_17199, t_17366, t_26905
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))            # trace_info : t_9052, t_15787, t_17200, t_17367, t_26906


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads        # trace_info : t_20384, t_23570, t_26756
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size# trace_info : t_20385, t_23571, t_26757
    # Group Query Attention.
    if not args.group_query_attention:                                         # trace_info : t_20386, t_23572, t_26758
        args.num_query_groups = args.num_attention_heads                       # trace_info : t_20387, t_23573, t_26759
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk# trace_info : t_20388, t_23574, t_26760
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1                      # trace_info : t_20389, t_23575, t_26761
    return (                                                                   # trace_info : t_20417, t_23603, t_26789
        12                                                                     # trace_info : t_20390, t_20392, t_20394, t_20396, t_20398, ...
        * batch_size                                                           # trace_info : t_20391, t_23577, t_26763
        * args.seq_length                                                      # trace_info : t_20393, t_23579, t_26765
        * args.num_layers                                                      # trace_info : t_20395, t_23581, t_26767
        * args.hidden_size                                                     # trace_info : t_20397, t_23583, t_26769
        * args.hidden_size                                                     # trace_info : t_20399, t_23585, t_26771
        * (
            # Attention.
            (                                                                  # trace_info : t_20413, t_20415, t_23599, t_23601, t_26785, ...
                (                                                              # trace_info : t_20407, t_23593, t_26779
                    1                                                          # trace_info : t_20401, t_20403, t_20405, t_23587, t_23589, ...
                    + (args.num_query_groups / args.num_attention_heads)       # trace_info : t_20402, t_23588, t_26774
                    + (args.seq_length / args.hidden_size)                     # trace_info : t_20404, t_23590, t_26776
                ) * query_projection_to_hidden_size_ratio                      # trace_info : t_20406, t_23592, t_26778
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)                      # trace_info : t_20408, t_20410, t_20412, t_23594, t_23596, ...
                * num_experts_routed_to                                        # trace_info : t_20409, t_23595, t_26781
                * gated_linear_multiplier                                      # trace_info : t_20411, t_23597, t_26783
            )
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))# trace_info : t_20414, t_23600, t_26786
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

    args = get_args()                                                          # trace_info : t_8895
    timers = get_timers()                                                      # trace_info : t_8899

    if args.log_progress:                                                      # trace_info : t_8903
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()                                                   # trace_info : t_8904

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],                      # trace_info : t_9035, t_9038
                                     dtype=torch.double,                       # trace_info : t_9036
                                     device='cuda')                            # trace_info : t_9037
    torch.distributed.all_reduce(start_time_tensor,                            # trace_info : t_9039, t_9041
                                 op=torch.distributed.ReduceOp.MIN)            # trace_info : t_9040
    _TRAIN_START_TIME = start_time_tensor.item()                               # trace_info : t_9042
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(       # trace_info : t_9043, t_9045
        time.time() - _TRAIN_START_TIME))                                      # trace_info : t_9044
    print_datetime('after megatron is initialized')                            # trace_info : t_9049

    args = get_args()                                                          # trace_info : t_9056
    timers = get_timers()                                                      # trace_info : t_9060

    one_logger = get_one_logger()                                              # trace_info : t_9064
    if one_logger:                                                             # trace_info : t_9066
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)       # trace_info : t_9067
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(         # trace_info : t_9088, t_9090
        model_provider, model_type)                                            # trace_info : t_9089

    timers('model-and-optimizer-setup').stop()                                 # trace_info : t_15773
    print_datetime('after model, optimizer, and learning rate '                # trace_info : t_15784
                   'scheduler are built')
    config = get_model_config(model[0])                                        # trace_info : t_15791

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(        # trace_info : t_15800, t_15816
        barrier=True)                                                          # trace_info : t_15815
    if args.virtual_pipeline_model_parallel_size is not None:                  # trace_info : t_15823
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
        train_data_iterator, valid_data_iterator, test_data_iterator \         # trace_info : t_17185
            = build_train_valid_test_data_iterators(                           # trace_info : t_15824, t_15826
                train_valid_test_dataset_provider)                             # trace_info : t_15825
    timers('train/valid/test-data-iterators-setup').stop()                     # trace_info : t_17186
    print_datetime('after dataloaders are built')                              # trace_info : t_17197

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = {}                                                 # trace_info : t_17204

    # Print setup timing.
    print_rank_0('done with setup ...')                                        # trace_info : t_17205
    timers.log(['model-and-optimizer-setup',                                   # trace_info : t_17209, t_17211, t_17213
                'train/valid/test-data-iterators-setup'], barrier=True)        # trace_info : t_17210, t_17212

    if not args.skip_train:                                                    # trace_info : t_17299
        print_rank_0('training ...')                                           # trace_info : t_17300

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:        # trace_info : t_17304
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0                                                          # trace_info : t_17305
        if args.do_train and args.train_iters > 0:                             # trace_info : t_17306
            iteration, num_floating_point_operations_so_far = train(           # trace_info : t_17307, t_17312
                forward_step_func,                                             # trace_info : t_17308
                model, optimizer, opt_param_scheduler,                         # trace_info : t_17309
                train_data_iterator, valid_data_iterator,                      # trace_info : t_17310
                process_non_loss_data_func, config, checkpointing_context)     # trace_info : t_17311

        print_datetime('after training is done')                               # trace_info : t_26903

        if args.save and iteration != 0 and iteration % args.save_interval != 0:# trace_info : t_26910
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:                                                          # trace_info : t_26911
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:                                                           # trace_info : t_26912
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    maybe_finalize_async_save(blocking=True)                                   # trace_info : t_26913



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
    args = get_args()                                                          # trace_info : t_9100
    args.model_type = model_type                                               # trace_info : t_9104

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \                  # trace_info : t_9105
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
        pre_process = mpu.is_pipeline_first_stage()                            # trace_info : t_9110
        post_process = mpu.is_pipeline_last_stage()                            # trace_info : t_9119
        add_encoder = True                                                     # trace_info : t_9134
        add_decoder = True                                                     # trace_info : t_9135
        if model_type == ModelType.encoder_and_decoder:                        # trace_info : t_9136
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
            model = model_provider_func(                                       # trace_info : t_9137, t_9140
                pre_process=pre_process,                                       # trace_info : t_9138
                post_process=post_process                                      # trace_info : t_9139
            )
        model.model_type = model_type                                          # trace_info : t_11554

    if not isinstance(model, list):                                            # trace_info : t_11555
        model = [model]                                                        # trace_info : t_11556

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:                                                 # trace_info : t_11557, t_11968
        for param in model_module.parameters():                                # trace_info : t_11558, t_11571, t_11587, t_11600, t_11616, ...
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)# trace_info : t_11559, t_11572, t_11588, t_11601, t_11617, ...

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:                                      # trace_info : t_11969
        print(' > number of parameters on (tensor, pipeline) '                 # trace_info : t_11977, t_11995
              'model parallel rank ({}, {}): {}'.format(                       # trace_info : t_11978, t_11993
            mpu.get_tensor_model_parallel_rank(),                              # trace_info : t_11979
            mpu.get_pipeline_model_parallel_rank(),                            # trace_info : t_11985
            sum([sum([p.nelement() for p in model_module.parameters()])        # trace_info : t_11990, t_11992
                 for model_module in model])), flush=True)                     # trace_info : t_11991, t_11994

    # GPU allocation.
    for model_module in model:                                                 # trace_info : t_11996, t_11998
        model_module.cuda(torch.cuda.current_device())                         # trace_info : t_11997

    # Fp16 conversion.
    if args.fp16 or args.bf16:                                                 # trace_info : t_11999
        model = [Float16Module(model_module, args) for model_module in model]  # trace_info : t_12000

    if wrap_with_ddp:                                                          # trace_info : t_12009
        config = get_model_config(model[0])                                    # trace_info : t_12010
        ddp_config = DistributedDataParallelConfig(                            # trace_info : t_12023, t_12029
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,       # trace_info : t_12024
            overlap_grad_reduce=args.overlap_grad_reduce,                      # trace_info : t_12025
            use_distributed_optimizer=args.use_distributed_optimizer,          # trace_info : t_12026
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,         # trace_info : t_12027
            bucket_size=args.ddp_bucket_size)                                  # trace_info : t_12028
        model = [DDP(config,                                                   # trace_info : t_12035, t_12037
                     ddp_config,
                     model_chunk,
                     data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                     expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]       # trace_info : t_12036

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:                                     # trace_info : t_14273
            for model_module in model:
                model_module.broadcast_params()

    return model                                                               # trace_info : t_14274


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()                                                          # trace_info : t_15675

    # Iteration-based training.
    if args.train_iters:                                                       # trace_info : t_15679
        if args.lr_decay_iters is None:                                        # trace_info : t_15680
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size          # trace_info : t_15681
        wd_incr_steps = args.train_iters * args.global_batch_size              # trace_info : t_15682
        if args.lr_warmup_fraction is not None:                                # trace_info : t_15683
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps         # trace_info : t_15684
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

    opt_param_scheduler = OptimizerParamScheduler(                             # trace_info : t_15685, t_15699
        optimizer,                                                             # trace_info : t_15686
        init_lr=args.lr_warmup_init,                                           # trace_info : t_15687
        max_lr=args.lr,                                                        # trace_info : t_15688
        min_lr=args.min_lr,                                                    # trace_info : t_15689
        lr_warmup_steps=lr_warmup_steps,                                       # trace_info : t_15690
        lr_decay_steps=lr_decay_steps,                                         # trace_info : t_15691
        lr_decay_style=args.lr_decay_style,                                    # trace_info : t_15692
        start_wd=args.start_weight_decay,                                      # trace_info : t_15693
        end_wd=args.end_weight_decay,                                          # trace_info : t_15694
        wd_incr_steps=wd_incr_steps,                                           # trace_info : t_15695
        wd_incr_style=args.weight_decay_incr_style,                            # trace_info : t_15696
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,# trace_info : t_15697
        override_opt_param_scheduler=args.override_opt_param_scheduler)        # trace_info : t_15698

    return opt_param_scheduler                                                 # trace_info : t_15765


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()                                                          # trace_info : t_9091
    timers = get_timers()                                                      # trace_info : t_9095

    model = get_model(model_provider_func, model_type)                         # trace_info : t_9099
    unwrapped_model = unwrap_model(model)                                      # trace_info : t_14275

    kwargs = {}                                                                # trace_info : t_14289
    for f in dataclasses.fields(OptimizerConfig):                              # trace_info : t_14290, t_14293, t_14296, t_14299, t_14302, ...
        if hasattr(args, f.name):                                              # trace_info : t_14291, t_14294, t_14297, t_14300, t_14303, ...
            kwargs[f.name] = getattr(args, f.name)                             # trace_info : t_14292, t_14295, t_14298, t_14301, t_14304, ...
    config = OptimizerConfig(**kwargs)                                         # trace_info : t_14365
    config.timers = timers                                                     # trace_info : t_14391
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,        # trace_info : t_14392, t_14394
                                       scale_lr_cond, lr_mult)                 # trace_info : t_14393
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)             # trace_info : t_15674

    if args.load is not None or args.pretrained_checkpoint is not None:        # trace_info : t_15766
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0                                                     # trace_info : t_15767
        args.num_floating_point_operations_so_far = 0                          # trace_info : t_15768

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \                     # trace_info : t_15769, t_15771
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):          # trace_info : t_15770
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler                               # trace_info : t_15772



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()                                                          # trace_info : t_17415, t_20545, t_23731
    timers = get_timers()                                                      # trace_info : t_17419, t_20549, t_23735

    # Set grad to zero.
    for model_chunk in model:                                                  # trace_info : t_17423, t_17521, t_20553, t_20651, t_23739, ...
        model_chunk.zero_grad_buffer()                                         # trace_info : t_17424, t_20554, t_23740
    optimizer.zero_grad()                                                      # trace_info : t_17522, t_20652, t_23838

    # Forward pass.
    forward_backward_func = get_forward_backward_func()                        # trace_info : t_17656, t_20842, t_24028
    losses_reduced = forward_backward_func(                                    # trace_info : t_17665, t_17676, t_20851, t_20862, t_24037, ...
        forward_step_func=forward_step_func,                                   # trace_info : t_17666, t_20852, t_24038
        data_iterator=data_iterator,                                           # trace_info : t_17667, t_20853, t_24039
        model=model,                                                           # trace_info : t_17668, t_20854, t_24040
        num_microbatches=get_num_microbatches(),                               # trace_info : t_17669, t_20855, t_24041
        seq_length=args.seq_length,                                            # trace_info : t_17672, t_20858, t_24044
        micro_batch_size=args.micro_batch_size,                                # trace_info : t_17673, t_20859, t_24045
        decoder_seq_length=args.decoder_seq_length,                            # trace_info : t_17674, t_20860, t_24046
        forward_only=False)                                                    # trace_info : t_17675, t_20861, t_24047

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:                                    # trace_info : t_19122, t_22308, t_25494
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_19123, t_22309, t_25495
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)  # trace_info : t_19124, t_22310, t_25496
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()         # trace_info : t_19131, t_22317, t_25503
    timers('optimizer').stop()                                                 # trace_info : t_20284, t_23470, t_26656

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_20292, t_23478, t_26664
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:                                                      # trace_info : t_20293, t_23479, t_26665
        increment = get_num_microbatches() * \                                 # trace_info : t_20294, t_20298, t_20300, t_23480, t_23484, ...
                    args.micro_batch_size * \                                  # trace_info : t_20297, t_23483, t_26669
                    args.data_parallel_size                                    # trace_info : t_20299, t_23485, t_26671
        opt_param_scheduler.step(increment=increment)                          # trace_info : t_20301, t_23487, t_26673
        skipped_iter = 0                                                       # trace_info : t_20340, t_23526, t_26712
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:                                    # trace_info : t_20341, t_23527, t_26713
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):                        # trace_info : t_20342, t_23528, t_26714
        # Average loss across microbatches.
        loss_reduced = {}                                                      # trace_info : t_20353, t_23539, t_26725
        for key in losses_reduced[0].keys():                                   # trace_info : t_20354, t_20364, t_23540, t_23550, t_26726, ...
            numerator = 0                                                      # trace_info : t_20355, t_23541, t_26727
            denominator = 0                                                    # trace_info : t_20356, t_23542, t_26728
            for x in losses_reduced:                                           # trace_info : t_20357, t_20362, t_23543, t_23548, t_26729, ...
                val = x[key]                                                   # trace_info : t_20358, t_23544, t_26730
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):            # trace_info : t_20359, t_23545, t_26731
                    numerator += val[0]                                        # trace_info : t_20360, t_23546, t_26732
                    denominator += val[1]                                      # trace_info : t_20361, t_23547, t_26733
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator                        # trace_info : t_20363, t_23549, t_26735
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad        # trace_info : t_20365, t_23551, t_26737
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()                                                          # trace_info : t_20444, t_23630, t_26816
    timers = get_timers()                                                      # trace_info : t_20448, t_23634, t_26820
    writer = get_tensorboard_writer()                                          # trace_info : t_20452, t_23638, t_26824
    wandb_writer = get_wandb_writer()                                          # trace_info : t_20454, t_23640, t_26826
    one_logger = get_one_logger()                                              # trace_info : t_20456, t_23642, t_26828

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'                                 # trace_info : t_20458, t_23644, t_26830
    skipped_iters_key = 'skipped iterations'                                   # trace_info : t_20459, t_23645, t_26831
    nan_iters_key = 'nan iterations'                                           # trace_info : t_20460, t_23646, t_26832
    # Advanced iterations.
    if not skipped_iter:                                                       # trace_info : t_20461, t_23647, t_26833
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(             # trace_info : t_20462, t_20464, t_20466, t_23648, t_23650, ...
            advanced_iters_key, 0) + 1                                         # trace_info : t_20463, t_20465, t_23649, t_23651, t_26835, ...
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(                  # trace_info : t_20467, t_20469, t_20471, t_23653, t_23655, ...
        skipped_iters_key, 0) + skipped_iter                                   # trace_info : t_20468, t_20470, t_23654, t_23656, t_26840, ...
    # Update losses and set nan iterations
    got_nan = False                                                            # trace_info : t_20472, t_23658, t_26844
    for key in loss_dict:                                                      # trace_info : t_20473, t_20480, t_23659, t_23666, t_26845, ...
        if not skipped_iter:                                                   # trace_info : t_20474, t_23660, t_26846
            total_loss_dict[key] = total_loss_dict.get(                        # trace_info : t_20475, t_20477, t_20479, t_23661, t_23663, ...
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]# trace_info : t_20476, t_20478, t_23662, t_23664, t_26848, ...
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(                      # trace_info : t_20481, t_20483, t_20485, t_23667, t_23669, ...
        nan_iters_key, 0) + int(got_nan)                                       # trace_info : t_20482, t_20484, t_23668, t_23670, t_26854, ...

    # Logging.
    timers_to_log = [                                                          # trace_info : t_20486, t_23672, t_26858
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
    batch_size = args.micro_batch_size * args.data_parallel_size * \           # trace_info : t_20487, t_20491, t_23673, t_23677, t_26859, ...
        get_num_microbatches()                                                 # trace_info : t_20488, t_23674, t_26860

    # Track app tag & app tag ID
    if one_logger:                                                             # trace_info : t_20492, t_23678, t_26864
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \                 # trace_info : t_20493, t_20495, t_23679, t_23681, t_26865, ...
                       total_loss_dict[skipped_iters_key]                      # trace_info : t_20494, t_23680, t_26866

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \                                    # trace_info : t_20496, t_23682, t_26868
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):            # trace_info : t_20497, t_23683, t_26869
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
    if args.num_experts is not None:                                           # trace_info : t_20498, t_23684, t_26870
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:                                     # trace_info : t_20499, t_23685, t_26871
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

    return report_memory_flag                                                  # trace_info : t_20500, t_23686, t_26872


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
    args = get_args()                                                          # trace_info : t_17313
    timers = get_timers()                                                      # trace_info : t_17317

    # Write args to tensorboard
    write_args_to_tensorboard()                                                # trace_info : t_17321

    # Turn on training mode which enables dropout.
    for model_module in model:                                                 # trace_info : t_17329, t_17331
        model_module.train()                                                   # trace_info : t_17330

    # Tracking loss.
    total_loss_dict = {}                                                       # trace_info : t_17332

    # Iterations.
    iteration = args.iteration                                                 # trace_info : t_17333
    one_logger = get_one_logger()                                              # trace_info : t_17334
    if one_logger:                                                             # trace_info : t_17336
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics({
            'train_samples_start': args.consumed_train_samples,
            'train_iterations_start': iteration,
            'train_samples_target': train_samples_target,
            'train_iterations_target': args.train_iters,
        })

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far# trace_info : t_17337

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss                              # trace_info : t_17338
    config.timers = timers                                                     # trace_info : t_17339
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:                 # trace_info : t_17340
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
    if args.overlap_param_gather and args.delay_param_gather:                  # trace_info : t_17341
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads                    # trace_info : t_17342

    timers('interval-time', log_level=0).start(barrier=True)                   # trace_info : t_17343
    print_datetime('before the start of training step')                        # trace_info : t_17364
    report_memory_flag = True                                                  # trace_info : t_17371
    exit = False                                                               # trace_info : t_17372

    if args.manual_gc:                                                         # trace_info : t_17373
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:                                                     # trace_info : t_17374
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    total_flops = 0.0                                                          # trace_info : t_17375

    num_microbatches = get_num_microbatches()                                  # trace_info : t_17376
    eval_duration = 0.0                                                        # trace_info : t_17379
    eval_iterations = 0                                                        # trace_info : t_17380
    def track_e2e_metrics():                                                   # trace_info : t_17381
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:                                                         # trace_info : t_26886
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

    while iteration < args.train_iters:                                        # trace_info : t_17382, t_20512, t_23698, t_26884
        if args.profile and \                                                  # trace_info : t_17383, t_20513, t_23699
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        maybe_finalize_async_save(False)                                       # trace_info : t_17384, t_20514, t_23700

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)# trace_info : t_17391, t_20521, t_23707
        if get_num_microbatches() != num_microbatches and iteration != 0:      # trace_info : t_17396, t_20526, t_23712
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
        num_microbatches = get_num_microbatches()                              # trace_info : t_17399, t_20529, t_23715
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)# trace_info : t_17402, t_20532, t_23718

        args.curr_iteration = iteration                                        # trace_info : t_17407, t_20537, t_23723
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \              # trace_info : t_20366, t_23552, t_26738
            train_step(forward_step_func,                                      # trace_info : t_17408, t_17414, t_20538, t_20544, t_23724, ...
                       train_data_iterator,                                    # trace_info : t_17409, t_20539, t_23725
                       model,                                                  # trace_info : t_17410, t_20540, t_23726
                       optimizer,                                              # trace_info : t_17411, t_20541, t_23727
                       opt_param_scheduler,                                    # trace_info : t_17412, t_20542, t_23728
                       config)                                                 # trace_info : t_17413, t_20543, t_23729
        iteration += 1                                                         # trace_info : t_20367, t_23553, t_26739
        batch_size = mpu.get_data_parallel_world_size() * \                    # trace_info : t_20368, t_20377, t_20381, t_23554, t_23563, ...
                     args.micro_batch_size * \                                 # trace_info : t_20376, t_23562, t_26748
                     get_num_microbatches()                                    # trace_info : t_20378, t_23564, t_26750
        args.consumed_train_samples += batch_size                              # trace_info : t_20382, t_23568, t_26754
        num_fp_ops = num_floating_point_operations(args, batch_size)           # trace_info : t_20383, t_23569, t_26755
        num_floating_point_operations_so_far += num_fp_ops                     # trace_info : t_20418, t_23604, t_26790
        total_flops += num_fp_ops                                              # trace_info : t_20419, t_23605, t_26791

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()                         # trace_info : t_20420, t_23606, t_26792
        params_norm = None                                                     # trace_info : t_20424, t_23610, t_26796
        if args.log_params_norm:                                               # trace_info : t_20425, t_23611, t_26797
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:                                 # trace_info : t_20426, t_23612, t_26798
            track_e2e_metrics()

        learning_rate = None                                                   # trace_info : t_20427, t_23613, t_26799
        decoupled_learning_rate = None                                         # trace_info : t_20428, t_23614, t_26800
        for param_group in optimizer.param_groups:                             # trace_info : t_20429, t_20433, t_20436, t_23615, t_23619, ...
            if param_group['is_decoupled_lr']:                                 # trace_info : t_20431, t_20434, t_23617, t_23620, t_26803, ...
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']                              # trace_info : t_20432, t_20435, t_23618, t_23621, t_26804, ...
        report_memory_flag = training_log(loss_dict, total_loss_dict,          # trace_info : t_20437, t_20443, t_23623, t_23629, t_26809, ...
                                          learning_rate,                       # trace_info : t_20438, t_23624, t_26810
                                          decoupled_learning_rate,             # trace_info : t_20439, t_23625, t_26811
                                          iteration, loss_scale,               # trace_info : t_20440, t_23626, t_26812
                                          report_memory_flag, skipped_iter,    # trace_info : t_20441, t_23627, t_26813
                                          grad_norm, params_norm, num_zeros_in_grad)# trace_info : t_20442, t_23628, t_26814
        # StragglerDetector
        if iteration % args.log_interval == 0 and args.log_straggler:          # trace_info : t_20501, t_23687, t_26873
            stimer.report(total_flops, args.log_interval)
            total_flops = 0.0

        if args.check_weight_hash_across_dp_replicas_interval is not None and \# trace_info : t_20502, t_23688, t_26874
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
        if args.adlr_autoresume and \                                          # trace_info : t_20503, t_23689, t_26875
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \    # trace_info : t_20504, t_23690, t_26876
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
        saved_checkpoint = False                                               # trace_info : t_20505, t_23691, t_26877
        if args.exit_signal_handler:                                           # trace_info : t_20506, t_23692, t_26878
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \                              # trace_info : t_20507, t_23693, t_26879
           iteration % args.save_interval == 0:
            timers('interval-time').stop()
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
            saved_checkpoint = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Exiting based on duration
        if args.exit_duration_in_mins:                                         # trace_info : t_20508, t_23694, t_26880
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
        if args.exit_interval and iteration % args.exit_interval == 0:         # trace_info : t_20509, t_23695, t_26881
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.profile and \                                                  # trace_info : t_20510, t_23696, t_26882
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()

        if args.manual_gc:                                                     # trace_info : t_20511, t_23697, t_26883
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    track_e2e_metrics()                                                        # trace_info : t_26885

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()                                          # trace_info : t_26887
    if writer:                                                                 # trace_info : t_26889
        writer.flush()
    wandb_writer = get_wandb_writer()                                          # trace_info : t_26890
    if wandb_writer:                                                           # trace_info : t_26892
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:           # trace_info : t_26893
        optimizer.disable_pre_hook()

    maybe_finalize_async_save(True)                                            # trace_info : t_26894

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:                                                                   # trace_info : t_26901
        sys.exit()

    return iteration, num_floating_point_operations_so_far                     # trace_info : t_26902


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

    args = get_args()                                                          # trace_info : t_15851

    # Number of train/valid/test samples.
    if args.train_samples:                                                     # trace_info : t_15855
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size              # trace_info : t_15856
    eval_iters = (args.train_iters // args.eval_interval + 1) * \              # trace_info : t_15857, t_15859
                 args.eval_iters                                               # trace_info : t_15858
    test_iters = args.eval_iters                                               # trace_info : t_15860

    return (                                                                   # trace_info : t_15864
        train_samples,                                                         # trace_info : t_15861
        eval_iters * args.global_batch_size,                                   # trace_info : t_15862
        test_iters * args.global_batch_size,                                   # trace_info : t_15863
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()          # trace_info : t_15850
    print_rank_0(' > datasets target sizes (minimum size):')                   # trace_info : t_15865
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0])) # trace_info : t_15869
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1])) # trace_info : t_15873
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2])) # trace_info : t_15877
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)# trace_info : t_15881


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()                                                          # trace_info : t_15834

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None) # trace_info : t_15838

    print_rank_0('> building train, validation, and test datasets ...')        # trace_info : t_15839

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:                # trace_info : t_15843
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:                # trace_info : t_15844
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)# trace_info : t_15845

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:            # trace_info : t_15846

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(         # trace_info : t_15847, t_15849
            build_train_valid_test_datasets_provider)                          # trace_info : t_15848
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(                      # trace_info : t_16896, t_16898
            train_ds, args.consumed_train_samples)                             # trace_info : t_16897
        if args.skip_train:                                                    # trace_info : t_16945
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(                  # trace_info : t_16946, t_16948
                valid_ds, args.consumed_valid_samples)                         # trace_info : t_16947
        test_dataloader = build_pretraining_data_loader(test_ds, 0)            # trace_info : t_16995

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0       # trace_info : t_17042
        do_valid = valid_dataloader is not None and args.eval_iters > 0        # trace_info : t_17043
        do_test = test_dataloader is not None and args.eval_iters > 0          # trace_info : t_17044
        flags = torch.tensor(                                                  # trace_info : t_17045, t_17048
            [int(do_train), int(do_valid), int(do_test)],                      # trace_info : t_17046
            dtype=torch.long, device='cuda')                                   # trace_info : t_17047
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)                                      # trace_info : t_17049

    args.do_train = getattr(args, "do_train", False) or flags[0].item()        # trace_info : t_17050
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()        # trace_info : t_17051
    args.do_test = getattr(args, "do_test", False) or flags[2].item()          # trace_info : t_17052

    return train_dataloader, valid_dataloader, test_dataloader                 # trace_info : t_17053


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()                                                          # trace_info : t_15827

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \                    # trace_info : t_17054
        build_train_valid_test_data_loaders(                                   # trace_info : t_15831, t_15833
            build_train_valid_test_datasets_provider)                          # trace_info : t_15832

    # Build iterators.
    dl_type = args.dataloader_type                                             # trace_info : t_17055
    assert dl_type in ['single', 'cyclic', 'external']                         # trace_info : t_17056

    def _get_iterator(dataloader_type, dataloader):                            # trace_info : t_17057
        """Return dataset iterator."""
        if dataloader_type == "single":                                        # trace_info : t_17060, t_17102, t_17144
            return iter(dataloader)                                            # trace_info : t_17061, t_17103, t_17145
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:                                           # trace_info : t_17058
        train_data_iterator = _get_iterator(dl_type, train_dataloader)         # trace_info : t_17059
    else:
        train_data_iterator = None

    if valid_dataloader is not None:                                           # trace_info : t_17100
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)         # trace_info : t_17101
    else:
        valid_data_iterator = None

    if test_dataloader is not None:                                            # trace_info : t_17142
        test_data_iterator = _get_iterator(dl_type, test_dataloader)           # trace_info : t_17143
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator        # trace_info : t_17184
