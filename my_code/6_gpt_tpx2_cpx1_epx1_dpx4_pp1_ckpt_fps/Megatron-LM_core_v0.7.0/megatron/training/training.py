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
    torch.distributed.barrier()                                                # trace_info : t_8769, t_15955, t_17476, t_17643, t_92507
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                    # trace_info : t_8770, t_15956, t_17477, t_17644, t_92508
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))            # trace_info : t_8771, t_15957, t_17478, t_17645, t_92509


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads        # trace_info : t_21114, t_24751, t_92358
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size# trace_info : t_21115, t_24752, t_92359
    # Group Query Attention.
    if not args.group_query_attention:                                         # trace_info : t_21116, t_24753, t_92360
        args.num_query_groups = args.num_attention_heads                       # trace_info : t_21117, t_24754, t_92361
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk# trace_info : t_21118, t_24755, t_92362
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1                      # trace_info : t_21119, t_24756, t_92363
    return (                                                                   # trace_info : t_21147, t_24784, t_92391
        12                                                                     # trace_info : t_21120, t_21122, t_21124, t_21126, t_21128, ...
        * batch_size                                                           # trace_info : t_21121, t_24758, t_92365
        * args.seq_length                                                      # trace_info : t_21123, t_24760, t_92367
        * args.num_layers                                                      # trace_info : t_21125, t_24762, t_92369
        * args.hidden_size                                                     # trace_info : t_21127, t_24764, t_92371
        * args.hidden_size                                                     # trace_info : t_21129, t_24766, t_92373
        * (
            # Attention.
            (                                                                  # trace_info : t_21143, t_21145, t_24780, t_24782, t_92387, ...
                (                                                              # trace_info : t_21137, t_24774, t_92381
                    1                                                          # trace_info : t_21131, t_21133, t_21135, t_24768, t_24770, ...
                    + (args.num_query_groups / args.num_attention_heads)       # trace_info : t_21132, t_24769, t_92376
                    + (args.seq_length / args.hidden_size)                     # trace_info : t_21134, t_24771, t_92378
                ) * query_projection_to_hidden_size_ratio                      # trace_info : t_21136, t_24773, t_92380
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)                      # trace_info : t_21138, t_21140, t_21142, t_24775, t_24777, ...
                * num_experts_routed_to                                        # trace_info : t_21139, t_24776, t_92383
                * gated_linear_multiplier                                      # trace_info : t_21141, t_24778, t_92385
            )
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))# trace_info : t_21144, t_24781, t_92388
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

    args = get_args()                                                          # trace_info : t_8614
    timers = get_timers()                                                      # trace_info : t_8618

    if args.log_progress:                                                      # trace_info : t_8622
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()                                                   # trace_info : t_8623

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],                      # trace_info : t_8754, t_8757
                                     dtype=torch.double,                       # trace_info : t_8755
                                     device='cuda')                            # trace_info : t_8756
    torch.distributed.all_reduce(start_time_tensor,                            # trace_info : t_8758, t_8760
                                 op=torch.distributed.ReduceOp.MIN)            # trace_info : t_8759
    _TRAIN_START_TIME = start_time_tensor.item()                               # trace_info : t_8761
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(       # trace_info : t_8762, t_8764
        time.time() - _TRAIN_START_TIME))                                      # trace_info : t_8763
    print_datetime('after megatron is initialized')                            # trace_info : t_8768

    args = get_args()                                                          # trace_info : t_8775
    timers = get_timers()                                                      # trace_info : t_8779

    one_logger = get_one_logger()                                              # trace_info : t_8783
    if one_logger:                                                             # trace_info : t_8785
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)       # trace_info : t_8786
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(         # trace_info : t_8807, t_8809
        model_provider, model_type)                                            # trace_info : t_8808

    timers('model-and-optimizer-setup').stop()                                 # trace_info : t_15943
    print_datetime('after model, optimizer, and learning rate '                # trace_info : t_15954
                   'scheduler are built')
    config = get_model_config(model[0])                                        # trace_info : t_15961

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(        # trace_info : t_15970, t_15986
        barrier=True)                                                          # trace_info : t_15985
    if args.virtual_pipeline_model_parallel_size is not None:                  # trace_info : t_15993
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
        train_data_iterator, valid_data_iterator, test_data_iterator \         # trace_info : t_17463
            = build_train_valid_test_data_iterators(                           # trace_info : t_15994, t_15996
                train_valid_test_dataset_provider)                             # trace_info : t_15995
    timers('train/valid/test-data-iterators-setup').stop()                     # trace_info : t_17464
    print_datetime('after dataloaders are built')                              # trace_info : t_17475

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = {}                                                 # trace_info : t_17482

    # Print setup timing.
    print_rank_0('done with setup ...')                                        # trace_info : t_17483
    timers.log(['model-and-optimizer-setup',                                   # trace_info : t_17487, t_17489, t_17491
                'train/valid/test-data-iterators-setup'], barrier=True)        # trace_info : t_17488, t_17490

    if not args.skip_train:                                                    # trace_info : t_17577
        print_rank_0('training ...')                                           # trace_info : t_17578

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:        # trace_info : t_17582
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0                                                          # trace_info : t_17583
        if args.do_train and args.train_iters > 0:                             # trace_info : t_17584
            iteration, num_floating_point_operations_so_far = train(           # trace_info : t_17585, t_17590
                forward_step_func,                                             # trace_info : t_17586
                model, optimizer, opt_param_scheduler,                         # trace_info : t_17587
                train_data_iterator, valid_data_iterator,                      # trace_info : t_17588
                process_non_loss_data_func, config, checkpointing_context)     # trace_info : t_17589

        print_datetime('after training is done')                               # trace_info : t_92506

        if args.save and iteration != 0 and iteration % args.save_interval != 0:# trace_info : t_92513
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,  # trace_info : t_92514, t_92516
                            num_floating_point_operations_so_far, checkpointing_context)# trace_info : t_92515
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:                                                          # trace_info : t_156128
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:                                                           # trace_info : t_156129
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    maybe_finalize_async_save(blocking=True)                                   # trace_info : t_156130



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
    args = get_args()                                                          # trace_info : t_8819
    args.model_type = model_type                                               # trace_info : t_8823

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \                  # trace_info : t_8824
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
        pre_process = mpu.is_pipeline_first_stage()                            # trace_info : t_8829
        post_process = mpu.is_pipeline_last_stage()                            # trace_info : t_8838
        add_encoder = True                                                     # trace_info : t_8853
        add_decoder = True                                                     # trace_info : t_8854
        if model_type == ModelType.encoder_and_decoder:                        # trace_info : t_8855
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
            model = model_provider_func(                                       # trace_info : t_8856, t_8859
                pre_process=pre_process,                                       # trace_info : t_8857
                post_process=post_process                                      # trace_info : t_8858
            )
        model.model_type = model_type                                          # trace_info : t_11724

    if not isinstance(model, list):                                            # trace_info : t_11725
        model = [model]                                                        # trace_info : t_11726

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:                                                 # trace_info : t_11727, t_12138
        for param in model_module.parameters():                                # trace_info : t_11728, t_11741, t_11757, t_11773, t_11789, ...
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)# trace_info : t_11729, t_11742, t_11758, t_11774, t_11790, ...

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:                                      # trace_info : t_12139
        print(' > number of parameters on (tensor, pipeline) '                 # trace_info : t_12147, t_12165
              'model parallel rank ({}, {}): {}'.format(                       # trace_info : t_12148, t_12163
            mpu.get_tensor_model_parallel_rank(),                              # trace_info : t_12149
            mpu.get_pipeline_model_parallel_rank(),                            # trace_info : t_12155
            sum([sum([p.nelement() for p in model_module.parameters()])        # trace_info : t_12160, t_12162
                 for model_module in model])), flush=True)                     # trace_info : t_12161, t_12164

    # GPU allocation.
    for model_module in model:                                                 # trace_info : t_12166, t_12168
        model_module.cuda(torch.cuda.current_device())                         # trace_info : t_12167

    # Fp16 conversion.
    if args.fp16 or args.bf16:                                                 # trace_info : t_12169
        model = [Float16Module(model_module, args) for model_module in model]  # trace_info : t_12170

    if wrap_with_ddp:                                                          # trace_info : t_12179
        config = get_model_config(model[0])                                    # trace_info : t_12180
        ddp_config = DistributedDataParallelConfig(                            # trace_info : t_12193, t_12199
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,       # trace_info : t_12194
            overlap_grad_reduce=args.overlap_grad_reduce,                      # trace_info : t_12195
            use_distributed_optimizer=args.use_distributed_optimizer,          # trace_info : t_12196
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,         # trace_info : t_12197
            bucket_size=args.ddp_bucket_size)                                  # trace_info : t_12198
        model = [DDP(config,                                                   # trace_info : t_12205, t_12207
                     ddp_config,
                     model_chunk,
                     data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                     expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]       # trace_info : t_12206

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:                                     # trace_info : t_14443
            for model_module in model:
                model_module.broadcast_params()

    return model                                                               # trace_info : t_14444


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()                                                          # trace_info : t_15845

    # Iteration-based training.
    if args.train_iters:                                                       # trace_info : t_15849
        if args.lr_decay_iters is None:                                        # trace_info : t_15850
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size          # trace_info : t_15851
        wd_incr_steps = args.train_iters * args.global_batch_size              # trace_info : t_15852
        if args.lr_warmup_fraction is not None:                                # trace_info : t_15853
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps         # trace_info : t_15854
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

    opt_param_scheduler = OptimizerParamScheduler(                             # trace_info : t_15855, t_15869
        optimizer,                                                             # trace_info : t_15856
        init_lr=args.lr_warmup_init,                                           # trace_info : t_15857
        max_lr=args.lr,                                                        # trace_info : t_15858
        min_lr=args.min_lr,                                                    # trace_info : t_15859
        lr_warmup_steps=lr_warmup_steps,                                       # trace_info : t_15860
        lr_decay_steps=lr_decay_steps,                                         # trace_info : t_15861
        lr_decay_style=args.lr_decay_style,                                    # trace_info : t_15862
        start_wd=args.start_weight_decay,                                      # trace_info : t_15863
        end_wd=args.end_weight_decay,                                          # trace_info : t_15864
        wd_incr_steps=wd_incr_steps,                                           # trace_info : t_15865
        wd_incr_style=args.weight_decay_incr_style,                            # trace_info : t_15866
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,# trace_info : t_15867
        override_opt_param_scheduler=args.override_opt_param_scheduler)        # trace_info : t_15868

    return opt_param_scheduler                                                 # trace_info : t_15935


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()                                                          # trace_info : t_8810
    timers = get_timers()                                                      # trace_info : t_8814

    model = get_model(model_provider_func, model_type)                         # trace_info : t_8818
    unwrapped_model = unwrap_model(model)                                      # trace_info : t_14445

    kwargs = {}                                                                # trace_info : t_14459
    for f in dataclasses.fields(OptimizerConfig):                              # trace_info : t_14460, t_14463, t_14466, t_14469, t_14472, ...
        if hasattr(args, f.name):                                              # trace_info : t_14461, t_14464, t_14467, t_14470, t_14473, ...
            kwargs[f.name] = getattr(args, f.name)                             # trace_info : t_14462, t_14465, t_14468, t_14471, t_14474, ...
    config = OptimizerConfig(**kwargs)                                         # trace_info : t_14535
    config.timers = timers                                                     # trace_info : t_14561
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,        # trace_info : t_14562, t_14564
                                       scale_lr_cond, lr_mult)                 # trace_info : t_14563
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)             # trace_info : t_15844

    if args.load is not None or args.pretrained_checkpoint is not None:        # trace_info : t_15936
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0                                                     # trace_info : t_15937
        args.num_floating_point_operations_so_far = 0                          # trace_info : t_15938

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \                     # trace_info : t_15939, t_15941
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):          # trace_info : t_15940
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler                               # trace_info : t_15942



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()                                                          # trace_info : t_17693, t_21276, t_88883
    timers = get_timers()                                                      # trace_info : t_17697, t_21280, t_88887

    # Set grad to zero.
    for model_chunk in model:                                                  # trace_info : t_17701, t_17799, t_21284, t_21382, t_88891, ...
        model_chunk.zero_grad_buffer()                                         # trace_info : t_17702, t_21285, t_88892
    optimizer.zero_grad()                                                      # trace_info : t_17800, t_21383, t_88990

    # Forward pass.
    forward_backward_func = get_forward_backward_func()                        # trace_info : t_17934, t_21573, t_89180
    losses_reduced = forward_backward_func(                                    # trace_info : t_17943, t_17954, t_21582, t_21593, t_89189, ...
        forward_step_func=forward_step_func,                                   # trace_info : t_17944, t_21583, t_89190
        data_iterator=data_iterator,                                           # trace_info : t_17945, t_21584, t_89191
        model=model,                                                           # trace_info : t_17946, t_21585, t_89192
        num_microbatches=get_num_microbatches(),                               # trace_info : t_17947, t_21586, t_89193
        seq_length=args.seq_length,                                            # trace_info : t_17950, t_21589, t_89196
        micro_batch_size=args.micro_batch_size,                                # trace_info : t_17951, t_21590, t_89197
        decoder_seq_length=args.decoder_seq_length,                            # trace_info : t_17952, t_21591, t_89198
        forward_only=False)                                                    # trace_info : t_17953, t_21592, t_89199

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:                                    # trace_info : t_19852, t_23489, t_91096
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_19853, t_23490, t_91097
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)  # trace_info : t_19854, t_23491, t_91098
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()         # trace_info : t_19861, t_23498, t_91105
    timers('optimizer').stop()                                                 # trace_info : t_21014, t_24651, t_92258

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":# trace_info : t_21022, t_24659, t_92266
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:                                                      # trace_info : t_21023, t_24660, t_92267
        increment = get_num_microbatches() * \                                 # trace_info : t_21024, t_21028, t_21030, t_24661, t_24665, ...
                    args.micro_batch_size * \                                  # trace_info : t_21027, t_24664, t_92271
                    args.data_parallel_size                                    # trace_info : t_21029, t_24666, t_92273
        opt_param_scheduler.step(increment=increment)                          # trace_info : t_21031, t_24668, t_92275
        skipped_iter = 0                                                       # trace_info : t_21070, t_24707, t_92314
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:                                    # trace_info : t_21071, t_24708, t_92315
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):                        # trace_info : t_21072, t_24709, t_92316
        # Average loss across microbatches.
        loss_reduced = {}                                                      # trace_info : t_21083, t_24720, t_92327
        for key in losses_reduced[0].keys():                                   # trace_info : t_21084, t_21094, t_24721, t_24731, t_92328, ...
            numerator = 0                                                      # trace_info : t_21085, t_24722, t_92329
            denominator = 0                                                    # trace_info : t_21086, t_24723, t_92330
            for x in losses_reduced:                                           # trace_info : t_21087, t_21092, t_24724, t_24729, t_92331, ...
                val = x[key]                                                   # trace_info : t_21088, t_24725, t_92332
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):            # trace_info : t_21089, t_24726, t_92333
                    numerator += val[0]                                        # trace_info : t_21090, t_24727, t_92334
                    denominator += val[1]                                      # trace_info : t_21091, t_24728, t_92335
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator                        # trace_info : t_21093, t_24730, t_92337
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad        # trace_info : t_21095, t_24732, t_92339
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()                                                          # trace_info : t_21174, t_24811, t_92418
    timers = get_timers()                                                      # trace_info : t_21178, t_24815, t_92422
    writer = get_tensorboard_writer()                                          # trace_info : t_21182, t_24819, t_92426
    wandb_writer = get_wandb_writer()                                          # trace_info : t_21184, t_24821, t_92428
    one_logger = get_one_logger()                                              # trace_info : t_21186, t_24823, t_92430

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'                                 # trace_info : t_21188, t_24825, t_92432
    skipped_iters_key = 'skipped iterations'                                   # trace_info : t_21189, t_24826, t_92433
    nan_iters_key = 'nan iterations'                                           # trace_info : t_21190, t_24827, t_92434
    # Advanced iterations.
    if not skipped_iter:                                                       # trace_info : t_21191, t_24828, t_92435
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(             # trace_info : t_21192, t_21194, t_21196, t_24829, t_24831, ...
            advanced_iters_key, 0) + 1                                         # trace_info : t_21193, t_21195, t_24830, t_24832, t_92437, ...
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(                  # trace_info : t_21197, t_21199, t_21201, t_24834, t_24836, ...
        skipped_iters_key, 0) + skipped_iter                                   # trace_info : t_21198, t_21200, t_24835, t_24837, t_92442, ...
    # Update losses and set nan iterations
    got_nan = False                                                            # trace_info : t_21202, t_24839, t_92446
    for key in loss_dict:                                                      # trace_info : t_21203, t_21210, t_24840, t_24847, t_92447, ...
        if not skipped_iter:                                                   # trace_info : t_21204, t_24841, t_92448
            total_loss_dict[key] = total_loss_dict.get(                        # trace_info : t_21205, t_21207, t_21209, t_24842, t_24844, ...
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]# trace_info : t_21206, t_21208, t_24843, t_24845, t_92450, ...
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(                      # trace_info : t_21211, t_21213, t_21215, t_24848, t_24850, ...
        nan_iters_key, 0) + int(got_nan)                                       # trace_info : t_21212, t_21214, t_24849, t_24851, t_92456, ...

    # Logging.
    timers_to_log = [                                                          # trace_info : t_21216, t_24853, t_92460
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
    batch_size = args.micro_batch_size * args.data_parallel_size * \           # trace_info : t_21217, t_21221, t_24854, t_24858, t_92461, ...
        get_num_microbatches()                                                 # trace_info : t_21218, t_24855, t_92462

    # Track app tag & app tag ID
    if one_logger:                                                             # trace_info : t_21222, t_24859, t_92466
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \                 # trace_info : t_21223, t_21225, t_24860, t_24862, t_92467, ...
                       total_loss_dict[skipped_iters_key]                      # trace_info : t_21224, t_24861, t_92468

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \                                    # trace_info : t_21226, t_24863, t_92470
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):            # trace_info : t_21227, t_24864, t_92471
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
    if args.num_experts is not None:                                           # trace_info : t_21228, t_24865, t_92472
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:                                     # trace_info : t_21229, t_24866, t_92473
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

    return report_memory_flag                                                  # trace_info : t_21230, t_24867, t_92474


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
    args = get_args()                                                          # trace_info : t_24892
    timers = get_timers()                                                      # trace_info : t_24896
    # Extra barrier is added to make sure all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)                 # trace_info : t_24900
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler,          # trace_info : t_24921, t_24923
                    num_floating_point_operations_so_far, checkpointing_context)# trace_info : t_24922
    timers('save-checkpoint').stop(barrier=True)                               # trace_info : t_88761
    timers.log(['save-checkpoint'])                                            # trace_info : t_88773

    if args.log_progress:                                                      # trace_info : t_88833
        compute_throughputs_and_append_to_progress_log(iteration,
                                                       num_floating_point_operations_so_far)


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config, checkpointing_context):
    """Train the model function."""
    args = get_args()                                                          # trace_info : t_17591
    timers = get_timers()                                                      # trace_info : t_17595

    # Write args to tensorboard
    write_args_to_tensorboard()                                                # trace_info : t_17599

    # Turn on training mode which enables dropout.
    for model_module in model:                                                 # trace_info : t_17607, t_17609
        model_module.train()                                                   # trace_info : t_17608

    # Tracking loss.
    total_loss_dict = {}                                                       # trace_info : t_17610

    # Iterations.
    iteration = args.iteration                                                 # trace_info : t_17611
    one_logger = get_one_logger()                                              # trace_info : t_17612
    if one_logger:                                                             # trace_info : t_17614
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics({
            'train_samples_start': args.consumed_train_samples,
            'train_iterations_start': iteration,
            'train_samples_target': train_samples_target,
            'train_iterations_target': args.train_iters,
        })

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far# trace_info : t_17615

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss                              # trace_info : t_17616
    config.timers = timers                                                     # trace_info : t_17617
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:                 # trace_info : t_17618
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
    if args.overlap_param_gather and args.delay_param_gather:                  # trace_info : t_17619
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads                    # trace_info : t_17620

    timers('interval-time', log_level=0).start(barrier=True)                   # trace_info : t_17621
    print_datetime('before the start of training step')                        # trace_info : t_17642
    report_memory_flag = True                                                  # trace_info : t_17649
    exit = False                                                               # trace_info : t_17650

    if args.manual_gc:                                                         # trace_info : t_17651
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:                                                     # trace_info : t_17652
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    total_flops = 0.0                                                          # trace_info : t_17653

    num_microbatches = get_num_microbatches()                                  # trace_info : t_17654
    eval_duration = 0.0                                                        # trace_info : t_17657
    eval_iterations = 0                                                        # trace_info : t_17658
    def track_e2e_metrics():                                                   # trace_info : t_17659
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:                                                         # trace_info : t_92489
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

    while iteration < args.train_iters:                                        # trace_info : t_17660, t_21243, t_88850, t_92487
        if args.profile and \                                                  # trace_info : t_17661, t_21244, t_88851
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        maybe_finalize_async_save(False)                                       # trace_info : t_17662, t_21245, t_88852

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)# trace_info : t_17669, t_21252, t_88859
        if get_num_microbatches() != num_microbatches and iteration != 0:      # trace_info : t_17674, t_21257, t_88864
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context)
        num_microbatches = get_num_microbatches()                              # trace_info : t_17677, t_21260, t_88867
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)# trace_info : t_17680, t_21263, t_88870

        args.curr_iteration = iteration                                        # trace_info : t_17685, t_21268, t_88875
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \              # trace_info : t_21096, t_24733, t_92340
            train_step(forward_step_func,                                      # trace_info : t_17686, t_17692, t_21269, t_21275, t_88876, ...
                       train_data_iterator,                                    # trace_info : t_17687, t_21270, t_88877
                       model,                                                  # trace_info : t_17688, t_21271, t_88878
                       optimizer,                                              # trace_info : t_17689, t_21272, t_88879
                       opt_param_scheduler,                                    # trace_info : t_17690, t_21273, t_88880
                       config)                                                 # trace_info : t_17691, t_21274, t_88881
        iteration += 1                                                         # trace_info : t_21097, t_24734, t_92341
        batch_size = mpu.get_data_parallel_world_size() * \                    # trace_info : t_21098, t_21107, t_21111, t_24735, t_24744, ...
                     args.micro_batch_size * \                                 # trace_info : t_21106, t_24743, t_92350
                     get_num_microbatches()                                    # trace_info : t_21108, t_24745, t_92352
        args.consumed_train_samples += batch_size                              # trace_info : t_21112, t_24749, t_92356
        num_fp_ops = num_floating_point_operations(args, batch_size)           # trace_info : t_21113, t_24750, t_92357
        num_floating_point_operations_so_far += num_fp_ops                     # trace_info : t_21148, t_24785, t_92392
        total_flops += num_fp_ops                                              # trace_info : t_21149, t_24786, t_92393

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()                         # trace_info : t_21150, t_24787, t_92394
        params_norm = None                                                     # trace_info : t_21154, t_24791, t_92398
        if args.log_params_norm:                                               # trace_info : t_21155, t_24792, t_92399
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:                                 # trace_info : t_21156, t_24793, t_92400
            track_e2e_metrics()

        learning_rate = None                                                   # trace_info : t_21157, t_24794, t_92401
        decoupled_learning_rate = None                                         # trace_info : t_21158, t_24795, t_92402
        for param_group in optimizer.param_groups:                             # trace_info : t_21159, t_21163, t_21166, t_24796, t_24800, ...
            if param_group['is_decoupled_lr']:                                 # trace_info : t_21161, t_21164, t_24798, t_24801, t_92405, ...
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']                              # trace_info : t_21162, t_21165, t_24799, t_24802, t_92406, ...
        report_memory_flag = training_log(loss_dict, total_loss_dict,          # trace_info : t_21167, t_21173, t_24804, t_24810, t_92411, ...
                                          learning_rate,                       # trace_info : t_21168, t_24805, t_92412
                                          decoupled_learning_rate,             # trace_info : t_21169, t_24806, t_92413
                                          iteration, loss_scale,               # trace_info : t_21170, t_24807, t_92414
                                          report_memory_flag, skipped_iter,    # trace_info : t_21171, t_24808, t_92415
                                          grad_norm, params_norm, num_zeros_in_grad)# trace_info : t_21172, t_24809, t_92416
        # StragglerDetector
        if iteration % args.log_interval == 0 and args.log_straggler:          # trace_info : t_21231, t_24868, t_92475
            stimer.report(total_flops, args.log_interval)
            total_flops = 0.0

        if args.check_weight_hash_across_dp_replicas_interval is not None and \# trace_info : t_21232, t_24869, t_92476
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
        if args.adlr_autoresume and \                                          # trace_info : t_21233, t_24870, t_92477
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \    # trace_info : t_21234, t_24871, t_92478
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
        saved_checkpoint = False                                               # trace_info : t_21235, t_24872, t_92479
        if args.exit_signal_handler:                                           # trace_info : t_21236, t_24873, t_92480
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \                              # trace_info : t_21237, t_24874, t_92481
           iteration % args.save_interval == 0:                                # trace_info : t_21238, t_24875, t_92482
            timers('interval-time').stop()                                     # trace_info : t_24876
            save_checkpoint_and_time(iteration, model, optimizer,              # trace_info : t_24887, t_24891
                                     opt_param_scheduler,                      # trace_info : t_24888
                                     num_floating_point_operations_so_far,     # trace_info : t_24889
                                     checkpointing_context)                    # trace_info : t_24890
            saved_checkpoint = True                                            # trace_info : t_88834
            timers('interval-time', log_level=0).start(barrier=True)           # trace_info : t_88835

        # Exiting based on duration
        if args.exit_duration_in_mins:                                         # trace_info : t_21239, t_88846, t_92483
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
        if args.exit_interval and iteration % args.exit_interval == 0:         # trace_info : t_21240, t_88847, t_92484
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.profile and \                                                  # trace_info : t_21241, t_88848, t_92485
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()

        if args.manual_gc:                                                     # trace_info : t_21242, t_88849, t_92486
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    track_e2e_metrics()                                                        # trace_info : t_92488

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()                                          # trace_info : t_92490
    if writer:                                                                 # trace_info : t_92492
        writer.flush()
    wandb_writer = get_wandb_writer()                                          # trace_info : t_92493
    if wandb_writer:                                                           # trace_info : t_92495
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:           # trace_info : t_92496
        optimizer.disable_pre_hook()

    maybe_finalize_async_save(True)                                            # trace_info : t_92497

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:                                                                   # trace_info : t_92504
        sys.exit()

    return iteration, num_floating_point_operations_so_far                     # trace_info : t_92505


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

    args = get_args()                                                          # trace_info : t_16021

    # Number of train/valid/test samples.
    if args.train_samples:                                                     # trace_info : t_16025
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size              # trace_info : t_16026
    eval_iters = (args.train_iters // args.eval_interval + 1) * \              # trace_info : t_16027, t_16029
                 args.eval_iters                                               # trace_info : t_16028
    test_iters = args.eval_iters                                               # trace_info : t_16030

    return (                                                                   # trace_info : t_16034
        train_samples,                                                         # trace_info : t_16031
        eval_iters * args.global_batch_size,                                   # trace_info : t_16032
        test_iters * args.global_batch_size,                                   # trace_info : t_16033
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()          # trace_info : t_16020
    print_rank_0(' > datasets target sizes (minimum size):')                   # trace_info : t_16035
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0])) # trace_info : t_16039
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1])) # trace_info : t_16043
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2])) # trace_info : t_16047
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)# trace_info : t_16051


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()                                                          # trace_info : t_16004

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None) # trace_info : t_16008

    print_rank_0('> building train, validation, and test datasets ...')        # trace_info : t_16009

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:                # trace_info : t_16013
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:                # trace_info : t_16014
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)# trace_info : t_16015

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:            # trace_info : t_16016

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(         # trace_info : t_16017, t_16019
            build_train_valid_test_datasets_provider)                          # trace_info : t_16018
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(                      # trace_info : t_17066, t_17068
            train_ds, args.consumed_train_samples)                             # trace_info : t_17067
        if args.skip_train:                                                    # trace_info : t_17115
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(                  # trace_info : t_17116, t_17118
                valid_ds, args.consumed_valid_samples)                         # trace_info : t_17117
        test_dataloader = build_pretraining_data_loader(test_ds, 0)            # trace_info : t_17165

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0       # trace_info : t_17212
        do_valid = valid_dataloader is not None and args.eval_iters > 0        # trace_info : t_17213
        do_test = test_dataloader is not None and args.eval_iters > 0          # trace_info : t_17214
        flags = torch.tensor(                                                  # trace_info : t_17215, t_17218
            [int(do_train), int(do_valid), int(do_test)],                      # trace_info : t_17216
            dtype=torch.long, device='cuda')                                   # trace_info : t_17217
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)                                      # trace_info : t_17219

    args.do_train = getattr(args, "do_train", False) or flags[0].item()        # trace_info : t_17220
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()        # trace_info : t_17221
    args.do_test = getattr(args, "do_test", False) or flags[2].item()          # trace_info : t_17222

    return train_dataloader, valid_dataloader, test_dataloader                 # trace_info : t_17223


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()                                                          # trace_info : t_15997

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \                    # trace_info : t_17224
        build_train_valid_test_data_loaders(                                   # trace_info : t_16001, t_16003
            build_train_valid_test_datasets_provider)                          # trace_info : t_16002

    # Build iterators.
    dl_type = args.dataloader_type                                             # trace_info : t_17225
    assert dl_type in ['single', 'cyclic', 'external']                         # trace_info : t_17226

    def _get_iterator(dataloader_type, dataloader):                            # trace_info : t_17227
        """Return dataset iterator."""
        if dataloader_type == "single":                                        # trace_info : t_17230, t_17308, t_17386
            return iter(dataloader)                                            # trace_info : t_17231, t_17309, t_17387
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:                                           # trace_info : t_17228
        train_data_iterator = _get_iterator(dl_type, train_dataloader)         # trace_info : t_17229
    else:
        train_data_iterator = None

    if valid_dataloader is not None:                                           # trace_info : t_17306
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)         # trace_info : t_17307
    else:
        valid_data_iterator = None

    if test_dataloader is not None:                                            # trace_info : t_17384
        test_data_iterator = _get_iterator(dl_type, test_dataloader)           # trace_info : t_17385
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator        # trace_info : t_17462
