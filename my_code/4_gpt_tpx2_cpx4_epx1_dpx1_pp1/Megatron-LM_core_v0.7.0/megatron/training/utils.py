# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import os
import sys
from datetime import datetime

import torch

try:
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    multi_tensor_applier = None

try:
    import amp_C
except ImportError:
    amp_C = None

from megatron.training import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import DistributedDataParallel as DDP
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.legacy.model import Float16Module
from megatron.legacy.model.module import param_is_not_shared


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True                                                         # trace_info : t_14276
    if not isinstance(model, list):                                            # trace_info : t_14277
        model = [model]
        return_list = False
    unwrapped_model = []                                                       # trace_info : t_14278
    for model_module in model:                                                 # trace_info : t_14279, t_14286
        while isinstance(model_module, module_instances):                      # trace_info : t_14280, t_14282, t_14284
            model_module = model_module.module                                 # trace_info : t_14281, t_14283
        unwrapped_model.append(model_module)                                   # trace_info : t_14285
    if not return_list:                                                        # trace_info : t_14287
        return unwrapped_model[0]
    return unwrapped_model                                                     # trace_info : t_14288


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if mpu.get_expert_model_parallel_rank() > 0:
                if not getattr(param, 'allreduce', True) and is_not_tp_duplicate:
                    assert param_is_not_shared(param)
                    params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                is_not_shared = param_is_not_shared(param)
                if is_not_shared and is_not_tp_duplicate:
                    params_data.append(param.data.float() if args.bf16 else param.data)

    # Check the availability of apex
    assert multi_tensor_applier is not None and amp_C is not None, \
        "apex is not available, please install it from https://github.com/NVIDIA/apex"

    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    if mpu.get_expert_model_parallel_world_size() == 1:
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    else:
        # Sum across tensor, pipeline and expert model-parallel GPUs.
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_and_expert_parallel_group())
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_pipeline_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    args = get_args()                                                          # trace_info : t_18016, t_21202, t_24388
    cp_size = args.context_parallel_size                                       # trace_info : t_18020, t_21206, t_24392
    if cp_size > 1:                                                            # trace_info : t_18021, t_21207, t_24393
        cp_rank = mpu.get_context_parallel_rank()                              # trace_info : t_18022, t_21208, t_24394
        for key, val in batch.items():                                         # trace_info : t_18028, t_18048, t_18068, t_18088, t_18108, ...
            if val is not None:                                                # trace_info : t_18029, t_18049, t_18069, t_18089, t_18109, ...
                seq_dim = 1 if key != 'attention_mask' else 2                  # trace_info : t_18030, t_18050, t_18070, t_18090, t_18110, ...
                val = val.view(                                                # trace_info : t_18031, t_18033, t_18035, t_18037, t_18039, ...
                    *val.shape[0:seq_dim],                                     # trace_info : t_18032, t_18052, t_18072, t_18092, t_18112, ...
                    2 * cp_size,                                               # trace_info : t_18034, t_18054, t_18074, t_18094, t_18114, ...
                    val.shape[seq_dim] // (2 * cp_size),                       # trace_info : t_18036, t_18056, t_18076, t_18096, t_18116, ...
                    *val.shape[(seq_dim + 1) :],                               # trace_info : t_18038, t_18058, t_18078, t_18098, t_18118, ...
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)],   # trace_info : t_18040, t_18042, t_18044, t_18060, t_18062, ...
                                     device="cpu", pin_memory=True).cuda(non_blocking=True)# trace_info : t_18041, t_18043, t_18061, t_18063, t_18081, ...
                val = val.index_select(seq_dim, index)                         # trace_info : t_18045, t_18065, t_18085, t_18105, t_18125, ...
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])# trace_info : t_18046, t_18066, t_18086, t_18106, t_18126, ...
                batch[key] = val                                               # trace_info : t_18047, t_18067, t_18087, t_18107, t_18127, ...

    return batch                                                               # trace_info : t_18129, t_21315, t_24501


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():                                     # trace_info : t_9046, t_9053, t_9147, t_15762, t_15788, ...
        if torch.distributed.get_rank() == 0:                                  # trace_info : t_9047, t_9054, t_9148, t_15763, t_15789, ...
            print(message, flush=True)                                         # trace_info : t_9048, t_9055, t_9149, t_15764, t_15790, ...
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def append_to_progress_log(string, barrier=True):
    """ Append given string to progress log. """
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    if barrier:
        torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, 'a') as f:
            job_id = os.getenv('SLURM_JOB_ID', '')
            num_gpus = args.world_size
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                    f"# GPUs: {num_gpus}\t{string}\n")


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()                                                          # trace_info : t_17945, t_21131, t_24317

    def _broadcast(item):                                                      # trace_info : t_17949, t_21135, t_24321
       if item is not None:                                                    # trace_info : t_17975, t_17983, t_17991, t_17999, t_18007, ...
           torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())# trace_info : t_17976, t_17984, t_17992, t_18000, t_18008, ...

    if mpu.get_tensor_model_parallel_rank() == 0:                              # trace_info : t_17950, t_21136, t_24322

       if data_iterator is not None:                                           # trace_info : t_17956, t_21142, t_24328
           data = next(data_iterator)                                          # trace_info : t_17957, t_21143, t_24329
       else:
           data = None

       batch = {                                                               # trace_info : t_17972, t_21158, t_24344
           'tokens': data["tokens"].cuda(non_blocking = True),                 # trace_info : t_17967, t_21153, t_24339
           'labels': data["labels"].cuda(non_blocking = True),                 # trace_info : t_17968, t_21154, t_24340
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),           # trace_info : t_17969, t_21155, t_24341
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),# trace_info : t_17970, t_21156, t_24342
           'position_ids': data["position_ids"].cuda(non_blocking = True)      # trace_info : t_17971, t_21157, t_24343
       }

       if args.pipeline_model_parallel_size == 1:                              # trace_info : t_17973, t_21159, t_24345
           _broadcast(batch['tokens'])                                         # trace_info : t_17974, t_21160, t_24346
           _broadcast(batch['labels'])                                         # trace_info : t_17982, t_21168, t_24354
           _broadcast(batch['loss_mask'])                                      # trace_info : t_17990, t_21176, t_24362
           _broadcast(batch['attention_mask'])                                 # trace_info : t_17998, t_21184, t_24370
           _broadcast(batch['position_ids'])                                   # trace_info : t_18006, t_21192, t_24378

       elif mpu.is_pipeline_first_stage():
           _broadcast(batch['tokens'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       elif mpu.is_pipeline_last_stage():
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])

    else:

       tokens=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       labels=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       loss_mask=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
       if args.create_attention_mask_in_dataloader:
           attention_mask=torch.empty(
                (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            )
       else:
           attention_mask=None
       position_ids=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())

       if args.pipeline_model_parallel_size == 1:
           _broadcast(tokens)
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
           _broadcast(position_ids)
 
       elif mpu.is_pipeline_first_stage():
           labels=None
           loss_mask=None
   
           _broadcast(tokens)
           _broadcast(attention_mask)
           _broadcast(position_ids)

       elif mpu.is_pipeline_last_stage():
           tokens=None
           position_ids=None
    
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
 
       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids
       }

    return batch                                                               # trace_info : t_18014, t_21200, t_24386
