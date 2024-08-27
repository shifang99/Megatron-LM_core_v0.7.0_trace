# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()                                                          # trace_info : t_5831
    use_te = args.transformer_impl == "transformer_engine"                     # trace_info : t_5835

    print_rank_0('building GPT model ...')                                     # trace_info : t_5836
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:                                              # trace_info : t_5840
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)                       # trace_info : t_5841

    if args.use_mcore_models:                                                  # trace_info : t_6324
        if args.spec is not None:                                              # trace_info : t_6325
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:                                                         # trace_info : t_6326
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)# trace_info : t_6327

        model = GPTModel(                                                      # trace_info : t_6391, t_6403
            config=config,                                                     # trace_info : t_6392
            transformer_layer_spec=transformer_layer_spec,                     # trace_info : t_6393
            vocab_size=args.padded_vocab_size,                                 # trace_info : t_6394
            max_sequence_length=args.max_position_embeddings,                  # trace_info : t_6395
            pre_process=pre_process,                                           # trace_info : t_6396
            post_process=post_process,                                         # trace_info : t_6397
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,                  # trace_info : t_6398
            parallel_output=True,                                              # trace_info : t_6399
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,# trace_info : t_6400
            position_embedding_type=args.position_embedding_type,              # trace_info : t_6401
            rotary_percent=args.rotary_percent,                                # trace_info : t_6402
        )
    else:
        assert (
            args.context_parallel_size == 1
        ), "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

    return model                                                               # trace_info : t_8694


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):# trace_info : t_15057, t_18698, t_22337
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)                           # trace_info : t_15066, t_18707, t_22346

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)                                   # trace_info : t_15137, t_18778, t_22417

    return batch.values()                                                      # trace_info : t_15145, t_18786, t_22425


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()                                                          # trace_info : t_16456, t_20095, t_23734

    losses = output_tensor.float()                                             # trace_info : t_16460, t_20099, t_23738
    loss_mask = loss_mask.view(-1).float()                                     # trace_info : t_16461, t_20100, t_23739
    total_tokens = loss_mask.sum()                                             # trace_info : t_16462, t_20101, t_23740
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])# trace_info : t_16463, t_20102, t_23741

    if args.context_parallel_size > 1:                                         # trace_info : t_16464, t_20103, t_23742
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:                                    # trace_info : t_16465, t_20104, t_23743
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()                                     # trace_info : t_16466, t_20105, t_23744
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())# trace_info : t_16467, t_20106, t_23745

    local_num_tokens = loss[1].clone().detach().to(torch.int)                  # trace_info : t_16471, t_20110, t_23749
    return (                                                                   # trace_info : t_16475, t_20114, t_23753
        loss[0] * args.context_parallel_size,                                  # trace_info : t_16472, t_20111, t_23750
        local_num_tokens,                                                      # trace_info : t_16473, t_20112, t_23751
        {'lm loss': (reporting_loss[0], reporting_loss[1])},                   # trace_info : t_16474, t_20113, t_23752
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()                                                          # trace_info : t_15033, t_18674, t_22313
    timers = get_timers()                                                      # trace_info : t_15037, t_18678, t_22317

    # Get the batch.
    timers('batch-generator', log_level=2).start()                             # trace_info : t_15041, t_18682, t_22321
    global stimer
    with stimer(bdata=True):                                                   # trace_info : t_15048, t_15146, t_18689, t_18787, t_22328, ...
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(   # trace_info : t_15054, t_15056, t_18695, t_18697, t_22334, ...
            data_iterator)                                                     # trace_info : t_15055, t_18696, t_22335
    timers('batch-generator').stop()                                           # trace_info : t_15152, t_18793, t_22432

    with stimer:                                                               # trace_info : t_15160, t_16430, t_18801, t_20069, t_22440, ...
        output_tensor = model(tokens, position_ids, attention_mask,            # trace_info : t_15164, t_15166, t_18805, t_18807, t_22444, ...
                              labels=labels)                                   # trace_info : t_15165, t_18806, t_22445

    return output_tensor, partial(loss_func, loss_mask)                        # trace_info : t_16436, t_20075, t_23714


def is_dataset_built_on_rank():
    return (                                                                   # trace_info : t_13198, t_13205, t_13397, t_13404, t_13612, ...
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()          # trace_info : t_13189, t_13388, t_13603, t_13818
    ) and mpu.get_tensor_model_parallel_rank() == 0                            # trace_info : t_13199, t_13398, t_13613, t_13828


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()                                                # trace_info : t_13028

    return GPTDatasetConfig(                                                   # trace_info : t_13032, t_13061
        random_seed=args.seed,                                                 # trace_info : t_13033
        sequence_length=args.seq_length,                                       # trace_info : t_13034
        blend=get_blend_from_list(args.data_path),                             # trace_info : t_13035
        blend_per_split=[                                                      # trace_info : t_13051
            get_blend_from_list(args.train_data_path),                         # trace_info : t_13042
            get_blend_from_list(args.valid_data_path),                         # trace_info : t_13045
            get_blend_from_list(args.test_data_path)                           # trace_info : t_13048
        ],
        split=args.split,                                                      # trace_info : t_13052
        num_dataset_builder_threads=args.num_dataset_builder_threads,          # trace_info : t_13053
        path_to_cache=args.data_cache_path,                                    # trace_info : t_13054
        mmap_bin_files=args.mmap_bin_files,                                    # trace_info : t_13055
        tokenizer=tokenizer,                                                   # trace_info : t_13056
        reset_position_ids=args.reset_position_ids,                            # trace_info : t_13057
        reset_attention_mask=args.reset_attention_mask,                        # trace_info : t_13058
        eod_mask_loss=args.eod_mask_loss,                                      # trace_info : t_13059
        create_attention_mask=args.create_attention_mask_in_dataloader,        # trace_info : t_13060
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()                                                          # trace_info : t_13023

    config = core_gpt_dataset_config_from_args(args)                           # trace_info : t_13027

    if args.mock_data:                                                         # trace_info : t_13137
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset                                              # trace_info : t_13138

    print_rank_0("> building train, validation, and test datasets for GPT ...")# trace_info : t_13139

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(               # trace_info : t_13143, t_13148, t_14031
        dataset_type,                                                          # trace_info : t_13144
        train_val_test_num_samples,                                            # trace_info : t_13145
        is_dataset_built_on_rank,                                              # trace_info : t_13146
        config                                                                 # trace_info : t_13147
    ).build()                                                                  # trace_info : t_13207

    print_rank_0("> finished creating GPT datasets ...")                       # trace_info : t_14032

    return train_ds, valid_ds, test_ds                                         # trace_info : t_14036


if __name__ == "__main__":

    # added by shifangxu
    from trace_cuda_memory import trace_with_params     
    trace_params = {}    
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    # 获取当前文件所在目录的上一级目录
    parent_dir = os.path.dirname(current_dir)
    trace_params["trace_in_which_dir"] = parent_dir # 按需修改
    trace_params["trace_filename"] = f"my_trace_rank{os.environ.get('RANK')}.csv"    
    trace_params["exclude_funcs"] = ["dictcomp", "listcomp"]     
    print(f"trace_params:{trace_params}")
    torch.multiprocessing.set_start_method('spawn',  force=True)         
    pretrain = trace_with_params(trace_params)(pretrain)    

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
