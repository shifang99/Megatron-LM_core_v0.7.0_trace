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
    args = get_args()                                                          # trace_info : t_9196
    use_te = args.transformer_impl == "transformer_engine"                     # trace_info : t_9200

    print_rank_0('building GPT model ...')                                     # trace_info : t_9201
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:                                              # trace_info : t_9205
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)                       # trace_info : t_9206

    if args.use_mcore_models:                                                  # trace_info : t_9690
        if args.spec is not None:                                              # trace_info : t_9691
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:                                                         # trace_info : t_9692
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)# trace_info : t_9693

        model = GPTModel(                                                      # trace_info : t_9757, t_9769
            config=config,                                                     # trace_info : t_9758
            transformer_layer_spec=transformer_layer_spec,                     # trace_info : t_9759
            vocab_size=args.padded_vocab_size,                                 # trace_info : t_9760
            max_sequence_length=args.max_position_embeddings,                  # trace_info : t_9761
            pre_process=pre_process,                                           # trace_info : t_9762
            post_process=post_process,                                         # trace_info : t_9763
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,                  # trace_info : t_9764
            parallel_output=True,                                              # trace_info : t_9765
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,# trace_info : t_9766
            position_embedding_type=args.position_embedding_type,              # trace_info : t_9767
            rotary_percent=args.rotary_percent,                                # trace_info : t_9768
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

    return model                                                               # trace_info : t_12014


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):# trace_info : t_18196, t_21926, t_25654
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)                           # trace_info : t_18205, t_21935, t_25663

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)                                   # trace_info : t_18269, t_21999, t_25727

    return batch.values()                                                      # trace_info : t_18277, t_22007, t_25735


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
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()                                                          # trace_info : t_18172, t_21902, t_25630
    timers = get_timers()                                                      # trace_info : t_18176, t_21906, t_25634

    # Get the batch.
    timers('batch-generator', log_level=2).start()                             # trace_info : t_18180, t_21910, t_25638
    global stimer
    with stimer(bdata=True):                                                   # trace_info : t_18187, t_18278, t_21917, t_22008, t_25645, ...
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(   # trace_info : t_18193, t_18195, t_21923, t_21925, t_25651, ...
            data_iterator)                                                     # trace_info : t_18194, t_21924, t_25652
    timers('batch-generator').stop()                                           # trace_info : t_18284, t_22014, t_25742

    with stimer:                                                               # trace_info : t_18292, t_19408, t_22022, t_23136, t_25750, ...
        output_tensor = model(tokens, position_ids, attention_mask,            # trace_info : t_18296, t_18298, t_22026, t_22028, t_25754, ...
                              labels=labels)                                   # trace_info : t_18297, t_22027, t_25755

    return output_tensor, partial(loss_func, loss_mask)                        # trace_info : t_19414, t_23142, t_26870


def is_dataset_built_on_rank():
    return (                                                                   # trace_info : t_16264, t_16271, t_16463, t_16470, t_16678, ...
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()          # trace_info : t_16255, t_16454, t_16669, t_16884
    ) and mpu.get_tensor_model_parallel_rank() == 0                            # trace_info : t_16265, t_16464, t_16679, t_16894


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()                                                # trace_info : t_16094

    return GPTDatasetConfig(                                                   # trace_info : t_16098, t_16127
        random_seed=args.seed,                                                 # trace_info : t_16099
        sequence_length=args.seq_length,                                       # trace_info : t_16100
        blend=get_blend_from_list(args.data_path),                             # trace_info : t_16101
        blend_per_split=[                                                      # trace_info : t_16117
            get_blend_from_list(args.train_data_path),                         # trace_info : t_16108
            get_blend_from_list(args.valid_data_path),                         # trace_info : t_16111
            get_blend_from_list(args.test_data_path)                           # trace_info : t_16114
        ],
        split=args.split,                                                      # trace_info : t_16118
        num_dataset_builder_threads=args.num_dataset_builder_threads,          # trace_info : t_16119
        path_to_cache=args.data_cache_path,                                    # trace_info : t_16120
        mmap_bin_files=args.mmap_bin_files,                                    # trace_info : t_16121
        tokenizer=tokenizer,                                                   # trace_info : t_16122
        reset_position_ids=args.reset_position_ids,                            # trace_info : t_16123
        reset_attention_mask=args.reset_attention_mask,                        # trace_info : t_16124
        eod_mask_loss=args.eod_mask_loss,                                      # trace_info : t_16125
        create_attention_mask=args.create_attention_mask_in_dataloader,        # trace_info : t_16126
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()                                                          # trace_info : t_16089

    config = core_gpt_dataset_config_from_args(args)                           # trace_info : t_16093

    if args.mock_data:                                                         # trace_info : t_16203
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset                                              # trace_info : t_16204

    print_rank_0("> building train, validation, and test datasets for GPT ...")# trace_info : t_16205

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(               # trace_info : t_16209, t_16214, t_17097
        dataset_type,                                                          # trace_info : t_16210
        train_val_test_num_samples,                                            # trace_info : t_16211
        is_dataset_built_on_rank,                                              # trace_info : t_16212
        config                                                                 # trace_info : t_16213
    ).build()                                                                  # trace_info : t_16273

    print_rank_0("> finished creating GPT datasets ...")                       # trace_info : t_17098

    return train_ds, valid_ds, test_ds                                         # trace_info : t_17102


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
