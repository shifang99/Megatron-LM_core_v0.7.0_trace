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
    args = get_args()                                                          # trace_info : t_9141
    use_te = args.transformer_impl == "transformer_engine"                     # trace_info : t_9145

    print_rank_0('building GPT model ...')                                     # trace_info : t_9146
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:                                              # trace_info : t_9150
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)                       # trace_info : t_9151

    if args.use_mcore_models:                                                  # trace_info : t_9634
        if args.spec is not None:                                              # trace_info : t_9635
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:                                                         # trace_info : t_9636
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)# trace_info : t_9637
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        model = GPTModel(                                                      # trace_info : t_9697, t_9709
            config=config,                                                     # trace_info : t_9698
            transformer_layer_spec=transformer_layer_spec,                     # trace_info : t_9699
            vocab_size=args.padded_vocab_size,                                 # trace_info : t_9700
            max_sequence_length=args.max_position_embeddings,                  # trace_info : t_9701
            pre_process=pre_process,                                           # trace_info : t_9702
            post_process=post_process,                                         # trace_info : t_9703
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,                  # trace_info : t_9704
            parallel_output=True,                                              # trace_info : t_9705
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,# trace_info : t_9706
            position_embedding_type=args.position_embedding_type,              # trace_info : t_9707
            rotary_percent=args.rotary_percent,                                # trace_info : t_9708
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

    return model                                                               # trace_info : t_11553


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):# trace_info : t_17935, t_21121, t_24307
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)                           # trace_info : t_17944, t_21130, t_24316

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)                                   # trace_info : t_18015, t_21201, t_24387

    return batch.values()                                                      # trace_info : t_18130, t_21316, t_24502


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
    args = get_args()                                                          # trace_info : t_18850, t_22036, t_25222

    losses = output_tensor.float()                                             # trace_info : t_18854, t_22040, t_25226
    loss_mask = loss_mask.view(-1).float()                                     # trace_info : t_18855, t_22041, t_25227
    total_tokens = loss_mask.sum()                                             # trace_info : t_18856, t_22042, t_25228
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])# trace_info : t_18857, t_22043, t_25229

    if args.context_parallel_size > 1:                                         # trace_info : t_18858, t_22044, t_25230
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())# trace_info : t_18859, t_22045, t_25231

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:                                    # trace_info : t_18863, t_22049, t_25235
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()                                     # trace_info : t_18864, t_22050, t_25236
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())# trace_info : t_18865, t_22051, t_25237

    local_num_tokens = loss[1].clone().detach().to(torch.int)                  # trace_info : t_18869, t_22055, t_25241
    return (                                                                   # trace_info : t_18873, t_22059, t_25245
        loss[0] * args.context_parallel_size,                                  # trace_info : t_18870, t_22056, t_25242
        local_num_tokens,                                                      # trace_info : t_18871, t_22057, t_25243
        {'lm loss': (reporting_loss[0], reporting_loss[1])},                   # trace_info : t_18872, t_22058, t_25244
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()                                                          # trace_info : t_17911, t_21097, t_24283
    timers = get_timers()                                                      # trace_info : t_17915, t_21101, t_24287

    # Get the batch.
    timers('batch-generator', log_level=2).start()                             # trace_info : t_17919, t_21105, t_24291
    global stimer
    with stimer(bdata=True):                                                   # trace_info : t_17926, t_18131, t_21112, t_21317, t_24298, ...
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(   # trace_info : t_17932, t_17934, t_21118, t_21120, t_24304, ...
            data_iterator)                                                     # trace_info : t_17933, t_21119, t_24305
    timers('batch-generator').stop()                                           # trace_info : t_18137, t_21323, t_24509

    with stimer:                                                               # trace_info : t_18145, t_18824, t_21331, t_22010, t_24517, ...
        output_tensor = model(tokens, position_ids, attention_mask,            # trace_info : t_18149, t_18151, t_21335, t_21337, t_24521, ...
                              labels=labels)                                   # trace_info : t_18150, t_21336, t_24522

    return output_tensor, partial(loss_func, loss_mask)                        # trace_info : t_18830, t_22016, t_25202


def is_dataset_built_on_rank():
    return (                                                                   # trace_info : t_16057, t_16064, t_16256, t_16263, t_16471, ...
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()          # trace_info : t_16048, t_16247, t_16462, t_16677
    ) and mpu.get_tensor_model_parallel_rank() == 0                            # trace_info : t_16058, t_16257, t_16472, t_16687


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()                                                # trace_info : t_15887

    return GPTDatasetConfig(                                                   # trace_info : t_15891, t_15920
        random_seed=args.seed,                                                 # trace_info : t_15892
        sequence_length=args.seq_length,                                       # trace_info : t_15893
        blend=get_blend_from_list(args.data_path),                             # trace_info : t_15894
        blend_per_split=[                                                      # trace_info : t_15910
            get_blend_from_list(args.train_data_path),                         # trace_info : t_15901
            get_blend_from_list(args.valid_data_path),                         # trace_info : t_15904
            get_blend_from_list(args.test_data_path)                           # trace_info : t_15907
        ],
        split=args.split,                                                      # trace_info : t_15911
        num_dataset_builder_threads=args.num_dataset_builder_threads,          # trace_info : t_15912
        path_to_cache=args.data_cache_path,                                    # trace_info : t_15913
        mmap_bin_files=args.mmap_bin_files,                                    # trace_info : t_15914
        tokenizer=tokenizer,                                                   # trace_info : t_15915
        reset_position_ids=args.reset_position_ids,                            # trace_info : t_15916
        reset_attention_mask=args.reset_attention_mask,                        # trace_info : t_15917
        eod_mask_loss=args.eod_mask_loss,                                      # trace_info : t_15918
        create_attention_mask=args.create_attention_mask_in_dataloader,        # trace_info : t_15919
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()                                                          # trace_info : t_15882

    config = core_gpt_dataset_config_from_args(args)                           # trace_info : t_15886

    if args.mock_data:                                                         # trace_info : t_15996
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset                                              # trace_info : t_15997

    print_rank_0("> building train, validation, and test datasets for GPT ...")# trace_info : t_15998

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(               # trace_info : t_16002, t_16007, t_16890
        dataset_type,                                                          # trace_info : t_16003
        train_val_test_num_samples,                                            # trace_info : t_16004
        is_dataset_built_on_rank,                                              # trace_info : t_16005
        config                                                                 # trace_info : t_16006
    ).build()                                                                  # trace_info : t_16066

    print_rank_0("> finished creating GPT datasets ...")                       # trace_info : t_16891

    return train_ds, valid_ds, test_ds                                         # trace_info : t_16895


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
