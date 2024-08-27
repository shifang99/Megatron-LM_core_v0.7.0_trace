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
    args = get_args()                                                          # trace_info : t_10698
    use_te = args.transformer_impl == "transformer_engine"                     # trace_info : t_10702

    print_rank_0('building GPT model ...')                                     # trace_info : t_10703
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:                                              # trace_info : t_10707
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)                       # trace_info : t_10708

    if args.use_mcore_models:                                                  # trace_info : t_11191
        if args.spec is not None:                                              # trace_info : t_11192
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:                                                         # trace_info : t_11193
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)# trace_info : t_11194

        model = GPTModel(                                                      # trace_info : t_11258, t_11270
            config=config,                                                     # trace_info : t_11259
            transformer_layer_spec=transformer_layer_spec,                     # trace_info : t_11260
            vocab_size=args.padded_vocab_size,                                 # trace_info : t_11261
            max_sequence_length=args.max_position_embeddings,                  # trace_info : t_11262
            pre_process=pre_process,                                           # trace_info : t_11263
            post_process=post_process,                                         # trace_info : t_11264
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,                  # trace_info : t_11265
            parallel_output=True,                                              # trace_info : t_11266
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,# trace_info : t_11267
            position_embedding_type=args.position_embedding_type,              # trace_info : t_11268
            rotary_percent=args.rotary_percent,                                # trace_info : t_11269
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

    return model                                                               # trace_info : t_13561


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):# trace_info : t_19923, t_23535, t_27145
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)                           # trace_info : t_19932, t_23544, t_27154

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)                                   # trace_info : t_20003, t_23615, t_27225

    return batch.values()                                                      # trace_info : t_20011, t_23623, t_27233


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
    args = get_args()                                                          # trace_info : t_21289, t_24899, t_28509

    losses = output_tensor.float()                                             # trace_info : t_21293, t_24903, t_28513
    loss_mask = loss_mask.view(-1).float()                                     # trace_info : t_21294, t_24904, t_28514
    total_tokens = loss_mask.sum()                                             # trace_info : t_21295, t_24905, t_28515
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])# trace_info : t_21296, t_24906, t_28516

    if args.context_parallel_size > 1:                                         # trace_info : t_21297, t_24907, t_28517
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:                                    # trace_info : t_21298, t_24908, t_28518
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()                                     # trace_info : t_21299, t_24909, t_28519
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())# trace_info : t_21300, t_24910, t_28520

    local_num_tokens = loss[1].clone().detach().to(torch.int)                  # trace_info : t_21304, t_24914, t_28524
    return (                                                                   # trace_info : t_21308, t_24918, t_28528
        loss[0] * args.context_parallel_size,                                  # trace_info : t_21305, t_24915, t_28525
        local_num_tokens,                                                      # trace_info : t_21306, t_24916, t_28526
        {'lm loss': (reporting_loss[0], reporting_loss[1])},                   # trace_info : t_21307, t_24917, t_28527
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()                                                          # trace_info : t_19899, t_23511, t_27121
    timers = get_timers()                                                      # trace_info : t_19903, t_23515, t_27125

    # Get the batch.
    timers('batch-generator', log_level=2).start()                             # trace_info : t_19907, t_23519, t_27129
    global stimer
    with stimer(bdata=True):                                                   # trace_info : t_19914, t_20012, t_23526, t_23624, t_27136, ...
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(   # trace_info : t_19920, t_19922, t_23532, t_23534, t_27142, ...
            data_iterator)                                                     # trace_info : t_19921, t_23533, t_27143
    timers('batch-generator').stop()                                           # trace_info : t_20018, t_23630, t_27240

    with stimer:                                                               # trace_info : t_20026, t_21263, t_23638, t_24873, t_27248, ...
        output_tensor = model(tokens, position_ids, attention_mask,            # trace_info : t_20030, t_20032, t_23642, t_23644, t_27252, ...
                              labels=labels)                                   # trace_info : t_20031, t_23643, t_27253

    return output_tensor, partial(loss_func, loss_mask)                        # trace_info : t_21269, t_24879, t_28489


def is_dataset_built_on_rank():
    return (                                                                   # trace_info : t_18065, t_18072, t_18264, t_18271, t_18479, ...
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()          # trace_info : t_18056, t_18255, t_18470, t_18685
    ) and mpu.get_tensor_model_parallel_rank() == 0                            # trace_info : t_18066, t_18265, t_18480, t_18695


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()                                                # trace_info : t_17895

    return GPTDatasetConfig(                                                   # trace_info : t_17899, t_17928
        random_seed=args.seed,                                                 # trace_info : t_17900
        sequence_length=args.seq_length,                                       # trace_info : t_17901
        blend=get_blend_from_list(args.data_path),                             # trace_info : t_17902
        blend_per_split=[                                                      # trace_info : t_17918
            get_blend_from_list(args.train_data_path),                         # trace_info : t_17909
            get_blend_from_list(args.valid_data_path),                         # trace_info : t_17912
            get_blend_from_list(args.test_data_path)                           # trace_info : t_17915
        ],
        split=args.split,                                                      # trace_info : t_17919
        num_dataset_builder_threads=args.num_dataset_builder_threads,          # trace_info : t_17920
        path_to_cache=args.data_cache_path,                                    # trace_info : t_17921
        mmap_bin_files=args.mmap_bin_files,                                    # trace_info : t_17922
        tokenizer=tokenizer,                                                   # trace_info : t_17923
        reset_position_ids=args.reset_position_ids,                            # trace_info : t_17924
        reset_attention_mask=args.reset_attention_mask,                        # trace_info : t_17925
        eod_mask_loss=args.eod_mask_loss,                                      # trace_info : t_17926
        create_attention_mask=args.create_attention_mask_in_dataloader,        # trace_info : t_17927
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()                                                          # trace_info : t_17890

    config = core_gpt_dataset_config_from_args(args)                           # trace_info : t_17894

    if args.mock_data:                                                         # trace_info : t_18004
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset                                              # trace_info : t_18005

    print_rank_0("> building train, validation, and test datasets for GPT ...")# trace_info : t_18006

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(               # trace_info : t_18010, t_18015, t_18898
        dataset_type,                                                          # trace_info : t_18011
        train_val_test_num_samples,                                            # trace_info : t_18012
        is_dataset_built_on_rank,                                              # trace_info : t_18013
        config                                                                 # trace_info : t_18014
    ).build()                                                                  # trace_info : t_18074

    print_rank_0("> finished creating GPT datasets ...")                       # trace_info : t_18899

    return train_ds, valid_ds, test_ds                                         # trace_info : t_18903


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
