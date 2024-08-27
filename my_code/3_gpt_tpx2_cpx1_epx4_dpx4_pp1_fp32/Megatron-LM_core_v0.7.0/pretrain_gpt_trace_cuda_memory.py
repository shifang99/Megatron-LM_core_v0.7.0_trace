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
    args = get_args()                                                          # trace_info : t_8785
    use_te = args.transformer_impl == "transformer_engine"                     # trace_info : t_8789

    print_rank_0('building GPT model ...')                                     # trace_info : t_8790
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:                                              # trace_info : t_8794
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)                       # trace_info : t_8795

    if args.use_mcore_models:                                                  # trace_info : t_9280
        if args.spec is not None:                                              # trace_info : t_9281
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:                                                         # trace_info : t_9282
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)# trace_info : t_9283

        model = GPTModel(                                                      # trace_info : t_9345, t_9357
            config=config,                                                     # trace_info : t_9346
            transformer_layer_spec=transformer_layer_spec,                     # trace_info : t_9347
            vocab_size=args.padded_vocab_size,                                 # trace_info : t_9348
            max_sequence_length=args.max_position_embeddings,                  # trace_info : t_9349
            pre_process=pre_process,                                           # trace_info : t_9350
            post_process=post_process,                                         # trace_info : t_9351
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,                  # trace_info : t_9352
            parallel_output=True,                                              # trace_info : t_9353
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,# trace_info : t_9354
            position_embedding_type=args.position_embedding_type,              # trace_info : t_9355
            rotary_percent=args.rotary_percent,                                # trace_info : t_9356
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

    return model                                                               # trace_info : t_11924


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):# trace_info : t_18091, t_22444, t_26789
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)                           # trace_info : t_18100, t_22453, t_26798

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)                                   # trace_info : t_18180, t_22533, t_26878

    return batch.values()                                                      # trace_info : t_18188, t_22541, t_26886


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
    args = get_args()                                                          # trace_info : t_20031, t_24376, t_28721

    losses = output_tensor.float()                                             # trace_info : t_20035, t_24380, t_28725
    loss_mask = loss_mask.view(-1).float()                                     # trace_info : t_20036, t_24381, t_28726
    total_tokens = loss_mask.sum()                                             # trace_info : t_20037, t_24382, t_28727
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])# trace_info : t_20038, t_24383, t_28728

    if args.context_parallel_size > 1:                                         # trace_info : t_20039, t_24384, t_28729
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:                                    # trace_info : t_20040, t_24385, t_28730
        global_rank = torch.distributed.get_rank()                             # trace_info : t_20041, t_24386, t_28731
        assert not loss[0].isnan(), (                                          # trace_info : t_20042, t_24387, t_28732
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()                                     # trace_info : t_20043, t_24388, t_28733
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())# trace_info : t_20044, t_24389, t_28734

    local_num_tokens = loss[1].clone().detach().to(torch.int)                  # trace_info : t_20048, t_24393, t_28738
    return (                                                                   # trace_info : t_20052, t_24397, t_28742
        loss[0] * args.context_parallel_size,                                  # trace_info : t_20049, t_24394, t_28739
        local_num_tokens,                                                      # trace_info : t_20050, t_24395, t_28740
        {'lm loss': (reporting_loss[0], reporting_loss[1])},                   # trace_info : t_20051, t_24396, t_28741
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()                                                          # trace_info : t_18067, t_22420, t_26765
    timers = get_timers()                                                      # trace_info : t_18071, t_22424, t_26769

    # Get the batch.
    timers('batch-generator', log_level=2).start()                             # trace_info : t_18075, t_22428, t_26773
    global stimer
    with stimer(bdata=True):                                                   # trace_info : t_18082, t_18189, t_22435, t_22542, t_26780, ...
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(   # trace_info : t_18088, t_18090, t_22441, t_22443, t_26786, ...
            data_iterator)                                                     # trace_info : t_18089, t_22442, t_26787
    timers('batch-generator').stop()                                           # trace_info : t_18195, t_22548, t_26893

    with stimer:                                                               # trace_info : t_18203, t_20005, t_22556, t_24350, t_26901, ...
        output_tensor = model(tokens, position_ids, attention_mask,            # trace_info : t_18207, t_18209, t_22560, t_22562, t_26905, ...
                              labels=labels)                                   # trace_info : t_18208, t_22561, t_26906

    return output_tensor, partial(loss_func, loss_mask)                        # trace_info : t_20011, t_24356, t_28701


def is_dataset_built_on_rank():
    return (                                                                   # trace_info : t_16158, t_16165, t_16357, t_16364, t_16572, ...
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()          # trace_info : t_16149, t_16348, t_16563, t_16778
    ) and mpu.get_tensor_model_parallel_rank() == 0                            # trace_info : t_16159, t_16358, t_16573, t_16788


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()                                                # trace_info : t_15988

    return GPTDatasetConfig(                                                   # trace_info : t_15992, t_16021
        random_seed=args.seed,                                                 # trace_info : t_15993
        sequence_length=args.seq_length,                                       # trace_info : t_15994
        blend=get_blend_from_list(args.data_path),                             # trace_info : t_15995
        blend_per_split=[                                                      # trace_info : t_16011
            get_blend_from_list(args.train_data_path),                         # trace_info : t_16002
            get_blend_from_list(args.valid_data_path),                         # trace_info : t_16005
            get_blend_from_list(args.test_data_path)                           # trace_info : t_16008
        ],
        split=args.split,                                                      # trace_info : t_16012
        num_dataset_builder_threads=args.num_dataset_builder_threads,          # trace_info : t_16013
        path_to_cache=args.data_cache_path,                                    # trace_info : t_16014
        mmap_bin_files=args.mmap_bin_files,                                    # trace_info : t_16015
        tokenizer=tokenizer,                                                   # trace_info : t_16016
        reset_position_ids=args.reset_position_ids,                            # trace_info : t_16017
        reset_attention_mask=args.reset_attention_mask,                        # trace_info : t_16018
        eod_mask_loss=args.eod_mask_loss,                                      # trace_info : t_16019
        create_attention_mask=args.create_attention_mask_in_dataloader,        # trace_info : t_16020
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()                                                          # trace_info : t_15983

    config = core_gpt_dataset_config_from_args(args)                           # trace_info : t_15987

    if args.mock_data:                                                         # trace_info : t_16097
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset                                              # trace_info : t_16098

    print_rank_0("> building train, validation, and test datasets for GPT ...")# trace_info : t_16099

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(               # trace_info : t_16103, t_16108, t_16991
        dataset_type,                                                          # trace_info : t_16104
        train_val_test_num_samples,                                            # trace_info : t_16105
        is_dataset_built_on_rank,                                              # trace_info : t_16106
        config                                                                 # trace_info : t_16107
    ).build()                                                                  # trace_info : t_16167

    print_rank_0("> finished creating GPT datasets ...")                       # trace_info : t_16992

    return train_ds, valid_ds, test_ds                                         # trace_info : t_16996


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
