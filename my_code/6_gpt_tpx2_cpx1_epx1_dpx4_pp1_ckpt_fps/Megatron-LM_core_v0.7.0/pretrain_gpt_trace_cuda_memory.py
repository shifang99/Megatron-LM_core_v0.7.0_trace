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
    args = get_args()                                                          # trace_info : t_8860
    use_te = args.transformer_impl == "transformer_engine"                     # trace_info : t_8864

    print_rank_0('building GPT model ...')                                     # trace_info : t_8865
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:                                              # trace_info : t_8869
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)                       # trace_info : t_8870

    if args.use_mcore_models:                                                  # trace_info : t_9353
        if args.spec is not None:                                              # trace_info : t_9354
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:                                                         # trace_info : t_9355
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)# trace_info : t_9356

        model = GPTModel(                                                      # trace_info : t_9420, t_9432
            config=config,                                                     # trace_info : t_9421
            transformer_layer_spec=transformer_layer_spec,                     # trace_info : t_9422
            vocab_size=args.padded_vocab_size,                                 # trace_info : t_9423
            max_sequence_length=args.max_position_embeddings,                  # trace_info : t_9424
            pre_process=pre_process,                                           # trace_info : t_9425
            post_process=post_process,                                         # trace_info : t_9426
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,                  # trace_info : t_9427
            parallel_output=True,                                              # trace_info : t_9428
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,# trace_info : t_9429
            position_embedding_type=args.position_embedding_type,              # trace_info : t_9430
            rotary_percent=args.rotary_percent,                                # trace_info : t_9431
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

    return model                                                               # trace_info : t_11723


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):# trace_info : t_18193, t_21832, t_89439
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)                           # trace_info : t_18202, t_21841, t_89448

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)                                   # trace_info : t_18282, t_21921, t_89528

    return batch.values()                                                      # trace_info : t_18290, t_21929, t_89536


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
    args = get_args()                                                          # trace_info : t_19584, t_23221, t_90828

    losses = output_tensor.float()                                             # trace_info : t_19588, t_23225, t_90832
    loss_mask = loss_mask.view(-1).float()                                     # trace_info : t_19589, t_23226, t_90833
    total_tokens = loss_mask.sum()                                             # trace_info : t_19590, t_23227, t_90834
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])# trace_info : t_19591, t_23228, t_90835

    if args.context_parallel_size > 1:                                         # trace_info : t_19592, t_23229, t_90836
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:                                    # trace_info : t_19593, t_23230, t_90837
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()                                     # trace_info : t_19594, t_23231, t_90838
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())# trace_info : t_19595, t_23232, t_90839

    local_num_tokens = loss[1].clone().detach().to(torch.int)                  # trace_info : t_19599, t_23236, t_90843
    return (                                                                   # trace_info : t_19603, t_23240, t_90847
        loss[0] * args.context_parallel_size,                                  # trace_info : t_19600, t_23237, t_90844
        local_num_tokens,                                                      # trace_info : t_19601, t_23238, t_90845
        {'lm loss': (reporting_loss[0], reporting_loss[1])},                   # trace_info : t_19602, t_23239, t_90846
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()                                                          # trace_info : t_18169, t_21808, t_89415
    timers = get_timers()                                                      # trace_info : t_18173, t_21812, t_89419

    # Get the batch.
    timers('batch-generator', log_level=2).start()                             # trace_info : t_18177, t_21816, t_89423
    global stimer
    with stimer(bdata=True):                                                   # trace_info : t_18184, t_18291, t_21823, t_21930, t_89430, ...
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(   # trace_info : t_18190, t_18192, t_21829, t_21831, t_89436, ...
            data_iterator)                                                     # trace_info : t_18191, t_21830, t_89437
    timers('batch-generator').stop()                                           # trace_info : t_18297, t_21936, t_89543

    with stimer:                                                               # trace_info : t_18305, t_19558, t_21944, t_23195, t_89551, ...
        output_tensor = model(tokens, position_ids, attention_mask,            # trace_info : t_18309, t_18311, t_21948, t_21950, t_89555, ...
                              labels=labels)                                   # trace_info : t_18310, t_21949, t_89556

    return output_tensor, partial(loss_func, loss_mask)                        # trace_info : t_19564, t_23201, t_90808


def is_dataset_built_on_rank():
    return (                                                                   # trace_info : t_16227, t_16234, t_16426, t_16433, t_16641, ...
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()          # trace_info : t_16218, t_16417, t_16632, t_16847
    ) and mpu.get_tensor_model_parallel_rank() == 0                            # trace_info : t_16228, t_16427, t_16642, t_16857


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()                                                # trace_info : t_16057

    return GPTDatasetConfig(                                                   # trace_info : t_16061, t_16090
        random_seed=args.seed,                                                 # trace_info : t_16062
        sequence_length=args.seq_length,                                       # trace_info : t_16063
        blend=get_blend_from_list(args.data_path),                             # trace_info : t_16064
        blend_per_split=[                                                      # trace_info : t_16080
            get_blend_from_list(args.train_data_path),                         # trace_info : t_16071
            get_blend_from_list(args.valid_data_path),                         # trace_info : t_16074
            get_blend_from_list(args.test_data_path)                           # trace_info : t_16077
        ],
        split=args.split,                                                      # trace_info : t_16081
        num_dataset_builder_threads=args.num_dataset_builder_threads,          # trace_info : t_16082
        path_to_cache=args.data_cache_path,                                    # trace_info : t_16083
        mmap_bin_files=args.mmap_bin_files,                                    # trace_info : t_16084
        tokenizer=tokenizer,                                                   # trace_info : t_16085
        reset_position_ids=args.reset_position_ids,                            # trace_info : t_16086
        reset_attention_mask=args.reset_attention_mask,                        # trace_info : t_16087
        eod_mask_loss=args.eod_mask_loss,                                      # trace_info : t_16088
        create_attention_mask=args.create_attention_mask_in_dataloader,        # trace_info : t_16089
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()                                                          # trace_info : t_16052

    config = core_gpt_dataset_config_from_args(args)                           # trace_info : t_16056

    if args.mock_data:                                                         # trace_info : t_16166
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset                                              # trace_info : t_16167

    print_rank_0("> building train, validation, and test datasets for GPT ...")# trace_info : t_16168

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(               # trace_info : t_16172, t_16177, t_17060
        dataset_type,                                                          # trace_info : t_16173
        train_val_test_num_samples,                                            # trace_info : t_16174
        is_dataset_built_on_rank,                                              # trace_info : t_16175
        config                                                                 # trace_info : t_16176
    ).build()                                                                  # trace_info : t_16236

    print_rank_0("> finished creating GPT datasets ...")                       # trace_info : t_17061

    return train_ds, valid_ds, test_ds                                         # trace_info : t_17065


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
