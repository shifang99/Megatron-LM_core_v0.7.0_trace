# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import argparse
import dataclasses
import json
import os
import torch
import types

import torch.nn.functional as F
from megatron.core.models.retro.utils import (
    get_config_path as get_retro_config_path,
    get_gpt_data_dir as get_retro_data_dir,
)
from megatron.core.transformer import TransformerConfig


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',      # trace_info : t_7, t_9
                                     allow_abbrev=False)                       # trace_info : t_8

    # Standard arguments.
    parser = _add_network_size_args(parser)                                    # trace_info : t_10
    parser = _add_regularization_args(parser)                                  # trace_info : t_98
    parser = _add_training_args(parser)                                        # trace_info : t_135
    parser = _add_initialization_args(parser)                                  # trace_info : t_327
    parser = _add_learning_rate_args(parser)                                   # trace_info : t_342
    parser = _add_checkpointing_args(parser)                                   # trace_info : t_388
    parser = _add_mixed_precision_args(parser)                                 # trace_info : t_453
    parser = _add_distributed_args(parser)                                     # trace_info : t_493
    parser = _add_validation_args(parser)                                      # trace_info : t_567
    parser = _add_data_args(parser)                                            # trace_info : t_580
    parser = _add_autoresume_args(parser)                                      # trace_info : t_668
    parser = _add_biencoder_args(parser)                                       # trace_info : t_677
    parser = _add_vision_args(parser)                                          # trace_info : t_725
    parser = _add_moe_args(parser)                                             # trace_info : t_811
    parser = _add_logging_args(parser)                                         # trace_info : t_863
    parser = _add_straggler_detector_args(parser)                              # trace_info : t_945
    parser = _add_inference_args(parser)                                       # trace_info : t_960
    parser = _add_transformer_engine_args(parser)                              # trace_info : t_978
    parser = _add_retro_args(parser)                                           # trace_info : t_1011
    parser = _add_experimental_args(parser)                                    # trace_info : t_1077

    # Custom arguments.
    if extra_args_provider is not None:                                        # trace_info : t_1086
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:                                                    # trace_info : t_1087
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()                                             # trace_info : t_1088

    # Experimental yaml
    if args.yaml_cfg is not None:                                              # trace_info : t_1089
        from .yaml_arguments import load_yaml
        assert args.yaml_cfg and args.use_mcore_models, "To use yaml, mcore must be enabled"
        args = load_yaml(args.yaml_cfg)


    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))                                    # trace_info : t_1090
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))                        # trace_info : t_1091

    return args                                                                # trace_info : t_1092


def load_retro_config(retro_project_dir):
    '''Load Retro's config.json.'''

    # Retro config path.
    retro_config_path = get_retro_config_path(retro_project_dir)
    assert os.path.exists(retro_config_path), \
        "Retro project dir missing config.json."

    # Load retro config.
    with open(retro_config_path) as f:
        retro_config = types.SimpleNamespace(**json.load(f))

    return retro_config


def load_retro_args(args):
    """Load predefined args from Retro config (if applicable).

    When using Retro (or GPT for comparison purposes), data arguments are
    overridden by the saved config.json within the Retro project directory. This
    is to ensure that the data used for pretraining is consistent with the data
    that was preprocessed using the Retro preprocessing pipeline (see
    `tools/retro/preprocess_data.py`).
    """

    # Return if no project directory is specified.
    if args.retro_project_dir is None:                                         # trace_info : t_1097
        return                                                                 # trace_info : t_1098

    # Load retro config.
    retro_config = load_retro_config(args.retro_project_dir)

    # Retro data path is relative to project dir (via hard or soft links).
    data_dir = get_retro_data_dir(args.retro_project_dir)
    data_path = list(retro_config.retro_gpt_data_path)
    if len(data_path) % 2 == 0:
        for i in range(len(data_path) - 1, -1, -2):
            data_path[i] = os.path.join(data_dir, data_path[i])
    else:
        assert len(data_path) == 1
        data_path[0] = os.path.join(data_dir, data_path[0])

    # Update args.
    args.data_cache_path = retro_config.retro_gpt_data_cache_path
    args.data_path = data_path if args.data_path is None else args.data_path
    args.eval_interval = retro_config.retro_gpt_eval_interval
    args.eval_iters = retro_config.retro_gpt_eval_iters
    args.global_batch_size = retro_config.retro_gpt_global_batch_size
    args.max_position_embeddings = retro_config.retro_gpt_seq_length
    args.merge_file = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_merge_file,
    ) if retro_config.retro_gpt_merge_file is not None else None
    args.seed = retro_config.retro_gpt_seed
    args.seq_length = retro_config.retro_gpt_seq_length
    args.tokenizer_model = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_tokenizer_model,
    ) if retro_config.retro_gpt_tokenizer_model is not None else None
    args.tokenizer_type = retro_config.retro_gpt_tokenizer_type
    args.train_samples = retro_config.retro_gpt_train_samples
    args.vocab_file = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_vocab_file,
    ) if retro_config.retro_gpt_vocab_file is not None else None

    # Retro-specific args.
    args.retro_block_size = retro_config.retro_block_size
    args.retro_chunk_length = retro_config.retro_gpt_chunk_length
    args.retro_neighbor_dirs = retro_config.retro_neighbor_dirs
    args.retro_split_preprocessing = retro_config.retro_gpt_split
    args.retro_bert_tokenizer_type = retro_config.retro_bert_tokenizer_type
    args.retro_bert_vocab_file = retro_config.retro_bert_vocab_file


def validate_args(args, defaults={}):

    # Load saved args from Retro (if applicable).
    load_retro_args(args)                                                      # trace_info : t_1096

    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(                                     # trace_info : t_1099, t_1101
        args.tensor_model_parallel_size, args.world_size)                      # trace_info : t_1100
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\# trace_info : t_1102
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)

    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(                                   # trace_info : t_1103, t_1106
        args.pipeline_model_parallel_size,                                     # trace_info : t_1104
        (args.world_size // args.tensor_model_parallel_size))                  # trace_info : t_1105
    args.transformer_pipeline_model_parallel_size = (                          # trace_info : t_1110
        args.pipeline_model_parallel_size - 1                                  # trace_info : t_1108
        if args.standalone_embedding_stage else                                # trace_info : t_1107
        args.pipeline_model_parallel_size                                      # trace_info : t_1109
    )

    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \                # trace_info : t_1111, t_1113
                          args.tensor_model_parallel_size                      # trace_info : t_1112
    assert args.world_size % (model_parallel_size * args.context_parallel_size) == 0, \# trace_info : t_1114
        'world size ({}) is not divisible by tensor parallel size ({}) times ' \
        'pipeline parallel size ({}) times context parallel size ({})'.format(
        args.world_size, args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size, args.context_parallel_size)
    args.data_parallel_size = args.world_size // (model_parallel_size * args.context_parallel_size)# trace_info : t_1115
    if args.rank == 0:                                                         # trace_info : t_1116
        print('using world size: {}, data-parallel size: {}, '                 # trace_info : t_1117, t_1125
              'context-parallel size: {} '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(                      # trace_info : t_1118, t_1123
                  args.world_size, args.data_parallel_size,                    # trace_info : t_1119
                  args.context_parallel_size,                                  # trace_info : t_1120
                  args.tensor_model_parallel_size,                             # trace_info : t_1121
                  args.pipeline_model_parallel_size), flush=True)              # trace_info : t_1122, t_1124
    if args.pipeline_model_parallel_size > 1:                                  # trace_info : t_1126
        if args.pipeline_model_parallel_split_rank is not None:
            assert args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size)

    if args.tp_comm_overlap:                                                   # trace_info : t_1127
        assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \    # trace_info : t_1128
        'valid, use --micro-batch-size instead'
    del args.batch_size                                                        # trace_info : t_1129
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \ # trace_info : t_1130
        '--lr-warmup-fraction instead'
    del args.warmup                                                            # trace_info : t_1131
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \  # trace_info : t_1132
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size                                               # trace_info : t_1133

    if args.checkpoint_activations:                                            # trace_info : t_1134
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations                                            # trace_info : t_1135

    if args.recompute_activations:                                             # trace_info : t_1136
        args.recompute_granularity = 'selective'
    del args.recompute_activations                                             # trace_info : t_1137

    # Set input defaults.
    for key in defaults:                                                       # trace_info : t_1138, t_1141
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:                               # trace_info : t_1139
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])                                  # trace_info : t_1140

    # Batch size.
    assert args.micro_batch_size is not None                                   # trace_info : t_1142
    assert args.micro_batch_size > 0                                           # trace_info : t_1143
    if args.global_batch_size is None:                                         # trace_info : t_1144
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0                                          # trace_info : t_1145
    if args.num_layers_per_virtual_pipeline_stage is not None:                 # trace_info : t_1146
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
            'number of layers should be divisible by the pipeline parallel size'
        num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
        assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None                       # trace_info : t_1147
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False                                          # trace_info : t_1148
        if args.rank == 0:                                                     # trace_info : t_1149
            print('WARNING: Setting args.overlap_p2p_comm to False since non-interleaved '# trace_info : t_1150
                  'schedule does not support overlapping p2p communication')

    if args.overlap_param_gather:                                              # trace_info : t_1151
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            '--overlap-grad-reduce should be turned on when using --overlap-param-gather'
        assert args.use_mcore_models, \
            '--overlap-param-gather only supported with MCore models'

    # Parameters dtype.
    args.params_dtype = torch.float                                            # trace_info : t_1152
    if args.fp16:                                                              # trace_info : t_1153
        assert not args.bf16                                                   # trace_info : t_1154
        args.params_dtype = torch.half                                         # trace_info : t_1155
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:                                                # trace_info : t_1156
            args.check_for_nan_in_loss_and_grad = False                        # trace_info : t_1157
            if args.rank == 0:                                                 # trace_info : t_1158
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '# trace_info : t_1159
                      'dynamic loss scaling is being used')
    if args.bf16:                                                              # trace_info : t_1160
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:                                                         # trace_info : t_1161
        print('using {} for parameters ...'.format(args.params_dtype),         # trace_info : t_1162, t_1164
              flush=True)                                                      # trace_info : t_1163

    if args.dataloader_type is None:                                           # trace_info : t_1165
        args.dataloader_type = 'single'                                        # trace_info : t_1166

    # data
    assert args.num_dataset_builder_threads > 0                                # trace_info : t_1167

    # Consumed tokens.
    args.consumed_train_samples = 0                                            # trace_info : t_1168
    args.consumed_valid_samples = 0                                            # trace_info : t_1169

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False                                          # trace_info : t_1170

    # Iteration-based training.
    if args.train_iters:                                                       # trace_info : t_1171
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \                                   # trace_info : t_1172
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \                                # trace_info : t_1173
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \                                  # trace_info : t_1174
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \                               # trace_info : t_1175
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:                                # trace_info : t_1176
            assert args.lr_warmup_iters == 0, \                                # trace_info : t_1177
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:                                                     # trace_info : t_1178
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is not None:                                            # trace_info : t_1179
        assert args.encoder_num_layers is None, \                              # trace_info : t_1180
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers                              # trace_info : t_1181
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',       # trace_info : t_1182
                     'max_position_embeddings']
    for req_arg in required_args:                                              # trace_info : t_1183, t_1186, t_1189, t_1192, t_1195
        _check_arg_is_not_none(args, req_arg)                                  # trace_info : t_1184, t_1187, t_1190, t_1193

    # Checks.
    if args.ffn_hidden_size is None:                                           # trace_info : t_1196
        if args.swiglu:                                                        # trace_info : t_1197
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size                        # trace_info : t_1198

    if args.kv_channels is None:                                               # trace_info : t_1199
        assert args.hidden_size % args.num_attention_heads == 0                # trace_info : t_1200
        args.kv_channels = args.hidden_size // args.num_attention_heads        # trace_info : t_1201

    if args.seq_length is not None:                                            # trace_info : t_1202
        assert args.encoder_seq_length is None                                 # trace_info : t_1203
        args.encoder_seq_length = args.seq_length                              # trace_info : t_1204
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:                                            # trace_info : t_1205
        assert args.max_position_embeddings >= args.seq_length                 # trace_info : t_1206
    if args.decoder_seq_length is not None:                                    # trace_info : t_1207
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:                                                    # trace_info : t_1208
        assert args.min_lr <= args.lr                                          # trace_info : t_1209
    if args.save is not None:                                                  # trace_info : t_1210
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:                                             # trace_info : t_1211
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:                                          # trace_info : t_1212
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.moe_grouped_gemm:                                                  # trace_info : t_1213
        assert args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
        dc = torch.cuda.get_device_capability()
        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."

    if args.weight_decay_incr_style == 'constant':                             # trace_info : t_1214
        assert args.start_weight_decay is None                                 # trace_info : t_1215
        assert args.end_weight_decay is None                                   # trace_info : t_1216
        args.start_weight_decay = args.weight_decay                            # trace_info : t_1217
        args.end_weight_decay = args.weight_decay                              # trace_info : t_1218
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])                         # trace_info : t_1219
    TORCH_MINOR = int(torch.__version__.split('.')[1])                         # trace_info : t_1220
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):             # trace_info : t_1221
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:                                      # trace_info : t_1222
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert (TORCH_MAJOR, TORCH_MINOR) >= (1, 10), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    if args.recompute_granularity == 'selective':                              # trace_info : t_1223
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:                                   # trace_info : t_1224
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:                                                 # trace_info : t_1225
        args.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":                   # trace_info : t_1226
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:                                               # trace_info : t_1227
        args.bias_gelu_fusion = False

    # Retro checks.
    if args.retro_add_retriever:                                               # trace_info : t_1228

        # Train samples should be auto-loaded.
        assert args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config."

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:     # trace_info : t_1229
        assert args.use_mcore_models, \
            '--decoupled-lr and --decoupled-min-lr only supported by Megatron Core, please add --use-mcore-models.'

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:                                    # trace_info : t_1230
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:                     # trace_info : t_1231
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and not args.use_mcore_models:                  # trace_info : t_1232
        raise RuntimeError('--rotary-interleaved only support Megatron Core, please add --use-mcore-models.')

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':# trace_info : t_1233
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # MoE Spec check
    if args.num_experts is not None:                                           # trace_info : t_1234
        assert args.spec is None, "Model Spec must be None when using MoEs"

    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:                                   # trace_info : t_1235
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.fp16, \
            "Expert parallelism is not supported with fp16 training."

    # Distributed checkpointing checks
    if args.use_dist_ckpt and not args.use_mcore_models:                       # trace_info : t_1236
        raise RuntimeError('--use-dist-ckpt only support Megatron Core, please add --use-mcore-models.')

    # Data blend checks
    assert args.mock_data + \                                                  # trace_info : t_1237, t_1239, t_1241, t_1243
           bool(args.data_path) + \                                            # trace_info : t_1238
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) \# trace_info : t_1240
           == 1, "A single data source must be provided"                       # trace_info : t_1242

    if args.use_tp_pp_dp_mapping:                                              # trace_info : t_1244
        assert args.context_parallel_size * args.expert_model_parallel_size <= 1, \
            "context_parallel and expert_model_parallel can't be used with tp-pp-dp mapping."

    # Print arguments.
    _print_args("arguments", args)                                             # trace_info : t_1245

    return args                                                                # trace_info : t_3098


def _print_args(title, args):
    """Print arguments."""
    if args.rank == 0:                                                         # trace_info : t_1246
        print(f'------------------------ {title} ------------------------',    # trace_info : t_1247, t_1249
              flush=True)                                                      # trace_info : t_1248
        str_list = []                                                          # trace_info : t_1250
        for arg in vars(args):                                                 # trace_info : t_1251, t_1254, t_1257, t_1260, t_1263, ...
            dots = '.' * (48 - len(arg))                                       # trace_info : t_1252, t_1255, t_1258, t_1261, t_1264, ...
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))# trace_info : t_1253, t_1256, t_1259, t_1262, t_1265, ...
        for arg in sorted(str_list, key=lambda x: x.lower()):                  # trace_info : t_2173, t_2174, t_2175, t_2176, t_2177, ...
            print(arg, flush=True)                                             # trace_info : t_2481, t_2483, t_2485, t_2487, t_2489, ...
        print(f'-------------------- end of {title} ---------------------',    # trace_info : t_3095, t_3097
              flush=True)                                                      # trace_info : t_3096


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)   # trace_info : t_1185, t_1188, t_1191, t_1194


def core_transformer_config_from_args(args, config_class=None):

    # Config class.
    config_class = config_class or TransformerConfig                           # trace_info : t_10709

    # Translate args to core transformer configuration
    kw_args = {}                                                               # trace_info : t_10710
    for f in dataclasses.fields(config_class):                                 # trace_info : t_10711, t_10714, t_10717, t_10720, t_10723, ...
        if hasattr(args, f.name):                                              # trace_info : t_10712, t_10715, t_10718, t_10721, t_10724, ...
            kw_args[f.name] = getattr(args, f.name)                            # trace_info : t_10713, t_10716, t_10719, t_10722, t_10725, ...
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm             # trace_info : t_11015
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p         # trace_info : t_11016
    kw_args['layernorm_epsilon'] = args.norm_epsilon                           # trace_info : t_11017
    kw_args['deallocate_pipeline_outputs'] = True                              # trace_info : t_11018
    kw_args['pipeline_dtype'] = args.params_dtype                              # trace_info : t_11019
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm                      # trace_info : t_11020
    kw_args['num_moe_experts'] = args.num_experts                              # trace_info : t_11021
    kw_args['rotary_interleaved'] = args.rotary_interleaved                    # trace_info : t_11022
    if args.swiglu:                                                            # trace_info : t_11023
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion              # trace_info : t_11024
    if args.squared_relu:                                                      # trace_info : t_11025
        assert not args.swiglu
        try:
            jit_fuser = torch.compile
        except:
            jit_fuser = torch.jit.script
        @jit_fuser
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        kw_args['activation_func'] = squared_relu
    if args.init_method_xavier_uniform:                                        # trace_info : t_11026
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:                                             # trace_info : t_11027
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None                                     # trace_info : t_11028

    # Return config.
    return config_class(**kw_args)                                             # trace_info : t_11029


def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')              # trace_info : t_979

    group.add_argument('--fp8-format', default=None,                           # trace_info : t_980, t_984
                       choices=['e4m3', 'hybrid'],                             # trace_info : t_981
                       help='Which fp8 format scheme to use for FP8 tensors in the forward and backward pass',# trace_info : t_982
                       dest='fp8')                                             # trace_info : t_983
    group.add_argument('--fp8-margin', type=int, default=0,                    # trace_info : t_985, t_988
                       help='Scaling margin for fp8',                          # trace_info : t_986
                       dest='fp8_margin')                                      # trace_info : t_987
    group.add_argument('--fp8-interval', type=int, default=1,                  # trace_info : t_989, t_992
                       help='Scaling update interval for fp8',                 # trace_info : t_990
                       dest='fp8_interval')                                    # trace_info : t_991
    group.add_argument('--fp8-amax-history-len', type=int, default=1,          # trace_info : t_993, t_996
                       help='Number of steps for which amax history is recorded per tensor',# trace_info : t_994
                       dest='fp8_amax_history_len')                            # trace_info : t_995
    group.add_argument('--fp8-amax-compute-algo', default='most_recent',       # trace_info : t_997, t_1001
                       choices=['most_recent', 'max'],                         # trace_info : t_998
                       help='Algorithm for computing amax from history',       # trace_info : t_999
                       dest='fp8_amax_compute_algo')                           # trace_info : t_1000
    group.add_argument('--no-fp8-wgrad', action='store_false',                 # trace_info : t_1002, t_1005
                       help='Execute wgrad in higher precision even for FP8 runs',# trace_info : t_1003
                       dest='fp8_wgrad')                                       # trace_info : t_1004
    group.add_argument('--transformer-impl', default='transformer_engine',     # trace_info : t_1006, t_1009
                       choices=['local', 'transformer_engine'],                # trace_info : t_1007
                       help='Which Transformer implementation to use.')        # trace_info : t_1008

    return parser                                                              # trace_info : t_1010

def _add_inference_args(parser):
    group = parser.add_argument_group(title='inference')                       # trace_info : t_961

    group.add_argument('--inference-batch-times-seqlen-threshold',             # trace_info : t_962, t_965
                       type=int, default=512,                                  # trace_info : t_963
                       help='During inference, if batch-size times '           # trace_info : t_964
                       'sequence-length is smaller than this threshold '
                       'then we will not use pipelining, otherwise we will.')
    group.add_argument('--max-tokens-to-oom',                                  # trace_info : t_966, t_969
                       type=int, default=12000,                                # trace_info : t_967
                       help='Maximum number of tokens during inference'        # trace_info : t_968
                       'tokens here is # in prompt + # to generate'
                       'Allows us to throw an error before OOM crashes server')
    group.add_argument('--output-bert-embeddings', action='store_true',        # trace_info : t_970, t_972
                       help='Output Bert embeddings (via mean pooling) from '  # trace_info : t_971
                       'model, rather than its binary head output or entire '
                       'hidden batch.')
    group.add_argument('--bert-embedder-type', default="megatron",             # trace_info : t_973, t_976
                       choices=["megatron", "huggingface"],                    # trace_info : t_974
                       help='Select either Megatron or Huggingface as the '    # trace_info : t_975
                       'Bert embedder.')

    return parser                                                              # trace_info : t_977


def _add_retro_args(parser):
    group = parser.add_argument_group(title='retro')                           # trace_info : t_1012

    group.add_argument('--retro-project-dir', default=None,                    # trace_info : t_1013, t_1015
                       help='Retro project directory, which contains the '     # trace_info : t_1014
                       'preprocessed data for pretraining. This directory '
                       'is built during preprocessing (see '
                       'tools/retro/README.md), and contains subdirectories '
                       'for the chunk database and pretraining neighbors.')
    group.add_argument('--retro-add-retriever',                                # trace_info : t_1016, t_1019
                       action='store_true', default=False,                     # trace_info : t_1017
                       help='Add a retriever to the transformer, for use in '  # trace_info : t_1018
                       'pretraining a Retro model.')
    group.add_argument('--retro-cyclic-train-iters', type=int, default=None,   # trace_info : t_1020, t_1022
                       help='Set number of training iterations for cyclic '    # trace_info : t_1021
                       'Retro training.')
    group.add_argument('--retro-encoder-layers', type=int, default=2,          # trace_info : t_1023, t_1025
                       help='Number of layers to use for the retrieval '       # trace_info : t_1024
                       'encoder.')
    group.add_argument('--retro-encoder-hidden-dropout',                       # trace_info : t_1026, t_1028
                       type=float, default=0.1, help='Hidden dropout for '     # trace_info : t_1027
                       'retrieval encoder.')
    group.add_argument('--retro-encoder-attention-dropout',                    # trace_info : t_1029, t_1031
                       type=float, default=0.1, help='Attention dropout for '  # trace_info : t_1030
                       'retrieval encoder.')
    group.add_argument("--retro-num-neighbors", type=int, default=2,           # trace_info : t_1032, t_1034
                       help='Number of neighbors to retrieve during '          # trace_info : t_1033
                       'pretraining.')
    group.add_argument("--retro-num-retrieved-chunks", type=int, default=2,    # trace_info : t_1035, t_1037
                       help='Number of chunks to retrieve from the retrieval ' # trace_info : t_1036
                       'database.')
    group.add_argument("--retro-attention-gate", type=float, default=1,        # trace_info : t_1038, t_1040
                       help="Gated cross attention.")                          # trace_info : t_1039
    group.add_argument("--retro-no-verify-neighbor-count", action="store_false",# trace_info : t_1041, t_1044
                       dest="retro_verify_neighbor_count",                     # trace_info : t_1042
                       help="Skip verifying that len(GPT dataset) == len(saved "# trace_info : t_1043
                       "neighbors).")

    # Enforce argument naming convention.
    for action in group._group_actions:                                        # trace_info : t_1045, t_1048, t_1051, t_1054, t_1057, ...
        prefix = action.dest.split("_")[0]                                     # trace_info : t_1046, t_1049, t_1052, t_1055, t_1058, ...
        assert prefix == "retro", \                                            # trace_info : t_1047, t_1050, t_1053, t_1056, t_1059, ...
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser                                                              # trace_info : t_1076


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')                    # trace_info : t_11

    group.add_argument('--num-layers', type=int, default=None,                 # trace_info : t_12, t_14
                       help='Number of transformer layers.')                   # trace_info : t_13
    group.add_argument('--encoder-num-layers', type=int, default=None,         # trace_info : t_15, t_17
                       help='Number of encoder transformer layers.')           # trace_info : t_16
    group.add_argument('--decoder-num-layers', type=int, default=None,         # trace_info : t_18, t_20
                       help='Number of decoder transformer layers.')           # trace_info : t_19
    group.add_argument('--hidden-size', type=int, default=None,                # trace_info : t_21, t_23
                       help='Tansformer hidden size.')                         # trace_info : t_22
    group.add_argument('--ffn-hidden-size', type=int, default=None,            # trace_info : t_24, t_26
                       help='Transformer Feed-Forward Network hidden size. '   # trace_info : t_25
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--num-attention-heads', type=int, default=None,        # trace_info : t_27, t_29
                       help='Number of transformer attention heads.')          # trace_info : t_28
    group.add_argument('--kv-channels', type=int, default=None,                # trace_info : t_30, t_32
                       help='Projection weights dimension in multi-head '      # trace_info : t_31
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--group-query-attention', action='store_true',         # trace_info : t_33, t_35
                          help='Use group-query attention.')                   # trace_info : t_34
    group.add_argument('--num-query-groups', type=int, default=1)              # trace_info : t_36

    group.add_argument('--max-position-embeddings', type=int, default=None,    # trace_info : t_37, t_39
                       help='Maximum number of position embeddings to use. '   # trace_info : t_38
                       'This is the size of position embedding.')
    group.add_argument('--position-embedding-type', type=str, default='learned_absolute',# trace_info : t_40, t_43
                       choices=['learned_absolute', 'rope'],                   # trace_info : t_41
                       help='Position embedding type.')                        # trace_info : t_42
    group.add_argument('--use-rotary-position-embeddings', action='store_true',# trace_info : t_44, t_46
                       help='Use rotary positional embeddings or not. '        # trace_info : t_45
                       'Deprecated: use --position-embedding-type')
    group.add_argument('--rotary-percent', type=float, default=1.0,            # trace_info : t_47, t_49
                       help='Percent of rotary dimension to use, default 100%%')# trace_info : t_48
    group.add_argument('--rotary-interleaved', action='store_true',            # trace_info : t_50, t_52
                          help='Use interleaved rotary embedding.')            # trace_info : t_51
    group.add_argument('--rotary-seq-len-interpolation-factor', type=int, default=None,# trace_info : t_53, t_55
                       help='Sequence length interpolation factor for rotary embeddings.')# trace_info : t_54
    group.add_argument('--no-position-embedding',                              # trace_info : t_56, t_60
                       action='store_false',                                   # trace_info : t_57
                       help='Disable position embedding. Deprecated: use --position-embedding-type',# trace_info : t_58
                       dest='add_position_embedding')                          # trace_info : t_59
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,# trace_info : t_61, t_63
                       help='Pad the vocab size to be divisible by this value.'# trace_info : t_62
                       'This is added for computational efficieny reasons.')
    group.add_argument('--normalization', default='LayerNorm',                 # trace_info : t_64, t_67
                       choices=['LayerNorm', 'RMSNorm'],                       # trace_info : t_65
                       help='Which normalization technique to use.')           # trace_info : t_66
    group.add_argument('--norm-epsilon', type=float, default=1e-5,             # trace_info : t_68, t_70
                       help='Epsilon for layer norm and RMS norm.')            # trace_info : t_69
    group.add_argument('--apply-layernorm-1p', action='store_true',            # trace_info : t_71, t_73
                       help='Adjust LayerNorm weights such that they are centered '# trace_info : t_72
                       'around zero. This improves numerical stability.')
    group.add_argument('--apply-residual-connection-post-layernorm',           # trace_info : t_74, t_77
                       action='store_true',                                    # trace_info : t_75
                       help='If set, use original BERT residula connection '   # trace_info : t_76
                       'ordering.')
    group.add_argument('--openai-gelu', action='store_true',                   # trace_info : t_78, t_80
                       help='Use OpenAIs GeLU implementation. This option'     # trace_info : t_79
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--squared-relu', action='store_true',                  # trace_info : t_81, t_83
                       help='Use squared relu activation instead of default gelu')# trace_info : t_82
    group.add_argument('--swiglu', action='store_true',                        # trace_info : t_84, t_86
                       help='Use gated linear units and SiLU activation instead of default gelu')# trace_info : t_85
    group.add_argument('--onnx-safe', type=bool, required=False,               # trace_info : t_87, t_89
                       help='Use workarounds for known problems with '         # trace_info : t_88
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',          # trace_info : t_90, t_93
                       help='Disable BERT binary head.',                       # trace_info : t_91
                       dest='bert_binary_head')                                # trace_info : t_92
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',# trace_info : t_94, t_96
                       help='Untie embeddings and output weights.'),           # trace_info : t_95
    return parser                                                              # trace_info : t_97

def _add_straggler_detector_args(parser):
    group = parser.add_argument_group(title='straggler')                       # trace_info : t_946
    group.add_argument('--log-straggler', action='store_true',                 # trace_info : t_947, t_949
                       help='If set, tracks and logs straggler per GPU.')      # trace_info : t_948
    group.add_argument('--disable-straggler-on-startup', action='store_true',  # trace_info : t_950, t_952
                       help='If set, StragglerDetector is disabled on startup.')# trace_info : t_951
    group.add_argument('--straggler-ctrlr-port', type=int, default=65535,      # trace_info : t_953, t_955
                       help='Port number to toggle StragglerDetector on/off at runtime')# trace_info : t_954
    group.add_argument('--straggler-minmax-count', type=int, default=1,        # trace_info : t_956, t_958
                       help='Number of ranks to report with high/low estimated throughput')# trace_info : t_957
    return parser                                                              # trace_info : t_959

def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')                         # trace_info : t_864

    group.add_argument('--log-params-norm', action='store_true',               # trace_info : t_865, t_867
                       help='If set, calculate and log parameters norm.')      # trace_info : t_866
    group.add_argument('--log-num-zeros-in-grad', action='store_true',         # trace_info : t_868, t_870
                       help='If set, calculate and log the number of zeros in gradient.')# trace_info : t_869
    group.add_argument('--log-throughput', action='store_true',                # trace_info : t_871, t_873
                       help='If set, calculate and log throughput per GPU.')   # trace_info : t_872
    group.add_argument('--log-progress', action='store_true',                  # trace_info : t_874, t_876
                       help='If set, log progress (in terms of number of processed tokens and '# trace_info : t_875
                       'number of floating-point operations) to progress.txt file in checkpoint '
                       'directory.')
    group.add_argument('--timing-log-level', type=int,                         # trace_info : t_877, t_880
                       default=0, choices=range(0,3),                          # trace_info : t_878
                       help='Granularity level to measure and report timing. ' # trace_info : t_879
                       '   0: report only iteration time and make sure timing '
                       '      does not introduce extra overhead.'
                       '   1: report timing for operations that are executed '
                       '      very limited times (basically once) during '
                       '      each iteration (such as gradient all-reduce) '
                       '   2: report timing for operations that migh be '
                       '      executed numerous times during each iteration. '
                       'Note that setting the level to 1 or 2 might '
                       'cause increase in iteration time.')
    group.add_argument('--no-barrier-with-level-1-timing', action='store_false',# trace_info : t_881, t_884
                       help='If not set, use barrier with level 1 time '       # trace_info : t_882
                       'measurements. Note that this is up to the user '
                       'to make sure calling barrier with their timers '
                       'will not result in hangs. This can happen if for '
                       'example the user adds a level 1 timer that is not '
                       'called by all ranks.',
                       dest='barrier_with_L1_time')                            # trace_info : t_883
    group.add_argument('--timing-log-option', type=str, default='minmax',      # trace_info : t_885, t_888
                       choices=['max', 'minmax', 'all'],                       # trace_info : t_886
                       help='Options for logging timing:'                      # trace_info : t_887
                       '  max: report the max timing across all ranks'
                       '  minmax: report min and max timings across all ranks'
                       '  all: report timings of all ranks.')
    group.add_argument('--tensorboard-log-interval', type=int, default=1,      # trace_info : t_889, t_891
                       help='Report to tensorboard interval.')                 # trace_info : t_890
    group.add_argument('--tensorboard-queue-size', type=int, default=1000,     # trace_info : t_892, t_894
                       help='Size of the tensorboard queue for pending events '# trace_info : t_893
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',     # trace_info : t_895, t_897
                       help='If set, write timers to tensorboard.')            # trace_info : t_896
    group.add_argument('--log-batch-size-to-tensorboard', action='store_true', # trace_info : t_898, t_900
                       help='If set, write batch-size to tensorboard.')        # trace_info : t_899
    group.add_argument('--no-log-learnig-rate-to-tensorboard',                 # trace_info : t_901, t_905
                       action='store_false',                                   # trace_info : t_902
                       help='Disable learning rate logging to tensorboard.',   # trace_info : t_903
                       dest='log_learning_rate_to_tensorboard')                # trace_info : t_904
    group.add_argument('--no-log-loss-scale-to-tensorboard',                   # trace_info : t_906, t_910
                       action='store_false',                                   # trace_info : t_907
                       help='Disable loss-scale logging to tensorboard.',      # trace_info : t_908
                       dest='log_loss_scale_to_tensorboard')                   # trace_info : t_909
    group.add_argument('--log-validation-ppl-to-tensorboard',                  # trace_info : t_911, t_914
                       action='store_true',                                    # trace_info : t_912
                       help='If set, write validation perplexity to '          # trace_info : t_913
                       'tensorboard.')
    group.add_argument('--log-memory-to-tensorboard',                          # trace_info : t_915, t_918
                       action='store_true',                                    # trace_info : t_916
                       help='Enable memory logging to tensorboard.')           # trace_info : t_917
    group.add_argument('--log-world-size-to-tensorboard',                      # trace_info : t_919, t_922
                       action='store_true',                                    # trace_info : t_920
                       help='Enable world size logging to tensorboard.')       # trace_info : t_921
    group.add_argument('--wandb-project', type=str, default='',                # trace_info : t_923, t_925
                       help='The wandb project name. Ignore wandb by default.')# trace_info : t_924
    group.add_argument('--wandb-exp-name', type=str, default='',               # trace_info : t_926, t_928
                       help='The wandb experiment name.')                      # trace_info : t_927
    group.add_argument('--wandb-save-dir', type=str, default='',               # trace_info : t_929, t_931
                       help='Path to save the wandb results locally.')         # trace_info : t_930
    group.add_argument('--enable-one-logger', action='store_true',             # trace_info : t_932, t_934
                       help='If set, use one_logger to track E2E metrics'      # trace_info : t_933
                       'Note that one_logger is an internal tool and not available externally. '
                       'For installation, please try command: `pip install '
                       '--index-url=https://sc-hw-artf.nvidia.com/api/pypi/hwinf-ml-pypi/simple'
                       ' one_logger` or go to https://gitlab-master.nvidia.com/hwinf-dcm/onelogger '
                       'for more details')
    group.add_argument('--one-logger-project', type=str, default='e2e-tracking',# trace_info : t_935, t_937
                       help='The one-logger project name. Will ignore if '     # trace_info : t_936
                       '--enable-one-logger is not set')
    group.add_argument('--one-logger-entity', type=str, default='hwinf_dcm',   # trace_info : t_938, t_940
                       help='The one-logger username or team name. Will ignore if '# trace_info : t_939
                       '--enable-one-logger is not set')
    group.add_argument('--one-logger-run-name', type=str, default=None,        # trace_info : t_941, t_943
                       help='The one-logger run name displayed. Will ignore if '# trace_info : t_942
                       '--enable-one-logger is not set')
    return parser                                                              # trace_info : t_944


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')                  # trace_info : t_99

    group.add_argument('--attention-dropout', type=float, default=0.1,         # trace_info : t_100, t_102
                       help='Post attention dropout probability.')             # trace_info : t_101
    group.add_argument('--hidden-dropout', type=float, default=0.1,            # trace_info : t_103, t_105
                       help='Dropout probability for hidden state transformer.')# trace_info : t_104
    group.add_argument('--weight-decay', type=float, default=0.01,             # trace_info : t_106, t_108
                       help='Weight decay coefficient for L2 regularization.') # trace_info : t_107
    group.add_argument('--start-weight-decay', type=float,                     # trace_info : t_109, t_111
                       help='Initial weight decay coefficient for L2 regularization.')# trace_info : t_110
    group.add_argument('--end-weight-decay', type=float,                       # trace_info : t_112, t_114
                       help='End of run weight decay coefficient for L2 regularization.')# trace_info : t_113
    group.add_argument('--weight-decay-incr-style', type=str, default='constant',# trace_info : t_115, t_118
                       choices=['constant', 'linear', 'cosine'],               # trace_info : t_116
                       help='Weight decay increment function.')                # trace_info : t_117
    group.add_argument('--clip-grad', type=float, default=1.0,                 # trace_info : t_119, t_121
                       help='Gradient clipping based on global L2 norm.')      # trace_info : t_120
    group.add_argument('--adam-beta1', type=float, default=0.9,                # trace_info : t_122, t_124
                       help='First coefficient for computing running averages '# trace_info : t_123
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,              # trace_info : t_125, t_127
                       help='Second coefficient for computing running averages '# trace_info : t_126
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,                # trace_info : t_128, t_130
                       help='Term added to the denominator to improve'         # trace_info : t_129
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,              # trace_info : t_131, t_133
                       help='Momentum factor for sgd')                         # trace_info : t_132
    return parser                                                              # trace_info : t_134


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')                        # trace_info : t_136

    group.add_argument('--micro-batch-size', type=int, default=None,           # trace_info : t_137, t_139
                       help='Batch size per model instance (local batch size). '# trace_info : t_138
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,                 # trace_info : t_140, t_142
                       help='Old batch size parameter, do not use. '           # trace_info : t_141
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,          # trace_info : t_143, t_145
                       help='Training batch size. If set, it should be a '     # trace_info : t_144
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,         # trace_info : t_146, t_148
                       help='Batch size ramp up with the following values:'    # trace_info : t_147
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--recompute-activations', action='store_true',         # trace_info : t_149, t_151
                       help='recompute activation to allow for training '      # trace_info : t_150
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,      # trace_info : t_152, t_155
                       choices=['full', 'selective'],                          # trace_info : t_153
                       help='Checkpoint activations to allow for training '    # trace_info : t_154
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    group.add_argument('--no-check-for-nan-in-loss-and-grad', action='store_false',# trace_info : t_156, t_159
                       help='Check for NaNs in loss and grad',                 # trace_info : t_157
                       dest='check_for_nan_in_loss_and_grad')                  # trace_info : t_158
    group.add_argument('--distribute-saved-activations',                       # trace_info : t_160, t_163
                       action='store_true',                                    # trace_info : t_161
                       help='If set, distribute recomputed activations '       # trace_info : t_162
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default=None,           # trace_info : t_164, t_167
                       choices=['uniform', 'block'],                           # trace_info : t_165
                       help='1) uniform: uniformly divide the total number of '# trace_info : t_166
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=None,       # trace_info : t_168, t_170
                       help='1) uniform: the number of Transformer layers in each '# trace_info : t_169
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--no-clone-scatter-output-in-embedding', action='store_false',# trace_info : t_171, t_174
                       help='If not set, clone the output of the scatter in embedding layer to GC original tensor.',# trace_info : t_172
                       dest='clone_scatter_output_in_embedding')               # trace_info : t_173
    group.add_argument('--profile', action='store_true',                       # trace_info : t_175, t_177
                       help='Enable nsys profiling. When using this option, nsys '# trace_info : t_176
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.')
    group.add_argument('--profile-step-start', type=int, default=10,           # trace_info : t_178, t_180
                       help='Global step to start profiling.')                 # trace_info : t_179
    group.add_argument('--profile-step-end', type=int, default=12,             # trace_info : t_181, t_183
                       help='Global step to stop profiling.')                  # trace_info : t_182
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],    # trace_info : t_184, t_186
                       help='Global ranks to profile.')                        # trace_info : t_185
    group.add_argument('--tp-comm-overlap', action='store_true', help='Enables the '# trace_info : t_187
                       ' overlap of Tensor parallel communication and GEMM kernels.')
    group.add_argument('--tp-comm-overlap-cfg', type=str, default=None,        # trace_info : t_188, t_190
                       help='Config file when tp_comm_overlap is enabled.')    # trace_info : t_189
    group.add_argument('--disable-tp-comm-overlap-ag', action='store_false',   # trace_info : t_191, t_194
                       help=('Disables the All-Gather overlap with GEMM by '   # trace_info : t_192
                             'pipelining the GEMM and All-Gather.'),
                       dest='tp_comm_overlap_ag')                              # trace_info : t_193
    group.add_argument('--disable-tp-comm-overlap-rs', action='store_false',   # trace_info : t_195, t_198
                       help=('Disables the Reduce-Scatter overlap with GEMM by '# trace_info : t_196
                             'pipelining the GEMM and Reduce-Scatter.'),
                       dest='tp_comm_overlap_rs')                              # trace_info : t_197
    group.add_argument('--tp-comm-overlap-rs-dgrad', action='store_true',      # trace_info : t_199, t_202
                       help = 'Enables the Reduce-Scatter overlap with dgrad GEMM.',# trace_info : t_200
                       dest='tp_comm_overlap_rs_dgrad')                        # trace_info : t_201
    group.add_argument('--disable-tp-comm-bulk-dgrad', action='store_false',   # trace_info : t_203, t_206
                       help='Disables the All-Gather overlap with bprop activation gradient GEMM.',# trace_info : t_204
                       dest='tp_comm_bulk_dgrad')                              # trace_info : t_205
    group.add_argument('--disable-tp-comm-bulk-wgrad', action='store_false',   # trace_info : t_207, t_210
                       help='Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.',# trace_info : t_208
                       dest='tp_comm_bulk_wgrad')                              # trace_info : t_209
    group.add_argument('--use-cpu-initialization', action='store_true',        # trace_info : t_211, t_214
                       default=None,                                           # trace_info : t_212
                       help='If set, initialize weights on the CPU. This eliminates init differences based on tensor parallelism.')# trace_info : t_213
    group.add_argument('--empty-unused-memory-level', default=0, type=int,     # trace_info : t_215, t_218
                       choices=[0, 1, 2],                                      # trace_info : t_216
                       help='Call torch.cuda.empty_cache() each iteration '    # trace_info : t_217
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--check-weight-hash-across-dp-replicas-interval', type=int, default=None,# trace_info : t_219, t_221
                       help='Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked.')# trace_info : t_220
    group.add_argument('--calculate-per-token-loss', action='store_true',      # trace_info : t_222, t_224
                       help=('Scale cross entropy loss by the number of non-padded tokens in the '# trace_info : t_223
                             'global batch, versus the default behavior of assuming all tokens are non-padded.'))

    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',        # trace_info : t_225, t_227
                       help='Checkpoint activation to allow for training '     # trace_info : t_226
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--train-iters', type=int, default=None,                # trace_info : t_228, t_230
                       help='Total number of iterations to train over all '    # trace_info : t_229
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,              # trace_info : t_231, t_233
                       help='Total number of samples to train over all '       # trace_info : t_232
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--log-interval', type=int, default=100,                # trace_info : t_234, t_236
                       help='Report loss and timing interval.')                # trace_info : t_235
    group.add_argument('--exit-interval', type=int, default=None,              # trace_info : t_237, t_239
                       help='Exit the program after the iteration is divisible '# trace_info : t_238
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,      # trace_info : t_240, t_242
                       help='Exit the program after this many minutes.')       # trace_info : t_241
    group.add_argument('--exit-signal-handler', action='store_true',           # trace_info : t_243, t_245
                       help='Dynamically save the checkpoint and shutdown the '# trace_info : t_244
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard-dir', type=str, default=None,            # trace_info : t_246, t_248
                       help='Write TensorBoard logs to this directory.')       # trace_info : t_247
    group.add_argument('--no-masked-softmax-fusion',                           # trace_info : t_249, t_253
                       action='store_false',                                   # trace_info : t_250
                       help='Disable fusion of query_key_value scaling, '      # trace_info : t_251
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')                           # trace_info : t_252
    group.add_argument('--no-bias-gelu-fusion', action='store_false',          # trace_info : t_254, t_257
                       help='Disable bias and gelu fusion.',                   # trace_info : t_255
                       dest='bias_gelu_fusion')                                # trace_info : t_256
    group.add_argument('--no-bias-swiglu-fusion', action='store_false',        # trace_info : t_258, t_261
                       help='Disable bias and swiglu fusion, the fusion is '   # trace_info : t_259
                       'available only when using megatron-core.',
                       dest='bias_swiglu_fusion')                              # trace_info : t_260
    group.add_argument('--no-bias-dropout-fusion', action='store_false',       # trace_info : t_262, t_265
                       help='Disable bias and dropout fusion.',                # trace_info : t_263
                       dest='bias_dropout_fusion')                             # trace_info : t_264
    group.add_argument('--no-rope-fusion', action='store_false',               # trace_info : t_266, t_269
                       help='Disable rope fusion, the fusion is available '    # trace_info : t_267
                       'only when using megatron-core.',
                       dest='apply_rope_fusion')                               # trace_info : t_268
    group.add_argument('--use-flash-attn', action='store_true',                # trace_info : t_270, t_272
                       help='use FlashAttention implementation of attention. ' # trace_info : t_271
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--disable-bias-linear', action='store_false',          # trace_info : t_273, t_276
                       help='Disable bias in the linear layers',               # trace_info : t_274
                       dest='add_bias_linear')                                 # trace_info : t_275
    group.add_argument('--add-qkv-bias', action='store_true',                  # trace_info : t_277, t_280
                       help='Enable bias only in the QKV linear layers',       # trace_info : t_278
                       dest='add_qkv_bias')                                    # trace_info : t_279
    group.add_argument('--optimizer', type=str, default='adam',                # trace_info : t_281, t_284
                       choices=['adam', 'sgd'],                                # trace_info : t_282
                       help='Optimizer function')                              # trace_info : t_283
    group.add_argument('--dataloader-type', type=str, default=None,            # trace_info : t_285, t_288
                       choices=['single', 'cyclic', 'external'],               # trace_info : t_286
                       help='Single pass vs multiple pass data loader')        # trace_info : t_287
    group.add_argument('--no-async-tensor-model-parallel-allreduce',           # trace_info : t_289, t_293
                       action='store_false',                                   # trace_info : t_290
                       help='DEPRECATED. This flag is ignored.',               # trace_info : t_291
                       dest='async_tensor_model_parallel_allreduce')           # trace_info : t_292
    group.add_argument('--no-persist-layer-norm', action='store_true',         # trace_info : t_294, t_296
                       help='Disable using persistent fused layer norm kernel. '# trace_info : t_295
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',             # trace_info : t_297, t_299
                       help='Enable sequence parallel optimization.')          # trace_info : t_298
    group.add_argument('--no-gradient-accumulation-fusion',                    # trace_info : t_300, t_304
                       action='store_false',                                   # trace_info : t_301
                       help='Disable fusing gradient accumulation to weight '  # trace_info : t_302
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')                    # trace_info : t_303
    group.add_argument('--use-mcore-models', action='store_true',              # trace_info : t_305, t_307
                       help='Use the implementation from megatron core')       # trace_info : t_306
    group.add_argument('--manual-gc', action='store_true',                     # trace_info : t_308, t_310
                       help='Disable the threshold-based default garbage '     # trace_info : t_309
                       'collector and trigger the garbage collection manually. '
                       'Manual garbage collection helps to align the timing of '
                       'the collection across ranks which mitigates the impact '
                       'of CPU-associated jitters. When the manual gc is enabled, '
                       'garbage collection is performed only at the start and the '
                       'end of the validation routine by default.')
    group.add_argument('--manual-gc-interval', type=int, default=0,            # trace_info : t_311, t_313
                       help='Training step interval to trigger manual garbage '# trace_info : t_312
                       'collection. When the value is set to 0, garbage '
                       'collection is not triggered between training steps.')
    group.add_argument('--no-manual-gc-eval', action='store_false',            # trace_info : t_314, t_317
                       help='When using manual garbage collection, disable '   # trace_info : t_315
                       'garbage collection at the start and the end of each '
                       'evaluation run.', dest='manual_gc_eval')               # trace_info : t_316
    group.add_argument('--disable-tp-comm-split-ag', action='store_false',     # trace_info : t_318, t_321
                       help='Disables the All-Gather overlap with fprop GEMM.',# trace_info : t_319
                       dest='tp_comm_split_ag')                                # trace_info : t_320
    group.add_argument('--disable-tp-comm-split-rs', action='store_false',     # trace_info : t_322, t_325
                       help='Disables the Reduce-Scatter overlap with fprop GEMM.',# trace_info : t_323
                       dest='tp_comm_split_rs')                                # trace_info : t_324

    return parser                                                              # trace_info : t_326


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')                  # trace_info : t_328

    group.add_argument('--seed', type=int, default=1234,                       # trace_info : t_329, t_331
                       help='Random seed used for python, numpy, '             # trace_info : t_330
                       'pytorch, and cuda.')
    group.add_argument('--data-parallel-random-init', action='store_true',     # trace_info : t_332, t_334
                       help='Enable random initialization of params '          # trace_info : t_333
                       'across data parallel ranks')
    group.add_argument('--init-method-std', type=float, default=0.02,          # trace_info : t_335, t_337
                       help='Standard deviation of the zero mean normal '      # trace_info : t_336
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',    # trace_info : t_338, t_340
                       help='Enable Xavier uniform parameter initialization')  # trace_info : t_339

    return parser                                                              # trace_info : t_341


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')                   # trace_info : t_343

    group.add_argument('--lr', type=float, default=None,                       # trace_info : t_344, t_346
                       help='Initial learning rate. Depending on decay style ' # trace_info : t_345
                       'and initial warmup, the learning rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',         # trace_info : t_347, t_350
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],# trace_info : t_348
                       help='Learning rate decay function.')                   # trace_info : t_349
    group.add_argument('--lr-decay-iters', type=int, default=None,             # trace_info : t_351, t_353
                       help='number of iterations to decay learning rate over,'# trace_info : t_352
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,           # trace_info : t_354, t_356
                       help='number of samples to decay learning rate over,'   # trace_info : t_355
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,       # trace_info : t_357, t_359
                       help='fraction of lr-warmup-(iters/samples) to use '    # trace_info : t_358
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,               # trace_info : t_360, t_362
                       help='number of iterations to linearly warmup '         # trace_info : t_361
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,             # trace_info : t_363, t_365
                       help='number of samples to linearly warmup '            # trace_info : t_364
                       'learning rate over.')
    group.add_argument('--lr-warmup-init', type=float, default=0.0,            # trace_info : t_366, t_368
                       help='Initial value for learning rate warmup. The '     # trace_info : t_367
                       'scheduler starts warmup from this value.')
    group.add_argument('--warmup', type=int, default=None,                     # trace_info : t_369, t_371
                       help='Old lr warmup argument, do not use. Use one of the'# trace_info : t_370
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,                    # trace_info : t_372, t_374
                       help='Minimum value for learning rate. The scheduler'   # trace_info : t_373
                       'clip values below this threshold.')
    group.add_argument('--override-opt_param-scheduler', action='store_true',  # trace_info : t_375, t_377
                       help='Reset the values of the scheduler (learning rate,'# trace_info : t_376
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-opt_param-scheduler', action='store_true',# trace_info : t_378, t_380
                       help='Use checkpoint to set the values of the scheduler '# trace_info : t_379
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    group.add_argument('--decoupled-lr', type=float, default=None,             # trace_info : t_381, t_383
                       help='Separate learning rate for the input and output layer')# trace_info : t_382
    group.add_argument('--decoupled-min-lr', type=float, default=None,         # trace_info : t_384, t_386
                       help='Minimum value for learning rate for the input and output layer. The scheduler'# trace_info : t_385
                       'clip values below this threshold')

    return parser                                                              # trace_info : t_387


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')                   # trace_info : t_389

    group.add_argument('--save', type=str, default=None,                       # trace_info : t_390, t_392
                       help='Output directory to save checkpoints to.')        # trace_info : t_391
    group.add_argument('--save-interval', type=int, default=None,              # trace_info : t_393, t_395
                       help='Number of iterations between checkpoint saves.')  # trace_info : t_394
    group.add_argument('--no-save-optim', action='store_true', default=None,   # trace_info : t_396, t_398
                       help='Do not save current optimizer.')                  # trace_info : t_397
    group.add_argument('--no-save-rng', action='store_true', default=None,     # trace_info : t_399, t_401
                       help='Do not save current rng state.')                  # trace_info : t_400
    group.add_argument('--load', type=str, default=None,                       # trace_info : t_402, t_404
                       help='Directory containing a model checkpoint.')        # trace_info : t_403
    group.add_argument('--no-load-optim', action='store_true', default=None,   # trace_info : t_405, t_407
                       help='Do not load optimizer when loading checkpoint.')  # trace_info : t_406
    group.add_argument('--no-load-rng', action='store_true', default=None,     # trace_info : t_408, t_410
                       help='Do not load rng state when loading checkpoint.')  # trace_info : t_409
    group.add_argument('--finetune', action='store_true',                      # trace_info : t_411, t_413
                       help='Load model for finetuning. Do not load optimizer '# trace_info : t_412
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,      # trace_info : t_414, t_416
                       help='Directory containing a pretrained model checkpoint for finetuning.')# trace_info : t_415
    group.add_argument('--ckpt-step', type=int, default=None,                  # trace_info : t_417, t_419
                       help='Checkpoint step to load model from.')             # trace_info : t_418
    group.add_argument('--no-initialization', action='store_false',            # trace_info : t_420, t_423
                       help='Do not perform initialization when building model, '# trace_info : t_421
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')                          # trace_info : t_422
    group.add_argument('--use-checkpoint-args', action='store_true',           # trace_info : t_424, t_426
                       help='Override any command line arguments with arguments '# trace_info : t_425
                       'from the checkpoint')
    group.add_argument('--exit-on-missing-checkpoint', action='store_true',    # trace_info : t_427, t_429
                       help="If '--load' is set, but checkpoint is not found " # trace_info : t_428
                       "(e.g., path typo), then exit instead of random "
                       "initialization.")
    group.add_argument('--use-dist-ckpt', action='store_true',                 # trace_info : t_430, t_432
                       help='Use distributed checkpoint format.')              # trace_info : t_431
    group.add_argument('--auto-detect-ckpt-format', action='store_true',       # trace_info : t_433, t_435
                       help='Determine if the checkpoint format is in legacy or distributed format.'# trace_info : t_434
                            ' If False, expects distributed checkpoint iff args.use_dist_ckpt.'
                            ' Might slow down loading a bit (double rank0 ckpt load).')
    group.add_argument('--dist-ckpt-format', type=str, default='torch_dist',   # trace_info : t_436, t_439
                       choices=['zarr', 'torch_dist'],                         # trace_info : t_437
                       help='Distributed checkpoint format to use.')           # trace_info : t_438
    group.add_argument('--ckpt-fully-parallel-save', action='store_true',      # trace_info : t_440, t_442
                       help='Apply full save parallelization across DP for'    # trace_info : t_441
                            ' distributed checkpoints. Depending on ckpt format'
                            ' might increase number of files in the checkpoint.')
    group.add_argument('--async-save', action='store_true', default=None,      # trace_info : t_443, t_445
                       help='Apply async checkpointing save. Currently works only with'# trace_info : t_444
                            '`torch_dist` distributed checkpoint format.')
    group.add_argument('--ckpt-fully-parallel-load', action='store_true',      # trace_info : t_446, t_448
                       help='Apply full load parallelization across DP for'    # trace_info : t_447
                            ' distributed checkpoints.')
    group.add_argument('--ckpt-assume-constant-structure', action='store_true',# trace_info : t_449, t_451
                       help='If the model and optimizer state dict structure is'# trace_info : t_450
                            'constant throughout a *single training job*, it allows for'
                            'different checkpointing performance optimizations.')
    return parser                                                              # trace_info : t_452


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')                 # trace_info : t_454

    group.add_argument('--fp16', action='store_true',                          # trace_info : t_455, t_457
                       help='Run model in fp16 mode.')                         # trace_info : t_456
    group.add_argument('--bf16', action='store_true',                          # trace_info : t_458, t_460
                       help='Run model in bfloat16 mode.')                     # trace_info : t_459
    group.add_argument('--loss-scale', type=float, default=None,               # trace_info : t_461, t_463
                       help='Static loss scaling, positive power of 2 '        # trace_info : t_462
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,      # trace_info : t_464, t_466
                       help='Initial loss-scale for dynamic loss scaling.')    # trace_info : t_465
    group.add_argument('--min-loss-scale', type=float, default=1.0,            # trace_info : t_467, t_469
                       help='Minimum loss scale for dynamic loss scaling.')    # trace_info : t_468
    group.add_argument('--loss-scale-window', type=float, default=1000,        # trace_info : t_470, t_472
                       help='Window over which to raise/lower dynamic scale.') # trace_info : t_471
    group.add_argument('--hysteresis', type=int, default=2,                    # trace_info : t_473, t_475
                       help='hysteresis for dynamic loss scaling')             # trace_info : t_474
    group.add_argument('--fp32-residual-connection', action='store_true',      # trace_info : t_476, t_478
                       help='Move residual connections to fp32.')              # trace_info : t_477
    group.add_argument('--apply-query-key-layer-scaling', action='store_true', # trace_info : t_479, t_481
                       help='Scale Q * K^T by 1 / layer-number. '              # trace_info : t_480
                       'Useful for fp16 training.')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',     # trace_info : t_482, t_484
                       help='Run attention masking and softmax in fp32. '      # trace_info : t_483
                       'This flag is ignored unless '
                       '--no-query-key-layer-scaling is specified.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',                 # trace_info : t_485, t_488
                       action='store_true',                                    # trace_info : t_486
                       help='Gradient accumulation and all-reduce in fp32.')   # trace_info : t_487
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',         # trace_info : t_489, t_491
                       help='Move the cross entropy unreduced loss calculation'# trace_info : t_490
                       'for lm head to fp16.')

    return parser                                                              # trace_info : t_492


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')                     # trace_info : t_494

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,    # trace_info : t_495, t_497
                       help='Degree of tensor model parallelism.')             # trace_info : t_496
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,  # trace_info : t_498, t_500
                       help='Degree of pipeline model parallelism.')           # trace_info : t_499
    group.add_argument('--pipeline-model-parallel-split-rank',                 # trace_info : t_501, t_504
                       type=int, default=None,                                 # trace_info : t_502
                       help='Rank where encoder and decoder should be split.') # trace_info : t_503
    group.add_argument('--model-parallel-size', type=int, default=None,        # trace_info : t_505, t_507
                       help='Old model parallel argument, do not use. Use '    # trace_info : t_506
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,# trace_info : t_508, t_510
                       help='Number of layers per virtual pipeline stage')     # trace_info : t_509
    group.add_argument('--no-overlap-p2p-communication', action='store_false', # trace_info : t_511, t_514
                       help='overlap pipeline parallel communication with forward and backward chunks',# trace_info : t_512
                       dest='overlap_p2p_comm')                                # trace_info : t_513
    group.add_argument('--distributed-backend', default='nccl',                # trace_info : t_515, t_518
                       choices=['nccl', 'gloo'],                               # trace_info : t_516
                       help='Which backend to use for distributed training.')  # trace_info : t_517
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,  # trace_info : t_519, t_521
                       help='Timeout minutes for torch.distributed.')          # trace_info : t_520
    group.add_argument('--overlap-grad-reduce', action='store_true',           # trace_info : t_522, t_524
                       default=False, help='If set, overlap DDP grad reduce.') # trace_info : t_523
    group.add_argument('--no-delay-grad-reduce', action='store_false',         # trace_info : t_525, t_528
                       help='If not set, delay / synchronize grad reductions in all but first PP stage.',# trace_info : t_526
                       dest='delay_grad_reduce')                               # trace_info : t_527
    group.add_argument('--ddp-bucket-size', type=int, default=None,            # trace_info : t_529, t_531
                       help='Bucket size for data-parallel communication')     # trace_info : t_530
    group.add_argument('--overlap-param-gather', action='store_true',          # trace_info : t_532, t_534
                       default=False, help='If set, overlap param all-gather in distributed optimizer.')# trace_info : t_533
    group.add_argument('--delay-param-gather', action='store_true',            # trace_info : t_535, t_537
                       default=False, help='If set, delay / synchronize param all-gathers in all but first PP stage.')# trace_info : t_536
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',# trace_info : t_538, t_541
                       help='If not set, use scatter/gather to optimize communication of tensors in pipeline.',# trace_info : t_539
                       dest='scatter_gather_tensors_in_pipeline')              # trace_info : t_540
    group.add_argument('--use-ring-exchange-p2p', action='store_true',         # trace_info : t_542, t_544
                       default=False, help='If set, use custom-built ring exchange '# trace_info : t_543
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local_rank', type=int, default=None,                 # trace_info : t_545, t_547
                       help='local rank passed from distributed launcher.')    # trace_info : t_546
    group.add_argument('--lazy-mpu-init', type=bool, required=False,           # trace_info : t_548, t_550
                       help='If set to True, initialize_megatron() '           # trace_info : t_549
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--standalone-embedding-stage', action='store_true',    # trace_info : t_551, t_553
                       default=False, help='If set, *input* embedding layer '  # trace_info : t_552
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',     # trace_info : t_554, t_556
                       help='Use distributed optimizer.')                      # trace_info : t_555
    group.add_argument('--context-parallel-size', type=int, default=1,         # trace_info : t_557, t_559
                       help='Degree of context parallelism.')                  # trace_info : t_558
    group.add_argument('--nccl-communicator-config-path', type=str, default=None,# trace_info : t_560, t_562
                       help='Path to the yaml file with NCCL communicator '    # trace_info : t_561
                       'configurations. The number of min/max thread groups and thread '
                       'group cluster size of each communicator can be configured by '
                       'setting `min_ctas`, `max_ctas`, and `cga_cluster_size`.')
    group.add_argument('--use-tp-pp-dp-mapping', action='store_true', default=False,# trace_info : t_563, t_565
                        help='If set, distributed ranks initialize order is changed '# trace_info : t_564
                        'from tp-dp-pp to tp-pp-dp. Make sure EP and CP aren\'t used '
                        'with this option enabled')
    return parser                                                              # trace_info : t_566


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')                      # trace_info : t_568

    group.add_argument('--eval-iters', type=int, default=100,                  # trace_info : t_569, t_571
                       help='Number of iterations to run for evaluation'       # trace_info : t_570
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,              # trace_info : t_572, t_574
                       help='Interval between running evaluation on '          # trace_info : t_573
                       'validation set.')
    group.add_argument("--test-mode", action="store_true", help='Run all real-time test alongside the experiment.')# trace_info : t_575
    group.add_argument('--skip-train', action='store_true',                    # trace_info : t_576, t_578
                       default=False, help='If set, bypass the training loop, '# trace_info : t_577
                       'optionally do evaluation for validation/test, and exit.')

    return parser                                                              # trace_info : t_579


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')             # trace_info : t_581

    group.add_argument('--data-path', nargs='*', default=None,                 # trace_info : t_582, t_584
                       help='The weight and prefix list for a set of train, validation, and test'# trace_info : t_583
                       'datasets which split according to --split. The accepted formats are: '
                       '(1) a single prefix, '
                       '(2) a list of weight prefix pairs e.g. weight1 prefix1 weight2 prefix2, '
                       '(3) a list of prefixes e.g. prefix1 prefix2. '
                       'For (3), weights are inferred from the lengths of the contributing datasets. '
                       'This argument is exclusive to the other independent --*-data-path arguments.')
    group.add_argument('--split', type=str, default='969, 30, 1',              # trace_info : t_585, t_587
                       help='Comma-separated list of proportions for training,'# trace_info : t_586
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train-data-path', nargs='*', default=None,           # trace_info : t_588, t_590
                       help='The weight and prefix list for an independent train dataset. '# trace_info : t_589
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--valid-data-path', nargs='*', default=None,           # trace_info : t_591, t_593
                       help='The weight and prefix list for an independent validation dataset. '# trace_info : t_592
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--test-data-path', nargs='*', default=None,            # trace_info : t_594, t_596
                       help='The weight and prefix list for an independent test dataset. '# trace_info : t_595
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--data-cache-path', default=None,                      # trace_info : t_597, t_599
                       help='Path to a directory to hold cached index files.') # trace_info : t_598
    group.add_argument('--no-mmap-bin-files', action='store_false',            # trace_info : t_600, t_603
                       help='Disable mmap-ing of .bin files.',                 # trace_info : t_601
                       dest='mmap_bin_files')                                  # trace_info : t_602
    group.add_argument('--mock-data', action='store_true',                     # trace_info : t_604, t_606
                       help='Skip data loading and validation and opt for artificial '# trace_info : t_605
                       'generation of mock data when an implementation is available.')
    group.add_argument('--vocab-size', type=int, default=None,                 # trace_info : t_607, t_609
                       help='Size of vocab before EOD or padding.')            # trace_info : t_608
    group.add_argument('--vocab-file', type=str, default=None,                 # trace_info : t_610, t_612
                       help='Path to the vocab file.')                         # trace_info : t_611
    group.add_argument('--merge-file', type=str, default=None,                 # trace_info : t_613, t_615
                       help='Path to the BPE merge file.')                     # trace_info : t_614
    group.add_argument('--vocab-extra-ids', type=int, default=0,               # trace_info : t_616, t_618
                       help='Number of additional vocabulary tokens. '         # trace_info : t_617
                            'They are used for span masking in the T5 model')
    group.add_argument('--seq-length', type=int, default=None,                 # trace_info : t_619, t_621
                       help='Maximum sequence length to process.')             # trace_info : t_620
    group.add_argument('--encoder-seq-length', type=int, default=None,         # trace_info : t_622, t_624
                       help='Maximum encoder sequence length to process.'      # trace_info : t_623
                       'This should be exclusive of --seq-length')
    group.add_argument('--decoder-seq-length', type=int, default=None,         # trace_info : t_625, t_627
                       help="Maximum decoder sequence length to process.")     # trace_info : t_626
    group.add_argument('--retriever-seq-length', type=int, default=256,        # trace_info : t_628, t_630
                       help='Maximum sequence length for the biencoder model ' # trace_info : t_629
                       'for retriever')
    group.add_argument('--sample-rate', type=float, default=1.0,               # trace_info : t_631, t_633
                       help='sample rate for training data. Supposed to be 0 ' # trace_info : t_632
                            ' < sample_rate < 1')
    group.add_argument('--mask-prob', type=float, default=0.15,                # trace_info : t_634, t_636
                       help='Probability of replacing a token with mask.')     # trace_info : t_635
    group.add_argument('--short-seq-prob', type=float, default=0.1,            # trace_info : t_637, t_639
                       help='Probability of producing a short sequence.')      # trace_info : t_638
    group.add_argument('--num-workers', type=int, default=2,                   # trace_info : t_640, t_642
                       help="Dataloader number of workers.")                   # trace_info : t_641
    group.add_argument('--tokenizer-type', type=str,                           # trace_info : t_643, t_647
                       default=None,                                           # trace_info : t_644
                       choices=['BertWordPieceLowerCase',                      # trace_info : t_645
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'Llama2Tokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')                  # trace_info : t_646
    group.add_argument('--tokenizer-model', type=str, default=None,            # trace_info : t_648, t_650
                       help='Sentencepiece tokenizer model.')                  # trace_info : t_649
    group.add_argument('--reset-position-ids', action='store_true',            # trace_info : t_651, t_653
                       help='Reset posistion ids after end-of-document token.')# trace_info : t_652
    group.add_argument('--reset-attention-mask', action='store_true',          # trace_info : t_654, t_656
                       help='Reset self attention maske after '                # trace_info : t_655
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',                 # trace_info : t_657, t_659
                       help='Mask loss for the end of document tokens.')       # trace_info : t_658
    group.add_argument('--no-create-attention-mask-in-dataloader', action='store_false',# trace_info : t_660, t_663
                       help='If set, do not create attention_masks in dataloader.',# trace_info : t_661
                       dest='create_attention_mask_in_dataloader')             # trace_info : t_662
    group.add_argument('--num-dataset-builder-threads', type=int, default=1,   # trace_info : t_664, t_666
                       help='Number of parallel threads per rank for dataset builder')# trace_info : t_665
    return parser                                                              # trace_info : t_667


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')                      # trace_info : t_669

    group.add_argument('--adlr-autoresume', action='store_true',               # trace_info : t_670, t_672
                       help='Enable autoresume on adlr cluster.')              # trace_info : t_671
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,   # trace_info : t_673, t_675
                       help='Intervals over which check for autoresume'        # trace_info : t_674
                       'termination signal')

    return parser                                                              # trace_info : t_676


def _add_biencoder_args(parser):
    group = parser.add_argument_group(title='biencoder')                       # trace_info : t_678

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,              # trace_info : t_679, t_681
                       help='Size of block embeddings to be used in ICT and '  # trace_info : t_680
                        'REALM (paper default: 128)')
    group.add_argument('--biencoder-projection-dim', type=int, default=0,      # trace_info : t_682, t_684
                       help='Size of projection head used in biencoder (paper' # trace_info : t_683
                        ' default: 128)')
    group.add_argument('--biencoder-shared-query-context-model', action='store_true',# trace_info : t_685, t_687
                        help='Whether to share the parameters of the query '   # trace_info : t_686
                        'and context models or not')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,                   # trace_info : t_688, t_690
                       help='Directory containing an ICTBertModel checkpoint') # trace_info : t_689
    group.add_argument('--bert-load', type=str, default=None,                  # trace_info : t_691, t_693
                       help='Directory containing an BertModel checkpoint '    # trace_info : t_692
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,           # trace_info : t_694, t_696
                       help='Path to titles dataset used for ICT')             # trace_info : t_695
    group.add_argument('--query-in-block-prob', type=float, default=0.1,       # trace_info : t_697, t_699
                       help='Probability of keeping query in block for '       # trace_info : t_698
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',             # trace_info : t_700, t_702
                       help='Whether to use one sentence documents in ICT')    # trace_info : t_701
    group.add_argument('--evidence-data-path', type=str, default=None,         # trace_info : t_703, t_705
                       help='Path to Wikipedia Evidence frm DPR paper')        # trace_info : t_704

    # training
    group.add_argument('--retriever-report-topk-accuracies', nargs='+', type=int,# trace_info : t_706, t_708
                        default=[], help="Which top-k accuracies to report "   # trace_info : t_707
                        "(e.g. '1 5 20')")
    group.add_argument('--retriever-score-scaling', action='store_true',       # trace_info : t_709, t_711
                       help='Whether to scale retriever scores by inverse '    # trace_info : t_710
                        'square root of hidden size')

    # faiss index
    group.add_argument('--block-data-path', type=str, default=None,            # trace_info : t_712, t_714
                       help='Where to save/load BlockData to/from')            # trace_info : t_713
    group.add_argument('--embedding-path', type=str, default=None,             # trace_info : t_715, t_717
                       help='Where to save/load Open-Retrieval Embedding'      # trace_info : t_716
                        ' data to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,          # trace_info : t_718, t_720
                       help='How large of batches to use when doing indexing ' # trace_info : t_719
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,       # trace_info : t_721, t_723
                       help='After how many batches should the indexer '       # trace_info : t_722
                       'report progress')
    return parser                                                              # trace_info : t_724


def _add_vision_args(parser):
    group = parser.add_argument_group(title="vision")                          # trace_info : t_726

    # general vision arguements
    group.add_argument('--num-classes', type=int, default=1000,                # trace_info : t_727, t_729
                       help='num of classes in vision classificaiton task')    # trace_info : t_728
    group.add_argument('--img-h', type=int, default=224,                       # trace_info : t_730, t_732
                       help='Image height for vision classification task')     # trace_info : t_731
    group.add_argument('--img-w', type=int, default=224,                       # trace_info : t_733, t_735
                       help='Image height for vision classification task')     # trace_info : t_734
    group.add_argument('--num-channels', type=int, default=3,                  # trace_info : t_736, t_738
                       help='Number of channels in input image data')          # trace_info : t_737
    group.add_argument('--patch-dim', type=int, default=16,                    # trace_info : t_739, t_741
                       help='patch dimension')                                 # trace_info : t_740
    group.add_argument('--classes-fraction', type=float, default=1.0,          # trace_info : t_742, t_744
                       help='training with fraction of classes.')              # trace_info : t_743
    group.add_argument('--data-per-class-fraction', type=float, default=1.0,   # trace_info : t_745, t_747
                       help='training with fraction of data per class.')       # trace_info : t_746
    group.add_argument('--no-data-sharding', action='store_false',             # trace_info : t_748, t_751
                       help='Disable data sharding.',                          # trace_info : t_749
                       dest='data_sharding')                                   # trace_info : t_750
    group.add_argument('--head-lr-mult', type=float, default=1.0,              # trace_info : t_752, t_754
                       help='learning rate multiplier for head during finetuning')# trace_info : t_753

    # pretraining type and backbone selection`
    group.add_argument('--vision-pretraining', action='store_true',            # trace_info : t_755, t_757
                       help='flag to indicate vision pretraining')             # trace_info : t_756
    group.add_argument('--vision-pretraining-type', type=str, default='classify',# trace_info : t_758, t_761
                       choices=['classify', 'inpaint', 'dino'],                # trace_info : t_759
                       help='pretraining objectives')                          # trace_info : t_760
    group.add_argument('--vision-backbone-type', type=str, default='vit',      # trace_info : t_762, t_765
                       choices=['vit', 'mit', 'swin'],                         # trace_info : t_763
                       help='backbone types types')                            # trace_info : t_764
    group.add_argument('--swin-backbone-type', type=str, default='tiny',       # trace_info : t_766, t_769
                       choices=['tiny', 'base', 'h3'],                         # trace_info : t_767
                       help='pretraining objectives')                          # trace_info : t_768
    # inpainting arguments
    group.add_argument('--mask-type', type=str, default='random',              # trace_info : t_770, t_773
                       choices=['random', 'row'],                              # trace_info : t_771
                       help='mask types')                                      # trace_info : t_772
    group.add_argument('--mask-factor', type=float, default=1.0,               # trace_info : t_774, t_776
                       help='mask size scaling parameter')                     # trace_info : t_775

    # dino arguments
    group.add_argument('--iter-per-epoch', type=int, default=1250,             # trace_info : t_777, t_779
                       help='iterations per epoch')                            # trace_info : t_778
    group.add_argument('--dino-local-img-size', type=int, default=96,          # trace_info : t_780, t_782
                       help='Image size for vision classification task')       # trace_info : t_781
    group.add_argument('--dino-local-crops-number', type=int, default=10,      # trace_info : t_783, t_785
                       help='Number of local crops')                           # trace_info : t_784
    group.add_argument('--dino-head-hidden-size', type=int, default=2048,      # trace_info : t_786, t_788
                       help='Hidden dimension size in dino head')              # trace_info : t_787
    group.add_argument('--dino-bottleneck-size', type=int, default=256,        # trace_info : t_789, t_791
                       help='Bottle neck dimension in dino head ')             # trace_info : t_790
    group.add_argument('--dino-freeze-last-layer', type=float, default=1,      # trace_info : t_792, t_794
                       help='Freezing last layer weights')                     # trace_info : t_793
    group.add_argument('--dino-norm-last-layer', action='store_true',          # trace_info : t_795, t_797
                       help='Disable Norm in last layer.')                     # trace_info : t_796
    group.add_argument('--dino-warmup-teacher-temp', type=float, default=0.04, # trace_info : t_798, t_800
                       help='warump teacher temperature')                      # trace_info : t_799
    group.add_argument('--dino-teacher-temp', type=float, default=0.07,        # trace_info : t_801, t_803
                       help='teacher temperature')                             # trace_info : t_802
    group.add_argument('--dino-warmup-teacher-temp-epochs', type=int, default=30,# trace_info : t_804, t_806
                       help='warmup teacher temperaure epochs')                # trace_info : t_805

    # regularization arguments
    group.add_argument('--qk-layernorm', action='store_true',                  # trace_info : t_807, t_809
                       help='Whether to layer normalize the q and k attention embeddings.')# trace_info : t_808

    return parser                                                              # trace_info : t_810

def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")                             # trace_info : t_812
    group.add_argument('--expert-model-parallel-size', type=int, default=1,    # trace_info : t_813, t_815
                       help='Degree of expert model parallelism.')             # trace_info : t_814
    group.add_argument('--num-experts', type=int, default=None,                # trace_info : t_816, t_818
                       help='Number of Experts in MoE (None means no MoE)')    # trace_info : t_817
    group.add_argument('--moe-router-load-balancing-type', type=str,           # trace_info : t_819, t_823
                       choices=['aux_loss', 'sinkhorn', "none"],               # trace_info : t_820
                       default='aux_loss',                                     # trace_info : t_821
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".')# trace_info : t_822
    group.add_argument('--moe-router-topk', type=int, default=2,               # trace_info : t_824, t_826
                       help='Number of experts to route to for each token. The default is 2.')# trace_info : t_825
    group.add_argument('--moe-grouped-gemm', action='store_true',              # trace_info : t_827, t_829
                       help='When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).')# trace_info : t_828
    group.add_argument('--moe-aux-loss-coeff', type=float, default=0.0,        # trace_info : t_830, t_832
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')# trace_info : t_831
    group.add_argument('--moe-z-loss-coeff', type=float, default=None,         # trace_info : t_833, t_835
                       help='Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.')# trace_info : t_834
    group.add_argument('--moe-input-jitter-eps', type=float, default=None,     # trace_info : t_836, t_838
                       help='Add noise to the input tensor by applying jitter with a specified epsilon value.')# trace_info : t_837
    group.add_argument('--moe-token-dispatcher-type', type=str,                # trace_info : t_839, t_843
                       choices=['allgather', 'alltoall'],                      # trace_info : t_840
                       default='allgather',                                    # trace_info : t_841
                       help='.')                                               # trace_info : t_842
    group.add_argument('--moe-per-layer-logging', action='store_true',         # trace_info : t_844, t_846
                       help='Enable per-layer logging for MoE, currently supports auxiliary loss and z loss.')# trace_info : t_845
    # Token dropping arguments
    group.add_argument('--moe-expert-capacity-factor', type=float, default=None,# trace_info : t_847, t_849
                       help='The capacity factor for each expert, None means no token will be dropped.')# trace_info : t_848
    group.add_argument('--moe-pad-expert-input-to-capacity', action='store_true',# trace_info : t_850, t_852
                       help='Pads the input for each expert to match the expert capacity length, effective only after the --moe-expert-capacity-factor is set.')# trace_info : t_851
    group.add_argument('--moe-token-drop-policy', type=str, default='probs', choices=['probs', 'position'],# trace_info : t_853, t_855
                       help='The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.')# trace_info : t_854
    group.add_argument('--moe-layer-recompute', action='store_true',           # trace_info : t_856, t_858
                       help='Enable checkpointing for moe_layer, should be used when memory is not sufficient.')# trace_info : t_857
    group.add_argument('--moe-extended-tp', action='store_true',               # trace_info : t_859, t_861
                       help='Alternative to expert parallelism, all experts are sharded across TPXEP domain.')# trace_info : t_860

    return parser                                                              # trace_info : t_862

def _add_experimental_args(parser):
    group = parser.add_argument_group(title='experimental')                    # trace_info : t_1078

    group.add_argument('--spec', type=str, default=None, nargs='*',            # trace_info : t_1079, t_1081
                       help='Specify the <module_location function_name> pair '# trace_info : t_1080
                       'that returns a spec to customize a model, transformer '
                       'block, or transformer layer, depending on the use case.'
                       'To use local spec specify local as the argument.'
                       'For more details, see the model class, '
                       '`transformer_block.py`, or `transformer_layer.py`')
    group.add_argument('--yaml-cfg', type=str, default=None,                   # trace_info : t_1082, t_1084
                       help = 'Config file to add additional arguments')       # trace_info : t_1083

    return parser                                                              # trace_info : t_1085
