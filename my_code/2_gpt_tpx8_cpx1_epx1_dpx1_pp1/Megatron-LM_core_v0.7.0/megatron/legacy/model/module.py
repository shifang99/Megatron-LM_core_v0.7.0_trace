# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron Module"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron.training import get_args
from megatron.core import mpu, tensor_parallel


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)



def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared



class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, config=None, share_embeddings_and_output_weights=True):
        super(MegatronModule, self).__init__()                                 # trace_info : t_14010
        self.config = config                                                   # trace_info : t_14011
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights# trace_info : t_14012


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(prefix=prefix, keep_vars=keep_vars)


    def shared_embedding_or_output_weight(self):
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_embeddings_and_output_weights:
                raise Exception('shared_embedding_or_output_weight() called for last '
                                'stage, but share_embeddings_and_output_weights is false')
            return self.word_embeddings.weight


    def initialize_word_embeddings(self):
        args = get_args()
        if not self.share_embeddings_and_output_weights:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_embeddings_and_output_weights is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if args.pipeline_model_parallel_size == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.shared_embedding_or_output_weight().zero_out_wgrad = True
            return

        if mpu.is_pipeline_first_stage() and self.pre_process and not self.post_process:
           self.shared_embedding_or_output_weight().shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage() and not self.pre_process:
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.padded_vocab_size, self.config.hidden_size,
                config=self.config, init_method=self.config.init_method)
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True
            self.word_embeddings.weight.shared_embedding = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not mpu.is_pipeline_first_stage(ignore_virtual=True) and \
                self.pre_process:
            self.language_model.embedding.zero_parameters()

        if not torch.distributed.is_initialized():
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print("WARNING! Distributed processes aren't initialized, so "
                      "word embeddings in the last layer are not initialized. "
                      "If you are just manipulating a model this is fine, but "
                      "this needs to be handled manually. If you are training "
                      "something is definitely wrong.")
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        if mpu.is_rank_in_embedding_group():
            self.shared_embedding_or_output_weight().data = self.shared_embedding_or_output_weight().data.cuda()
            torch.distributed.all_reduce(self.shared_embedding_or_output_weight().data,
                                         group=mpu.get_embedding_group())

        # Ensure that encoder(first stage) and decoder(split stage) position
        # embeddings have the same initial parameter values
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if mpu.is_rank_in_position_embedding_group() and \
                args.pipeline_model_parallel_split_rank is not None:
            # TODO: Support tokentype embedding.
            self.language_model.embedding.cuda()
            position_embeddings = self.language_model.embedding.position_embeddings
            torch.distributed.all_reduce(position_embeddings.weight.data,
                                         group=mpu.get_position_embedding_group())


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):                                     # trace_info : t_20046, t_20048, t_20055, t_20062, t_21255, ...
        return conversion(val)                                                 # trace_info : t_20049, t_20056, t_20063, t_21256, t_23661, ...
    rtn = [conversion_helper(v, conversion) for v in val]                      # trace_info : t_20047, t_23659, t_27269
    if isinstance(val, tuple):                                                 # trace_info : t_20069, t_23681, t_27291
        rtn = tuple(rtn)                                                       # trace_info : t_20070, t_23682, t_27292
    return rtn                                                                 # trace_info : t_20071, t_23683, t_27293


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):                                                  # trace_info : t_20044, t_23656, t_27266
        val_typecheck = val                                                    # trace_info : t_20050, t_20057, t_20064, t_23662, t_23669, ...
        if isinstance(val_typecheck, (Parameter, Variable)):                   # trace_info : t_20051, t_20058, t_20065, t_23663, t_23670, ...
            val_typecheck = val.data                                           # trace_info : t_20052, t_20059, t_20066, t_23664, t_23671, ...
        if isinstance(val_typecheck, _FLOAT_TYPES):                            # trace_info : t_20053, t_20060, t_20067, t_23665, t_23672, ...
            val = float16_convertor(val)
        return val                                                             # trace_info : t_20054, t_20061, t_20068, t_23666, t_23673, ...
    return conversion_helper(val, half_conversion)                             # trace_info : t_20045, t_23657, t_27267


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):                                                 # trace_info : t_21253, t_24863, t_28473
        val_typecheck = val                                                    # trace_info : t_21257, t_24867, t_28477
        if isinstance(val_typecheck, (Parameter, Variable)):                   # trace_info : t_21258, t_24868, t_28478
            val_typecheck = val.data                                           # trace_info : t_21259, t_24869, t_28479
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):              # trace_info : t_21260, t_24870, t_28480
            val = val.float()
        return val                                                             # trace_info : t_21261, t_24871, t_28481
    return conversion_helper(val, float_conversion)                            # trace_info : t_21254, t_24864, t_28474



class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()                                  # trace_info : t_14009

        if args.fp16:                                                          # trace_info : t_14013
            self.add_module('module', module.half())                           # trace_info : t_14014
            def float16_convertor(val):                                        # trace_info : t_14015
                return val.half()
        elif args.bf16:
            self.add_module('module', module.bfloat16())
            def float16_convertor(val):
                return val.bfloat16()
        else:
            raise Exception('should not be here')

        self.float16_convertor = float16_convertor                             # trace_info : t_14016


    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)                      # trace_info : t_19889, t_23501, t_27111


    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():                                      # trace_info : t_20034, t_23646, t_27256
            inputs = fp32_to_float16(inputs, self.float16_convertor)           # trace_info : t_20043, t_23655, t_27265
        outputs = self.module(*inputs, **kwargs)                               # trace_info : t_20072, t_23684, t_27294
        if mpu.is_pipeline_last_stage():                                       # trace_info : t_21237, t_24847, t_28457
            outputs = float16_to_fp32(outputs)                                 # trace_info : t_21252, t_24862, t_28472
        return outputs                                                         # trace_info : t_21262, t_24872, t_28482


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
