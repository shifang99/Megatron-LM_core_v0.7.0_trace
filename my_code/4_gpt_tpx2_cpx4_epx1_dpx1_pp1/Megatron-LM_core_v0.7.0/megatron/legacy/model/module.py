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
        super(MegatronModule, self).__init__()                                 # trace_info : t_12002
        self.config = config                                                   # trace_info : t_12003
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights# trace_info : t_12004


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
    if not isinstance(val, (tuple, list)):                                     # trace_info : t_18165, t_18167, t_18174, t_18181, t_18816, ...
        return conversion(val)                                                 # trace_info : t_18168, t_18175, t_18182, t_18817, t_21354, ...
    rtn = [conversion_helper(v, conversion) for v in val]                      # trace_info : t_18166, t_21352, t_24538
    if isinstance(val, tuple):                                                 # trace_info : t_18188, t_21374, t_24560
        rtn = tuple(rtn)                                                       # trace_info : t_18189, t_21375, t_24561
    return rtn                                                                 # trace_info : t_18190, t_21376, t_24562


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):                                                  # trace_info : t_18163, t_21349, t_24535
        val_typecheck = val                                                    # trace_info : t_18169, t_18176, t_18183, t_21355, t_21362, ...
        if isinstance(val_typecheck, (Parameter, Variable)):                   # trace_info : t_18170, t_18177, t_18184, t_21356, t_21363, ...
            val_typecheck = val.data                                           # trace_info : t_18171, t_18178, t_18185, t_21357, t_21364, ...
        if isinstance(val_typecheck, _FLOAT_TYPES):                            # trace_info : t_18172, t_18179, t_18186, t_21358, t_21365, ...
            val = float16_convertor(val)
        return val                                                             # trace_info : t_18173, t_18180, t_18187, t_21359, t_21366, ...
    return conversion_helper(val, half_conversion)                             # trace_info : t_18164, t_21350, t_24536


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):                                                 # trace_info : t_18814, t_22000, t_25186
        val_typecheck = val                                                    # trace_info : t_18818, t_22004, t_25190
        if isinstance(val_typecheck, (Parameter, Variable)):                   # trace_info : t_18819, t_22005, t_25191
            val_typecheck = val.data                                           # trace_info : t_18820, t_22006, t_25192
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):              # trace_info : t_18821, t_22007, t_25193
            val = val.float()
        return val                                                             # trace_info : t_18822, t_22008, t_25194
    return conversion_helper(val, float_conversion)                            # trace_info : t_18815, t_22001, t_25187



class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()                                  # trace_info : t_12001

        if args.fp16:                                                          # trace_info : t_12005
            self.add_module('module', module.half())                           # trace_info : t_12006
            def float16_convertor(val):                                        # trace_info : t_12007
                return val.half()
        elif args.bf16:
            self.add_module('module', module.bfloat16())
            def float16_convertor(val):
                return val.bfloat16()
        else:
            raise Exception('should not be here')

        self.float16_convertor = float16_convertor                             # trace_info : t_12008


    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)                      # trace_info : t_17901, t_21087, t_24273


    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():                                      # trace_info : t_18153, t_21339, t_24525
            inputs = fp32_to_float16(inputs, self.float16_convertor)           # trace_info : t_18162, t_21348, t_24534
        outputs = self.module(*inputs, **kwargs)                               # trace_info : t_18191, t_21377, t_24563
        if mpu.is_pipeline_last_stage():                                       # trace_info : t_18798, t_21984, t_25170
            outputs = float16_to_fp32(outputs)                                 # trace_info : t_18813, t_21999, t_25185
        return outputs                                                         # trace_info : t_18823, t_22009, t_25195


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
