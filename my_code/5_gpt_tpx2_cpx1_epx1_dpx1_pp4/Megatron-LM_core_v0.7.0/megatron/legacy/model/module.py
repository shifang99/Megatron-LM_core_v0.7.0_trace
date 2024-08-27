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
        super(MegatronModule, self).__init__()                                 # trace_info : t_12431
        self.config = config                                                   # trace_info : t_12432
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights# trace_info : t_12433


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
    if not isinstance(val, (tuple, list)):                                     # trace_info : t_18312, t_18314, t_18321, t_18328, t_22042, ...
        return conversion(val)                                                 # trace_info : t_18315, t_18322, t_18329, t_22045, t_22052, ...
    rtn = [conversion_helper(v, conversion) for v in val]                      # trace_info : t_18313, t_22043, t_25771
    if isinstance(val, tuple):                                                 # trace_info : t_18335, t_22065, t_25793
        rtn = tuple(rtn)                                                       # trace_info : t_18336, t_22066, t_25794
    return rtn                                                                 # trace_info : t_18337, t_22067, t_25795


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):                                                  # trace_info : t_18310, t_22040, t_25768
        val_typecheck = val                                                    # trace_info : t_18316, t_18323, t_18330, t_22046, t_22053, ...
        if isinstance(val_typecheck, (Parameter, Variable)):                   # trace_info : t_18317, t_18324, t_18331, t_22047, t_22054, ...
            val_typecheck = val.data                                           # trace_info : t_18318, t_18325, t_18332, t_22048, t_22055, ...
        if isinstance(val_typecheck, _FLOAT_TYPES):                            # trace_info : t_18319, t_18326, t_18333, t_22049, t_22056, ...
            val = float16_convertor(val)
        return val                                                             # trace_info : t_18320, t_18327, t_18334, t_22050, t_22057, ...
    return conversion_helper(val, half_conversion)                             # trace_info : t_18311, t_22041, t_25769


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)



class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()                                  # trace_info : t_12430

        if args.fp16:                                                          # trace_info : t_12434
            self.add_module('module', module.half())                           # trace_info : t_12435
            def float16_convertor(val):                                        # trace_info : t_12436
                return val.half()
        elif args.bf16:
            self.add_module('module', module.bfloat16())
            def float16_convertor(val):
                return val.bfloat16()
        else:
            raise Exception('should not be here')

        self.float16_convertor = float16_convertor                             # trace_info : t_12437


    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)                      # trace_info : t_18162, t_21892, t_25620


    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():                                      # trace_info : t_18300, t_22030, t_25758
            inputs = fp32_to_float16(inputs, self.float16_convertor)           # trace_info : t_18309, t_22039, t_25767
        outputs = self.module(*inputs, **kwargs)                               # trace_info : t_18338, t_22068, t_25796
        if mpu.is_pipeline_last_stage():                                       # trace_info : t_19392, t_23120, t_26848
            outputs = float16_to_fp32(outputs)
        return outputs                                                         # trace_info : t_19407, t_23135, t_26863


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
