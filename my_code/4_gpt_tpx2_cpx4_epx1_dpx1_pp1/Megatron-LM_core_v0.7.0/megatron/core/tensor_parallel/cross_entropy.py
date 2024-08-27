# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]               # trace_info : t_18733, t_21919, t_25105
        torch.distributed.all_reduce(                                          # trace_info : t_18734, t_18739, t_21920, t_21925, t_25106, ...
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()# trace_info : t_18735, t_21921, t_25107
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)# trace_info : t_18740, t_21926, t_25112

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size# trace_info : t_18741, t_21927, t_25113
        partition_vocab_size = vocab_parallel_logits.size()[-1]                # trace_info : t_18742, t_21928, t_25114
        rank = get_tensor_model_parallel_rank()                                # trace_info : t_18743, t_21929, t_25115
        world_size = get_tensor_model_parallel_world_size()                    # trace_info : t_18749, t_21935, t_25121
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)# trace_info : t_18755, t_21941, t_25127

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)# trace_info : t_18759, t_21945, t_25131
        masked_target = target.clone() - vocab_start_index                     # trace_info : t_18760, t_21946, t_25132
        masked_target[target_mask] = 0                                         # trace_info : t_18761, t_21947, t_25133

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)       # trace_info : t_18762, t_21948, t_25134
        masked_target_1d = masked_target.view(-1)                              # trace_info : t_18763, t_21949, t_25135
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)# trace_info : t_18764, t_21950, t_25136
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]           # trace_info : t_18765, t_21951, t_25137
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()         # trace_info : t_18766, t_21952, t_25138
        predicted_logits = predicted_logits_1d.view_as(target)                 # trace_info : t_18767, t_21953, t_25139
        predicted_logits[target_mask] = 0.0                                    # trace_info : t_18768, t_21954, t_25140
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_18769, t_18776, t_21955, t_21962, t_25141, ...
            predicted_logits,                                                  # trace_info : t_18770, t_21956, t_25142
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_18771, t_21957, t_25143
            group=get_tensor_model_parallel_group(),                           # trace_info : t_18772, t_21958, t_25144
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits                                     # trace_info : t_18777, t_21963, t_25149
        torch.exp(vocab_parallel_logits, out=exp_logits)                       # trace_info : t_18778, t_21964, t_25150
        sum_exp_logits = exp_logits.sum(dim=-1)                                # trace_info : t_18779, t_21965, t_25151
        torch.distributed.all_reduce(                                          # trace_info : t_18780, t_18787, t_21966, t_21973, t_25152, ...
            sum_exp_logits,                                                    # trace_info : t_18781, t_21967, t_25153
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_18782, t_21968, t_25154
            group=get_tensor_model_parallel_group(),                           # trace_info : t_18783, t_21969, t_25155
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits                    # trace_info : t_18788, t_21974, t_25160

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))                      # trace_info : t_18789, t_21975, t_25161

        vocab_size = exp_logits.size(-1)                                       # trace_info : t_18790, t_21976, t_25162
        if label_smoothing > 0:                                                # trace_info : t_18791, t_21977, t_25163
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size      # trace_info : t_18792, t_21978, t_25164

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)       # trace_info : t_18793, t_21979, t_25165

        return loss                                                            # trace_info : t_18794, t_21980, t_25166

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        lobal_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)# trace_info : t_18732, t_21918, t_25104
