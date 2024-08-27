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
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]               # trace_info : t_16339, t_19978, t_23617
        torch.distributed.all_reduce(                                          # trace_info : t_16340, t_16345, t_19979, t_19984, t_23618, ...
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()# trace_info : t_16341, t_19980, t_23619
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)# trace_info : t_16346, t_19985, t_23624

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size# trace_info : t_16347, t_19986, t_23625
        partition_vocab_size = vocab_parallel_logits.size()[-1]                # trace_info : t_16348, t_19987, t_23626
        rank = get_tensor_model_parallel_rank()                                # trace_info : t_16349, t_19988, t_23627
        world_size = get_tensor_model_parallel_world_size()                    # trace_info : t_16355, t_19994, t_23633
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)# trace_info : t_16361, t_20000, t_23639

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)# trace_info : t_16365, t_20004, t_23643
        masked_target = target.clone() - vocab_start_index                     # trace_info : t_16366, t_20005, t_23644
        masked_target[target_mask] = 0                                         # trace_info : t_16367, t_20006, t_23645

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)       # trace_info : t_16368, t_20007, t_23646
        masked_target_1d = masked_target.view(-1)                              # trace_info : t_16369, t_20008, t_23647
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)# trace_info : t_16370, t_20009, t_23648
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]           # trace_info : t_16371, t_20010, t_23649
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()         # trace_info : t_16372, t_20011, t_23650
        predicted_logits = predicted_logits_1d.view_as(target)                 # trace_info : t_16373, t_20012, t_23651
        predicted_logits[target_mask] = 0.0                                    # trace_info : t_16374, t_20013, t_23652
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_16375, t_16382, t_20014, t_20021, t_23653, ...
            predicted_logits,                                                  # trace_info : t_16376, t_20015, t_23654
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_16377, t_20016, t_23655
            group=get_tensor_model_parallel_group(),                           # trace_info : t_16378, t_20017, t_23656
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits                                     # trace_info : t_16383, t_20022, t_23661
        torch.exp(vocab_parallel_logits, out=exp_logits)                       # trace_info : t_16384, t_20023, t_23662
        sum_exp_logits = exp_logits.sum(dim=-1)                                # trace_info : t_16385, t_20024, t_23663
        torch.distributed.all_reduce(                                          # trace_info : t_16386, t_16393, t_20025, t_20032, t_23664, ...
            sum_exp_logits,                                                    # trace_info : t_16387, t_20026, t_23665
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_16388, t_20027, t_23666
            group=get_tensor_model_parallel_group(),                           # trace_info : t_16389, t_20028, t_23667
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits                    # trace_info : t_16394, t_20033, t_23672

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))                      # trace_info : t_16395, t_20034, t_23673

        vocab_size = exp_logits.size(-1)                                       # trace_info : t_16396, t_20035, t_23674
        if label_smoothing > 0:                                                # trace_info : t_16397, t_20036, t_23675
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

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size      # trace_info : t_16398, t_20037, t_23676

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)       # trace_info : t_16399, t_20038, t_23677

        return loss                                                            # trace_info : t_16400, t_20039, t_23678

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
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)# trace_info : t_16338, t_19977, t_23616
