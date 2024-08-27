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
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]               # trace_info : t_19467, t_23104, t_90711
        torch.distributed.all_reduce(                                          # trace_info : t_19468, t_19473, t_23105, t_23110, t_90712, ...
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()# trace_info : t_19469, t_23106, t_90713
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)# trace_info : t_19474, t_23111, t_90718

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size# trace_info : t_19475, t_23112, t_90719
        partition_vocab_size = vocab_parallel_logits.size()[-1]                # trace_info : t_19476, t_23113, t_90720
        rank = get_tensor_model_parallel_rank()                                # trace_info : t_19477, t_23114, t_90721
        world_size = get_tensor_model_parallel_world_size()                    # trace_info : t_19483, t_23120, t_90727
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)# trace_info : t_19489, t_23126, t_90733

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)# trace_info : t_19493, t_23130, t_90737
        masked_target = target.clone() - vocab_start_index                     # trace_info : t_19494, t_23131, t_90738
        masked_target[target_mask] = 0                                         # trace_info : t_19495, t_23132, t_90739

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)       # trace_info : t_19496, t_23133, t_90740
        masked_target_1d = masked_target.view(-1)                              # trace_info : t_19497, t_23134, t_90741
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)# trace_info : t_19498, t_23135, t_90742
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]           # trace_info : t_19499, t_23136, t_90743
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()         # trace_info : t_19500, t_23137, t_90744
        predicted_logits = predicted_logits_1d.view_as(target)                 # trace_info : t_19501, t_23138, t_90745
        predicted_logits[target_mask] = 0.0                                    # trace_info : t_19502, t_23139, t_90746
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_19503, t_19510, t_23140, t_23147, t_90747, ...
            predicted_logits,                                                  # trace_info : t_19504, t_23141, t_90748
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_19505, t_23142, t_90749
            group=get_tensor_model_parallel_group(),                           # trace_info : t_19506, t_23143, t_90750
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits                                     # trace_info : t_19511, t_23148, t_90755
        torch.exp(vocab_parallel_logits, out=exp_logits)                       # trace_info : t_19512, t_23149, t_90756
        sum_exp_logits = exp_logits.sum(dim=-1)                                # trace_info : t_19513, t_23150, t_90757
        torch.distributed.all_reduce(                                          # trace_info : t_19514, t_19521, t_23151, t_23158, t_90758, ...
            sum_exp_logits,                                                    # trace_info : t_19515, t_23152, t_90759
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_19516, t_23153, t_90760
            group=get_tensor_model_parallel_group(),                           # trace_info : t_19517, t_23154, t_90761
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits                    # trace_info : t_19522, t_23159, t_90766

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))                      # trace_info : t_19523, t_23160, t_90767

        vocab_size = exp_logits.size(-1)                                       # trace_info : t_19524, t_23161, t_90768
        if label_smoothing > 0:                                                # trace_info : t_19525, t_23162, t_90769
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

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size      # trace_info : t_19526, t_23163, t_90770

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)       # trace_info : t_19527, t_23164, t_90771

        return loss                                                            # trace_info : t_19528, t_23165, t_90772

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
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)# trace_info : t_19466, t_23103, t_90710
