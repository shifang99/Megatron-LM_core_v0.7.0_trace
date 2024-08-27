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
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]               # trace_info : t_21172, t_24782, t_28392
        torch.distributed.all_reduce(                                          # trace_info : t_21173, t_21178, t_24783, t_24788, t_28393, ...
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()# trace_info : t_21174, t_24784, t_28394
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)# trace_info : t_21179, t_24789, t_28399

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size# trace_info : t_21180, t_24790, t_28400
        partition_vocab_size = vocab_parallel_logits.size()[-1]                # trace_info : t_21181, t_24791, t_28401
        rank = get_tensor_model_parallel_rank()                                # trace_info : t_21182, t_24792, t_28402
        world_size = get_tensor_model_parallel_world_size()                    # trace_info : t_21188, t_24798, t_28408
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)# trace_info : t_21194, t_24804, t_28414

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)# trace_info : t_21198, t_24808, t_28418
        masked_target = target.clone() - vocab_start_index                     # trace_info : t_21199, t_24809, t_28419
        masked_target[target_mask] = 0                                         # trace_info : t_21200, t_24810, t_28420

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)       # trace_info : t_21201, t_24811, t_28421
        masked_target_1d = masked_target.view(-1)                              # trace_info : t_21202, t_24812, t_28422
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)# trace_info : t_21203, t_24813, t_28423
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]           # trace_info : t_21204, t_24814, t_28424
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()         # trace_info : t_21205, t_24815, t_28425
        predicted_logits = predicted_logits_1d.view_as(target)                 # trace_info : t_21206, t_24816, t_28426
        predicted_logits[target_mask] = 0.0                                    # trace_info : t_21207, t_24817, t_28427
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_21208, t_21215, t_24818, t_24825, t_28428, ...
            predicted_logits,                                                  # trace_info : t_21209, t_24819, t_28429
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_21210, t_24820, t_28430
            group=get_tensor_model_parallel_group(),                           # trace_info : t_21211, t_24821, t_28431
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits                                     # trace_info : t_21216, t_24826, t_28436
        torch.exp(vocab_parallel_logits, out=exp_logits)                       # trace_info : t_21217, t_24827, t_28437
        sum_exp_logits = exp_logits.sum(dim=-1)                                # trace_info : t_21218, t_24828, t_28438
        torch.distributed.all_reduce(                                          # trace_info : t_21219, t_21226, t_24829, t_24836, t_28439, ...
            sum_exp_logits,                                                    # trace_info : t_21220, t_24830, t_28440
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_21221, t_24831, t_28441
            group=get_tensor_model_parallel_group(),                           # trace_info : t_21222, t_24832, t_28442
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits                    # trace_info : t_21227, t_24837, t_28447

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))                      # trace_info : t_21228, t_24838, t_28448

        vocab_size = exp_logits.size(-1)                                       # trace_info : t_21229, t_24839, t_28449
        if label_smoothing > 0:                                                # trace_info : t_21230, t_24840, t_28450
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

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size      # trace_info : t_21231, t_24841, t_28451

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)       # trace_info : t_21232, t_24842, t_28452

        return loss                                                            # trace_info : t_21233, t_24843, t_28453

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
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)# trace_info : t_21171, t_24781, t_28391
