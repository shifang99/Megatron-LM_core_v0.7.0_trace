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
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]               # trace_info : t_19940, t_24285, t_28630
        torch.distributed.all_reduce(                                          # trace_info : t_19941, t_19946, t_24286, t_24291, t_28631, ...
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()# trace_info : t_19942, t_24287, t_28632
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)# trace_info : t_19947, t_24292, t_28637

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size# trace_info : t_19948, t_24293, t_28638
        partition_vocab_size = vocab_parallel_logits.size()[-1]                # trace_info : t_19949, t_24294, t_28639
        rank = get_tensor_model_parallel_rank()                                # trace_info : t_19950, t_24295, t_28640
        world_size = get_tensor_model_parallel_world_size()                    # trace_info : t_19956, t_24301, t_28646
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)# trace_info : t_19962, t_24307, t_28652

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)# trace_info : t_19966, t_24311, t_28656
        masked_target = target.clone() - vocab_start_index                     # trace_info : t_19967, t_24312, t_28657
        masked_target[target_mask] = 0                                         # trace_info : t_19968, t_24313, t_28658

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)       # trace_info : t_19969, t_24314, t_28659
        masked_target_1d = masked_target.view(-1)                              # trace_info : t_19970, t_24315, t_28660
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)# trace_info : t_19971, t_24316, t_28661
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]           # trace_info : t_19972, t_24317, t_28662
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()         # trace_info : t_19973, t_24318, t_28663
        predicted_logits = predicted_logits_1d.view_as(target)                 # trace_info : t_19974, t_24319, t_28664
        predicted_logits[target_mask] = 0.0                                    # trace_info : t_19975, t_24320, t_28665
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(                                          # trace_info : t_19976, t_19983, t_24321, t_24328, t_28666, ...
            predicted_logits,                                                  # trace_info : t_19977, t_24322, t_28667
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_19978, t_24323, t_28668
            group=get_tensor_model_parallel_group(),                           # trace_info : t_19979, t_24324, t_28669
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits                                     # trace_info : t_19984, t_24329, t_28674
        torch.exp(vocab_parallel_logits, out=exp_logits)                       # trace_info : t_19985, t_24330, t_28675
        sum_exp_logits = exp_logits.sum(dim=-1)                                # trace_info : t_19986, t_24331, t_28676
        torch.distributed.all_reduce(                                          # trace_info : t_19987, t_19994, t_24332, t_24339, t_28677, ...
            sum_exp_logits,                                                    # trace_info : t_19988, t_24333, t_28678
            op=torch.distributed.ReduceOp.SUM,                                 # trace_info : t_19989, t_24334, t_28679
            group=get_tensor_model_parallel_group(),                           # trace_info : t_19990, t_24335, t_28680
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits                    # trace_info : t_19995, t_24340, t_28685

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))                      # trace_info : t_19996, t_24341, t_28686

        vocab_size = exp_logits.size(-1)                                       # trace_info : t_19997, t_24342, t_28687
        if label_smoothing > 0:                                                # trace_info : t_19998, t_24343, t_28688
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

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size      # trace_info : t_19999, t_24344, t_28689

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)       # trace_info : t_20000, t_24345, t_28690

        return loss                                                            # trace_info : t_20001, t_24346, t_28691

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
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)# trace_info : t_19939, t_24284, t_28629
