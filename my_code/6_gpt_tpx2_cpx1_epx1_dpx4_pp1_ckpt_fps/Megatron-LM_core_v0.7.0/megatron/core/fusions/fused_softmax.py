# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import get_default_causal_mask


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda                         # trace_info : t_18601, t_19094, t_22238, t_22731, t_89845, ...

        scale_t = torch.tensor([scale])                                        # trace_info : t_18602, t_19095, t_22239, t_22732, t_89846, ...
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])# trace_info : t_18603, t_19096, t_22240, t_22733, t_89847, ...

        ctx.save_for_backward(softmax_results, scale_t)                        # trace_info : t_18604, t_19097, t_22241, t_22734, t_89848, ...
        return softmax_results                                                 # trace_info : t_18605, t_19098, t_22242, t_22735, t_89849, ...

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Args:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()                          # trace_info : t_9833, t_10835
        self.input_in_fp16 = input_in_fp16                                     # trace_info : t_9834, t_10836
        self.input_in_bf16 = input_in_bf16                                     # trace_info : t_9835, t_10837
        assert not (                                                           # trace_info : t_9837, t_9839, t_10839, t_10841
            self.input_in_fp16 and self.input_in_bf16                          # trace_info : t_9836, t_9838, t_10838, t_10840
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16       # trace_info : t_9840, t_10842
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_9841, t_10843
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion       # trace_info : t_9842, t_10844
        self.mask_func = mask_func                                             # trace_info : t_9843, t_10845
        self.softmax_in_fp32 = softmax_in_fp32                                 # trace_info : t_9844, t_10846
        self.scale = scale                                                     # trace_info : t_9845, t_10847

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"# trace_info : t_9846, t_10848

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4                                                # trace_info : t_18576, t_19069, t_22213, t_22706, t_89820, ...

        if self.is_kernel_available(mask, *input.size()):                      # trace_info : t_18577, t_19070, t_22214, t_22707, t_89821, ...
            return self.forward_fused_softmax(input, mask)                     # trace_info : t_18594, t_19087, t_22231, t_22724, t_89838, ...
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np                                                  # trace_info : t_18578, t_19071, t_22215, t_22708, t_89822, ...

        if (                                                                   # trace_info : t_18580, t_18582, t_19073, t_19075, t_22217, ...
            self.scaled_masked_softmax_fusion  # user want to fuse             # trace_info : t_18579, t_19072, t_22216, t_22709, t_89823, ...
            and self.input_in_float16  # input must be fp16                    # trace_info : t_18581, t_19074, t_22218, t_22711, t_89825, ...
            and 16 < sk <= 4096  # sk must be 16 ~ 2048                        # trace_info : t_18583, t_19076, t_22220, t_22713, t_89827, ...
            and sq % 4 == 0  # sq must be divisor of 4                         # trace_info : t_18584, t_19077, t_22221, t_22714, t_89828, ...
            and sk % 4 == 0  # sk must be divisor of 4                         # trace_info : t_18585, t_19078, t_22222, t_22715, t_89829, ...
            and attn_batches % 4 == 0  # np * b must be divisor of 4           # trace_info : t_18586, t_19079, t_22223, t_22716, t_89830, ...
        ):
            if 0 <= sk <= 4096:                                                # trace_info : t_18587, t_19080, t_22224, t_22717, t_89831, ...
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)      # trace_info : t_18588, t_19081, t_22225, t_22718, t_89832, ...

                if self.attn_mask_type == AttnMaskType.causal:                 # trace_info : t_18591, t_19084, t_22228, t_22721, t_89835, ...
                    if attn_batches % batch_per_block == 0:                    # trace_info : t_18592, t_19085, t_22229, t_22722, t_89836, ...
                        return True                                            # trace_info : t_18593, t_19086, t_22230, t_22723, t_89837, ...
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()                                           # trace_info : t_18595, t_19088, t_22232, t_22725, t_89839, ...
        scale = self.scale if self.scale is not None else 1.0                  # trace_info : t_18596, t_19089, t_22233, t_22726, t_89840, ...

        if self.attn_mask_type == AttnMaskType.causal:                         # trace_info : t_18597, t_19090, t_22234, t_22727, t_89841, ...
            assert sq == sk, "causal mask is only for self attention"          # trace_info : t_18598, t_19091, t_22235, t_22728, t_89842, ...

            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)                                     # trace_info : t_18599, t_19092, t_22236, t_22729, t_89843, ...
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)         # trace_info : t_18600, t_19093, t_22237, t_22730, t_89844, ...
            return probs.view(b, np, sq, sk)                                   # trace_info : t_18606, t_19099, t_22243, t_22736, t_89850, ...
        else:
            # input is 4D tensor (b, np, sq, sk)
            if mask is not None:
                return ScaledMaskedSoftmax.apply(input, mask, scale)
            else:
                return ScaledSoftmax.apply(input, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale

        # Generate causal mask if not given
        sq, sk = input.size(2), input.size(3)
        if self.attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:
            # If sq == 1 then either KV cache is used or one-element context is passed
            # so keeping mask=None in this case; subsequent code should handle it
            assert sq == sk, "causal mask is only for self attention"
            mask = get_default_causal_mask(sq)

        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda                                      # trace_info : t_18589, t_19082, t_22226, t_22719, t_89833, ...

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)   # trace_info : t_18590, t_19083, t_22227, t_22720, t_89834, ...
