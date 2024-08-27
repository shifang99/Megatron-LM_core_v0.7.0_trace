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
        import scaled_upper_triang_masked_softmax_cuda

        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

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
        super(FusedScaleMaskSoftmax, self).__init__()                          # trace_info : t_9758, t_10898
        self.input_in_fp16 = input_in_fp16                                     # trace_info : t_9759, t_10899
        self.input_in_bf16 = input_in_bf16                                     # trace_info : t_9760, t_10900
        assert not (                                                           # trace_info : t_9762, t_10902
            self.input_in_fp16 and self.input_in_bf16                          # trace_info : t_9761, t_10901
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16       # trace_info : t_9763, t_10903
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_9764, t_10904
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion       # trace_info : t_9765, t_10905
        self.mask_func = mask_func                                             # trace_info : t_9766, t_10906
        self.softmax_in_fp32 = softmax_in_fp32                                 # trace_info : t_9767, t_10907
        self.scale = scale                                                     # trace_info : t_9768, t_10908

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"# trace_info : t_9769, t_10909

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4                                                # trace_info : t_18522, t_19280, t_22870, t_23625, t_27215, ...

        if self.is_kernel_available(mask, *input.size()):                      # trace_info : t_18523, t_19281, t_22871, t_23626, t_27216, ...
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)                     # trace_info : t_18530, t_19288, t_22878, t_23633, t_27223, ...

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np                                                  # trace_info : t_18524, t_19282, t_22872, t_23627, t_27217, ...

        if (                                                                   # trace_info : t_18526, t_18528, t_19284, t_19286, t_22874, ...
            self.scaled_masked_softmax_fusion  # user want to fuse             # trace_info : t_18525, t_19283, t_22873, t_23628, t_27218, ...
            and self.input_in_float16  # input must be fp16                    # trace_info : t_18527, t_19285, t_22875, t_23630, t_27220, ...
            and 16 < sk <= 4096  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False                                                           # trace_info : t_18529, t_19287, t_22877, t_23632, t_27222, ...

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)
            return probs.view(b, np, sq, sk)
        else:
            # input is 4D tensor (b, np, sq, sk)
            if mask is not None:
                return ScaledMaskedSoftmax.apply(input, mask, scale)
            else:
                return ScaledSoftmax.apply(input, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:                     # trace_info : t_18531, t_19289, t_22879, t_23634, t_27224, ...
            input = input.float()

        if self.scale is not None:                                             # trace_info : t_18532, t_19290, t_22880, t_23635, t_27225, ...
            input = input * self.scale

        # Generate causal mask if not given
        sq, sk = input.size(2), input.size(3)                                  # trace_info : t_18533, t_19291, t_22881, t_23636, t_27226, ...
        if self.attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:# trace_info : t_18534, t_19292, t_22882, t_23637, t_27227, ...
            # If sq == 1 then either KV cache is used or one-element context is passed
            # so keeping mask=None in this case; subsequent code should handle it
            assert sq == sk, "causal mask is only for self attention"
            mask = get_default_causal_mask(sq)

        mask_output = self.mask_func(input, mask) if mask is not None else input# trace_info : t_18535, t_19293, t_22883, t_23638, t_27228, ...
        probs = torch.nn.Softmax(dim=-1)(mask_output)                          # trace_info : t_18538, t_19296, t_22886, t_23641, t_27231, ...

        if self.input_in_float16 and self.softmax_in_fp32:                     # trace_info : t_18539, t_19297, t_22887, t_23642, t_27232, ...
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs                                                           # trace_info : t_18540, t_19298, t_22888, t_23643, t_27233, ...

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
