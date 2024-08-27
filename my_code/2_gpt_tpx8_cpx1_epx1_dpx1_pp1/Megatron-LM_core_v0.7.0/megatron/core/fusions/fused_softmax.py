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
        super(FusedScaleMaskSoftmax, self).__init__()                          # trace_info : t_11671, t_12673
        self.input_in_fp16 = input_in_fp16                                     # trace_info : t_11672, t_12674
        self.input_in_bf16 = input_in_bf16                                     # trace_info : t_11673, t_12675
        assert not (                                                           # trace_info : t_11675, t_11677, t_12677, t_12679
            self.input_in_fp16 and self.input_in_bf16                          # trace_info : t_11674, t_11676, t_12676, t_12678
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16       # trace_info : t_11678, t_12680
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_11679, t_12681
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion       # trace_info : t_11680, t_12682
        self.mask_func = mask_func                                             # trace_info : t_11681, t_12683
        self.softmax_in_fp32 = softmax_in_fp32                                 # trace_info : t_11682, t_12684
        self.scale = scale                                                     # trace_info : t_11683, t_12685

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"# trace_info : t_11684, t_12686

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4                                                # trace_info : t_20297, t_20782, t_23907, t_24392, t_27517, ...

        if self.is_kernel_available(mask, *input.size()):                      # trace_info : t_20298, t_20783, t_23908, t_24393, t_27518, ...
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)                     # trace_info : t_20309, t_20794, t_23919, t_24404, t_27529, ...

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np                                                  # trace_info : t_20299, t_20784, t_23909, t_24394, t_27519, ...

        if (                                                                   # trace_info : t_20301, t_20303, t_20786, t_20788, t_23911, ...
            self.scaled_masked_softmax_fusion  # user want to fuse             # trace_info : t_20300, t_20785, t_23910, t_24395, t_27520, ...
            and self.input_in_float16  # input must be fp16                    # trace_info : t_20302, t_20787, t_23912, t_24397, t_27522, ...
            and 16 < sk <= 4096  # sk must be 16 ~ 2048                        # trace_info : t_20304, t_20789, t_23914, t_24399, t_27524, ...
            and sq % 4 == 0  # sq must be divisor of 4                         # trace_info : t_20305, t_20790, t_23915, t_24400, t_27525, ...
            and sk % 4 == 0  # sk must be divisor of 4                         # trace_info : t_20306, t_20791, t_23916, t_24401, t_27526, ...
            and attn_batches % 4 == 0  # np * b must be divisor of 4           # trace_info : t_20307, t_20792, t_23917, t_24402, t_27527, ...
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False                                                           # trace_info : t_20308, t_20793, t_23918, t_24403, t_27528, ...

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
        if self.input_in_float16 and self.softmax_in_fp32:                     # trace_info : t_20310, t_20795, t_23920, t_24405, t_27530, ...
            input = input.float()

        if self.scale is not None:                                             # trace_info : t_20311, t_20796, t_23921, t_24406, t_27531, ...
            input = input * self.scale

        # Generate causal mask if not given
        sq, sk = input.size(2), input.size(3)                                  # trace_info : t_20312, t_20797, t_23922, t_24407, t_27532, ...
        if self.attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:# trace_info : t_20313, t_20798, t_23923, t_24408, t_27533, ...
            # If sq == 1 then either KV cache is used or one-element context is passed
            # so keeping mask=None in this case; subsequent code should handle it
            assert sq == sk, "causal mask is only for self attention"
            mask = get_default_causal_mask(sq)

        mask_output = self.mask_func(input, mask) if mask is not None else input# trace_info : t_20314, t_20799, t_23924, t_24409, t_27534, ...
        probs = torch.nn.Softmax(dim=-1)(mask_output)                          # trace_info : t_20317, t_20802, t_23927, t_24412, t_27537, ...

        if self.input_in_float16 and self.softmax_in_fp32:                     # trace_info : t_20318, t_20803, t_23928, t_24413, t_27538, ...
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs                                                           # trace_info : t_20319, t_20804, t_23929, t_24414, t_27539, ...

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
