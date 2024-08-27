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
        import scaled_upper_triang_masked_softmax_cuda                         # trace_info : t_18588, t_19081, t_22316, t_22809, t_26044, ...

        scale_t = torch.tensor([scale])                                        # trace_info : t_18589, t_19082, t_22317, t_22810, t_26045, ...
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])# trace_info : t_18590, t_19083, t_22318, t_22811, t_26046, ...

        ctx.save_for_backward(softmax_results, scale_t)                        # trace_info : t_18591, t_19084, t_22319, t_22812, t_26047, ...
        return softmax_results                                                 # trace_info : t_18592, t_19085, t_22320, t_22813, t_26048, ...

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
        super(FusedScaleMaskSoftmax, self).__init__()                          # trace_info : t_10170, t_11172
        self.input_in_fp16 = input_in_fp16                                     # trace_info : t_10171, t_11173
        self.input_in_bf16 = input_in_bf16                                     # trace_info : t_10172, t_11174
        assert not (                                                           # trace_info : t_10174, t_10176, t_11176, t_11178
            self.input_in_fp16 and self.input_in_bf16                          # trace_info : t_10173, t_10175, t_11175, t_11177
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16       # trace_info : t_10177, t_11179
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_10178, t_11180
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion       # trace_info : t_10179, t_11181
        self.mask_func = mask_func                                             # trace_info : t_10180, t_11182
        self.softmax_in_fp32 = softmax_in_fp32                                 # trace_info : t_10181, t_11183
        self.scale = scale                                                     # trace_info : t_10182, t_11184

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"# trace_info : t_10183, t_11185

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4                                                # trace_info : t_18563, t_19056, t_22291, t_22784, t_26019, ...

        if self.is_kernel_available(mask, *input.size()):                      # trace_info : t_18564, t_19057, t_22292, t_22785, t_26020, ...
            return self.forward_fused_softmax(input, mask)                     # trace_info : t_18581, t_19074, t_22309, t_22802, t_26037, ...
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np                                                  # trace_info : t_18565, t_19058, t_22293, t_22786, t_26021, ...

        if (                                                                   # trace_info : t_18567, t_18569, t_19060, t_19062, t_22295, ...
            self.scaled_masked_softmax_fusion  # user want to fuse             # trace_info : t_18566, t_19059, t_22294, t_22787, t_26022, ...
            and self.input_in_float16  # input must be fp16                    # trace_info : t_18568, t_19061, t_22296, t_22789, t_26024, ...
            and 16 < sk <= 4096  # sk must be 16 ~ 2048                        # trace_info : t_18570, t_19063, t_22298, t_22791, t_26026, ...
            and sq % 4 == 0  # sq must be divisor of 4                         # trace_info : t_18571, t_19064, t_22299, t_22792, t_26027, ...
            and sk % 4 == 0  # sk must be divisor of 4                         # trace_info : t_18572, t_19065, t_22300, t_22793, t_26028, ...
            and attn_batches % 4 == 0  # np * b must be divisor of 4           # trace_info : t_18573, t_19066, t_22301, t_22794, t_26029, ...
        ):
            if 0 <= sk <= 4096:                                                # trace_info : t_18574, t_19067, t_22302, t_22795, t_26030, ...
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)      # trace_info : t_18575, t_19068, t_22303, t_22796, t_26031, ...

                if self.attn_mask_type == AttnMaskType.causal:                 # trace_info : t_18578, t_19071, t_22306, t_22799, t_26034, ...
                    if attn_batches % batch_per_block == 0:                    # trace_info : t_18579, t_19072, t_22307, t_22800, t_26035, ...
                        return True                                            # trace_info : t_18580, t_19073, t_22308, t_22801, t_26036, ...
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()                                           # trace_info : t_18582, t_19075, t_22310, t_22803, t_26038, ...
        scale = self.scale if self.scale is not None else 1.0                  # trace_info : t_18583, t_19076, t_22311, t_22804, t_26039, ...

        if self.attn_mask_type == AttnMaskType.causal:                         # trace_info : t_18584, t_19077, t_22312, t_22805, t_26040, ...
            assert sq == sk, "causal mask is only for self attention"          # trace_info : t_18585, t_19078, t_22313, t_22806, t_26041, ...

            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)                                     # trace_info : t_18586, t_19079, t_22314, t_22807, t_26042, ...
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)         # trace_info : t_18587, t_19080, t_22315, t_22808, t_26043, ...
            return probs.view(b, np, sq, sk)                                   # trace_info : t_18593, t_19086, t_22321, t_22814, t_26049, ...
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
        import scaled_masked_softmax_cuda                                      # trace_info : t_18576, t_19069, t_22304, t_22797, t_26032, ...

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)   # trace_info : t_18577, t_19070, t_22305, t_22798, t_26033, ...
