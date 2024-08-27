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
        import scaled_upper_triang_masked_softmax_cuda                         # trace_info : t_15457, t_15958, t_19096, t_19597, t_22735, ...

        scale_t = torch.tensor([scale])                                        # trace_info : t_15458, t_15959, t_19097, t_19598, t_22736, ...
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])# trace_info : t_15459, t_15960, t_19098, t_19599, t_22737, ...

        ctx.save_for_backward(softmax_results, scale_t)                        # trace_info : t_15460, t_15961, t_19099, t_19600, t_22738, ...
        return softmax_results                                                 # trace_info : t_15461, t_15962, t_19100, t_19601, t_22739, ...

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
        super(FusedScaleMaskSoftmax, self).__init__()                          # trace_info : t_6804, t_7806
        self.input_in_fp16 = input_in_fp16                                     # trace_info : t_6805, t_7807
        self.input_in_bf16 = input_in_bf16                                     # trace_info : t_6806, t_7808
        assert not (                                                           # trace_info : t_6808, t_6810, t_7810, t_7812
            self.input_in_fp16 and self.input_in_bf16                          # trace_info : t_6807, t_6809, t_7809, t_7811
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16       # trace_info : t_6811, t_7813
        self.attn_mask_type = attn_mask_type                                   # trace_info : t_6812, t_7814
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion       # trace_info : t_6813, t_7815
        self.mask_func = mask_func                                             # trace_info : t_6814, t_7816
        self.softmax_in_fp32 = softmax_in_fp32                                 # trace_info : t_6815, t_7817
        self.scale = scale                                                     # trace_info : t_6816, t_7818

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"# trace_info : t_6817, t_7819

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4                                                # trace_info : t_15432, t_15933, t_19071, t_19572, t_22710, ...

        if self.is_kernel_available(mask, *input.size()):                      # trace_info : t_15433, t_15934, t_19072, t_19573, t_22711, ...
            return self.forward_fused_softmax(input, mask)                     # trace_info : t_15450, t_15951, t_19089, t_19590, t_22728, ...
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np                                                  # trace_info : t_15434, t_15935, t_19073, t_19574, t_22712, ...

        if (                                                                   # trace_info : t_15436, t_15438, t_15937, t_15939, t_19075, ...
            self.scaled_masked_softmax_fusion  # user want to fuse             # trace_info : t_15435, t_15936, t_19074, t_19575, t_22713, ...
            and self.input_in_float16  # input must be fp16                    # trace_info : t_15437, t_15938, t_19076, t_19577, t_22715, ...
            and 16 < sk <= 4096  # sk must be 16 ~ 2048                        # trace_info : t_15439, t_15940, t_19078, t_19579, t_22717, ...
            and sq % 4 == 0  # sq must be divisor of 4                         # trace_info : t_15440, t_15941, t_19079, t_19580, t_22718, ...
            and sk % 4 == 0  # sk must be divisor of 4                         # trace_info : t_15441, t_15942, t_19080, t_19581, t_22719, ...
            and attn_batches % 4 == 0  # np * b must be divisor of 4           # trace_info : t_15442, t_15943, t_19081, t_19582, t_22720, ...
        ):
            if 0 <= sk <= 4096:                                                # trace_info : t_15443, t_15944, t_19082, t_19583, t_22721, ...
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)      # trace_info : t_15444, t_15945, t_19083, t_19584, t_22722, ...

                if self.attn_mask_type == AttnMaskType.causal:                 # trace_info : t_15447, t_15948, t_19086, t_19587, t_22725, ...
                    if attn_batches % batch_per_block == 0:                    # trace_info : t_15448, t_15949, t_19087, t_19588, t_22726, ...
                        return True                                            # trace_info : t_15449, t_15950, t_19088, t_19589, t_22727, ...
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()                                           # trace_info : t_15451, t_15952, t_19090, t_19591, t_22729, ...
        scale = self.scale if self.scale is not None else 1.0                  # trace_info : t_15452, t_15953, t_19091, t_19592, t_22730, ...

        if self.attn_mask_type == AttnMaskType.causal:                         # trace_info : t_15453, t_15954, t_19092, t_19593, t_22731, ...
            assert sq == sk, "causal mask is only for self attention"          # trace_info : t_15454, t_15955, t_19093, t_19594, t_22732, ...

            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)                                     # trace_info : t_15455, t_15956, t_19094, t_19595, t_22733, ...
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)         # trace_info : t_15456, t_15957, t_19095, t_19596, t_22734, ...
            return probs.view(b, np, sq, sk)                                   # trace_info : t_15462, t_15963, t_19101, t_19602, t_22740, ...
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
        import scaled_masked_softmax_cuda                                      # trace_info : t_15445, t_15946, t_19084, t_19585, t_22723, ...

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)   # trace_info : t_15446, t_15947, t_19085, t_19586, t_22724, ...
