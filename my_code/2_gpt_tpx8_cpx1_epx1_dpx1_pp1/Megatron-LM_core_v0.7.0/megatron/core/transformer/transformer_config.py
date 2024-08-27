# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import types                                                                   # trace_info : t_11030
from dataclasses import dataclass                                              # trace_info : t_11031
from typing import Callable, Optional, Tuple                                   # trace_info : t_11032
                                                                               # trace_info : t_11033
import torch                                                                   # trace_info : t_11034
import torch.nn.functional as F                                                # trace_info : t_11035
                                                                               # trace_info : t_11036
from ..model_parallel_config import ModelParallelConfig                        # trace_info : t_11037
from ..utils import init_method_normal, scaled_init_method_normal              # trace_info : t_11038
                                                                               # trace_info : t_11039
                                                                               # trace_info : t_11040
@dataclass                                                                     # trace_info : t_11041
class TransformerConfig(ModelParallelConfig):                                  # trace_info : t_11042
    """Configuration object for megatron-core transformers.                    # trace_info : t_11043
                                                                               # trace_info : t_11044
    The initialization function has an argument for each parameter, including those in ModelParallelConfig.# trace_info : t_11045
    """                                                                        # trace_info : t_11046
                                                                               # trace_info : t_11047
    ####################                                                       # trace_info : t_11048
    # model architecture                                                       # trace_info : t_11049
    ####################                                                       # trace_info : t_11050
    num_layers: int = 0                                                        # trace_info : t_11051
    """Number of transformer layers in a transformer block."""                 # trace_info : t_11052
                                                                               # trace_info : t_11053
    hidden_size: int = 0                                                       # trace_info : t_11054
    """Transformer hidden size."""                                             # trace_info : t_11055
                                                                               # trace_info : t_11056
    num_attention_heads: int = 0                                               # trace_info : t_11057
    """Number of transformer attention heads."""                               # trace_info : t_11058
                                                                               # trace_info : t_11059
    num_query_groups: int = None                                               # trace_info : t_11060
    """Number of query groups for group query attention. If None, normal attention is used."""# trace_info : t_11061
                                                                               # trace_info : t_11062
    ffn_hidden_size: int = None                                                # trace_info : t_11063
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size if not provided."""# trace_info : t_11064
                                                                               # trace_info : t_11065
    kv_channels: int = None                                                    # trace_info : t_11066
    """Projection weights dimension in multi-head attention. This is set to hidden_size //# trace_info : t_11067
    num_attention_heads if not provided."""                                    # trace_info : t_11068
                                                                               # trace_info : t_11069
    hidden_dropout: float = 0.1                                                # trace_info : t_11070
    """Dropout probability for transformer hidden state."""                    # trace_info : t_11071
                                                                               # trace_info : t_11072
    attention_dropout: float = 0.1                                             # trace_info : t_11073
    """Post attention dropout probability."""                                  # trace_info : t_11074
                                                                               # trace_info : t_11075
    fp32_residual_connection: bool = False                                     # trace_info : t_11076
    """If true, move residual connections to fp32."""                          # trace_info : t_11077
                                                                               # trace_info : t_11078
    # @jcasper should we keep this option?                                     # trace_info : t_11079
    apply_residual_connection_post_layernorm: bool = False                     # trace_info : t_11080
    """If True, uses the original BERT residule connection ordering."""        # trace_info : t_11081
                                                                               # trace_info : t_11082
    layernorm_epsilon: float = 1e-5                                            # trace_info : t_11083
    """Epsilon value for any LayerNorm operations."""                          # trace_info : t_11084
                                                                               # trace_info : t_11085
    layernorm_zero_centered_gamma: bool = False                                # trace_info : t_11086
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves# trace_info : t_11087
    numerical stability."""                                                    # trace_info : t_11088
                                                                               # trace_info : t_11089
    add_bias_linear: bool = True                                               # trace_info : t_11090
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in# trace_info : t_11091
    MLP layer)."""                                                             # trace_info : t_11092
                                                                               # trace_info : t_11093
    add_qkv_bias: bool = False                                                 # trace_info : t_11094
    """Add a bias term only for QKV projections."""                            # trace_info : t_11095
                                                                               # trace_info : t_11096
    gated_linear_unit: bool = False                                            # trace_info : t_11097
    """Use a gated linear unit for the first linear layer in the MLP."""       # trace_info : t_11098
                                                                               # trace_info : t_11099
    activation_func: Callable = F.gelu                                         # trace_info : t_11100
    """Activation function to use for the non-linearity in the MLP."""         # trace_info : t_11101
                                                                               # trace_info : t_11102
    activation_func_fp8_input_store: bool = False                              # trace_info : t_11103
    """Store the input of MLP activation function in FP8 for backprop to save memory.# trace_info : t_11104
    The stored input is casted back to the original precision before backprop compuatation."""# trace_info : t_11105
                                                                               # trace_info : t_11106
    num_moe_experts: int = None                                                # trace_info : t_11107
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None# trace_info : t_11108
    for no MoE."""                                                             # trace_info : t_11109
                                                                               # trace_info : t_11110
    rotary_interleaved: bool = False                                           # trace_info : t_11111
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of# trace_info : t_11112
    first half and second half (LLaMa style). Default to False."""             # trace_info : t_11113
                                                                               # trace_info : t_11114
    window_size: Optional[Tuple[int, int]] = None                              # trace_info : t_11115
    """If not None, then will use sliding window attention. The size of the window is specified by# trace_info : t_11116
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""# trace_info : t_11117
                                                                               # trace_info : t_11118
    normalization: bool = "LayerNorm"                                          # trace_info : t_11119
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""# trace_info : t_11120
                                                                               # trace_info : t_11121
    qk_layernorm: bool = False                                                 # trace_info : t_11122
    """Whether to apply LayerNorm to the query and key embeddings."""          # trace_info : t_11123
                                                                               # trace_info : t_11124
    test_mode: bool = False                                                    # trace_info : t_11125
    """Whether to run real-time tests."""                                      # trace_info : t_11126
                                                                               # trace_info : t_11127
    calculate_per_token_loss: bool = False                                     # trace_info : t_11128
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the# trace_info : t_11129
    global batch, versus the default behavior of assuming all tokens are non-padded."""# trace_info : t_11130
                                                                               # trace_info : t_11131
    ####################                                                       # trace_info : t_11132
    # initialization                                                           # trace_info : t_11133
    ####################                                                       # trace_info : t_11134
    init_method: Callable = None                                               # trace_info : t_11135
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that# trace_info : t_11136
    takes a single Tensor and initializes it. If None, will be set to          # trace_info : t_11137
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with# trace_info : t_11138
    mean=0.0 and std=init_method_std."""                                       # trace_info : t_11139
                                                                               # trace_info : t_11140
    output_layer_init_method: Callable = None                                  # trace_info : t_11141
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,# trace_info : t_11142
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn# trace_info : t_11143
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""# trace_info : t_11144

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel. This kernel only supports a fixed set
    of hidden sizes."""

    memory_efficient_layer_norm: bool = False
    """If True, and using local layers (not from TransformerEngine), tells Apex to use the memory
    efficient fused LayerNorm kernel. Ignored if not using LayerNorm."""

    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: str = None
    recompute_granularity: str = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where only the memory intensive part of attention is checkpointed.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: str = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: int = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    distribute_saved_activations: bool = None
    """If True, distribute recomputed activations across the model parallel group."""

    ####################
    # fp8 related
    ####################
    fp8: str = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """Controls how often the scaling factor is recomputed."""

    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.

    """

    fp8_wgrad: bool = True
    """When set to False, override FP8 config options and do the wgrad computation in higher precision."""

    fp8_dot_product_attention: bool = False
    """When set to True, use the FP8 implementation of Dot Product Attention."""

    fp8_multi_head_attention: bool = False
    """When set to True, use the FP8 implementation of Multi Head Attention."""

    ####################
    # MoE related
    ####################
    moe_router_load_balancing_type: str = "aux_loss"
    """Determines the load balancing strategy for the router. "aux_loss" corresponds to the load
    balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing
    algorithm used in S-BASE, and "none" implies no load balancing."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).

    """

    moe_aux_loss_coeff: float = 0  # 1e-2 would be a good start value for load balance loss.
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: float = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: float = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    moe_token_dropping: bool = False  # TODO: Support token dropping.
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather' and 'alltoall'."""
    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: float = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False."""

    moe_token_drop_policy: str = 'probs'
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
    """
    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    enable_cuda_graph: bool = False
    """When set to true, TransformerLayer blocks are wrapped with CUDA graph."""

    # These 2 attributes are WAR for TRTLLM export. DO NOT USE!! WILL BE DEPRECATED SOON!!
    max_position_embeddings: int = 0
    """Deprecated. Do not use."""

    rotary_percent: float = 0
    """Deprecated. Do not use."""

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        super().__post_init__()                                                # trace_info : t_11145
        if self.fp16 and self.bf16:                                            # trace_info : t_11153
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:    # trace_info : t_11154
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:                                       # trace_info : t_11155
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:                                           # trace_info : t_11156
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:                                      # trace_info : t_11157
            self.num_query_groups = self.num_attention_heads                   # trace_info : t_11158

        if self.num_query_groups % self.tensor_model_parallel_size != 0:       # trace_info : t_11159
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:                                 # trace_info : t_11160
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:# trace_info : t_11161
            raise ValueError(f'num_moe_experts must be non None to use expert-parallel.')

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:     # trace_info : t_11162
            raise ValueError(f'num_moe_experts must be non-negative.')

        if self.moe_expert_capacity_factor is not None:                        # trace_info : t_11163
            if self.moe_token_dispatcher_type != "alltoall":
                raise ValueError(
                    f'moe_expert_capacity_factor only works with alltoall token dispatcher'
                )
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["aux_loss", "none"]:
                raise ValueError(
                    f'moe_expert_capacity_factor only works with aux_loss or none load balancing'
                )

        if self.moe_pad_expert_input_to_capacity:                              # trace_info : t_11164
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    f'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
                )

        if self.cpu_offloading and (                                           # trace_info : t_11165
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f'CPU offloading can be done only for layers less than {self.num_layers}'
            )

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:      # trace_info : t_11166
            raise ValueError(
                f'Currently there is no support for Pipeline parallelism with CPU offloading'
            )

        if self.cpu_offloading and self.recompute_granularity is not None:     # trace_info : t_11167
            raise ValueError(
                f'CPU offloading does not work when activation recomputation is enabled'
            )

        if self.recompute_granularity is not None:                             # trace_info : t_11168
            if not self.recompute_granularity in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
                )

            if self.recompute_method is not None:
                if not self.recompute_method in ['block', 'uniform']:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != 'selective':
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be between '
                    f'1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}'
                )
            elif (
                self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be None.'
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel}'
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f'num_layers: {self.num_layers} must be divisible by virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
                    )

        if self.apply_query_key_layer_scaling:                                 # trace_info : t_11169
            self.attention_softmax_in_fp32 = True

        if self.bias_activation_fusion:                                        # trace_info : t_11170
            if self.activation_func not in [F.gelu, F.silu]:                   # trace_info : t_11171
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either gelu or swiglu"
                )
            if (
                self.activation_func == F.gelu                                 # trace_info : t_11172, t_11174, t_11176
                and not self.gated_linear_unit                                 # trace_info : t_11173
                and not self.add_bias_linear                                   # trace_info : t_11175
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False, "
                    "and activation function is gelu, add_bias_linear must also be True."
                )
        if self.activation_func_fp8_input_store:                               # trace_info : t_11177
            if self.activation_func != F.silu or not self.gated_linear_unit:
                raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")
        if self.apply_rope_fusion and self.rotary_interleaved:                 # trace_info : t_11178
            raise ValueError(f'rotary_interleaved does not work with apply_rope_fusion.')

        if self.init_method is None:                                           # trace_info : t_11179
            self.init_method = init_method_normal(self.init_method_std)        # trace_info : t_11180

        if self.output_layer_init_method is None:                              # trace_info : t_11183
            self.output_layer_init_method = scaled_init_method_normal(         # trace_info : t_11184, t_11186
                self.init_method_std, self.num_layers                          # trace_info : t_11185
            )

        if self.moe_extended_tp:                                               # trace_info : t_11190
            if self.moe_token_dispatcher_type != 'allgather':
                raise ValueError(
                    "Moe extended TP parallelism only applies to allgather based token dispatcher."
                )
            extended_tp_size = self.tensor_model_parallel_size * self.expert_model_parallel_size
            if self.ffn_hidden_size % extended_tp_size != 0:
                raise ValueError(
                    f'ffn_hidden_size: {self.ffn_hidden_size} must be divisible by extended_tp_size {extended_tp_size}'
                )
