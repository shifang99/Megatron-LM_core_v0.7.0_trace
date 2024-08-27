# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import os
from importlib.metadata import version
from typing import Callable

import torch
import transformer_engine as te
from pkg_resources import packaging
from torch import Tensor

from megatron.core import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

_te_version = packaging.version.Version(version("transformer-engine"))


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {                                        # trace_info : t_10145, t_10243, t_10496, t_10612, t_10923, ...
        "params_dtype": config.params_dtype,                                   # trace_info : t_10144, t_10242, t_10495, t_10611, t_10922, ...
    }

    if _te_version >= packaging.version.Version("0.12.0"):                     # trace_info : t_10146, t_10244, t_10497, t_10613, t_10924, ...
        if config.use_cpu_initialization:                                      # trace_info : t_10147, t_10245, t_10498, t_10614, t_10925, ...
            extra_transformer_engine_kwargs["device"] = 'cpu'
        else:
            extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()# trace_info : t_10148, t_10246, t_10499, t_10615, t_10926, ...
    return extra_transformer_engine_kwargs                                     # trace_info : t_10149, t_10247, t_10500, t_10616, t_10927, ...


def condition_init_method(config, init_method):
    return init_method if config.perform_initialization else (lambda w: None)  # trace_info : t_10132, t_10169, t_10270, t_10523, t_10599, ...


class TENorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        if config.normalization == "LayerNorm":                                # trace_info : t_11465
            instance = te.pytorch.LayerNorm(                                   # trace_info : t_11466, t_11471, t_11479
                hidden_size=hidden_size,                                       # trace_info : t_11467
                eps=eps,                                                       # trace_info : t_11468
                sequence_parallel=config.sequence_parallel,                    # trace_info : t_11469
                zero_centered_gamma=config.layernorm_zero_centered_gamma,      # trace_info : t_11470
                **_get_extra_te_kwargs(config),                                # trace_info : t_11472
            )
        elif config.normalization == "RMSNorm":
            assert hasattr(
                te.pytorch, "RMSNorm"
            ), "Transformer-Engine >= v0.11 required to use this feature"
            instance = te.pytorch.RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance                                                        # trace_info : t_11480


class TELinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: str,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: str = None,
    ):
        self.config = config                                                   # trace_info : t_10138, t_10605, t_10916, t_11383

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias                           # trace_info : t_10139, t_10606, t_10917, t_11384
        self.is_first_microbatch = True                                        # trace_info : t_10140, t_10607, t_10918, t_11385
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache# trace_info : t_10141, t_10608, t_10919, t_11386
        if skip_weight_param_allocation:                                       # trace_info : t_10142, t_10609, t_10920, t_11387
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        extra_kwargs = _get_extra_te_kwargs(config)                            # trace_info : t_10143, t_10610, t_10921, t_11388

        if _te_version >= packaging.version.Version("0.8.0"):                  # trace_info : t_10150, t_10617, t_10928, t_11395
            if self.config.tp_comm_overlap:                                    # trace_info : t_10151, t_10618, t_10929, t_11396
                if _te_version > packaging.version.Version("1.5.0"):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    extra_kwargs["ub_overlap_rs"] = (
                        self.config.tp_comm_overlap_rs
                        if hasattr(self.config, "tp_comm_overlap_rs")
                        else self.config.tp_comm_split_rs or self.config.tp_comm_atomic_rs
                    )
                else:
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
                    extra_kwargs["ub_atomic_gemm_rs"] = self.config.tp_comm_atomic_rs
                if _te_version > packaging.version.Version("1.0.0"):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        super().__init__(                                                      # trace_info : t_10152, t_10173, t_10175, t_10619, t_10640, ...
            in_features=input_size,                                            # trace_info : t_10153, t_10620, t_10931, t_11398
            out_features=output_size,                                          # trace_info : t_10154, t_10621, t_10932, t_11399
            sequence_parallel=self.config.sequence_parallel,                   # trace_info : t_10155, t_10622, t_10933, t_11400
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,  # trace_info : t_10156, t_10623, t_10934, t_11401
            tp_group=get_tensor_model_parallel_group(check_initialized=False), # trace_info : t_10157, t_10624, t_10935, t_11402
            tp_size=self.config.tensor_model_parallel_size,                    # trace_info : t_10160, t_10627, t_10938, t_11405
            get_rng_state_tracker=get_cuda_rng_tracker                         # trace_info : t_10167, t_10634, t_10945, t_11412
            if get_cuda_rng_tracker().is_initialized()                         # trace_info : t_10161, t_10628, t_10939, t_11406
            else None,
            init_method=condition_init_method(config, init_method),            # trace_info : t_10168, t_10635, t_10946, t_11413
            bias=bias,                                                         # trace_info : t_10170, t_10637, t_10948, t_11415
            return_bias=self.te_return_bias,                                   # trace_info : t_10171, t_10638, t_10949, t_11416
            parallel_mode=parallel_mode,                                       # trace_info : t_10172, t_10639, t_10950, t_11417
            **extra_kwargs,                                                    # trace_info : t_10174, t_10641, t_10952, t_11419
        )

    def forward(self, x):
        _is_first_microbatch = (                                               # trace_info : t_18377, t_18431, t_18584, t_18638, t_21563, ...
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch# trace_info : t_18376, t_18430, t_18583, t_18637, t_21562, ...
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)     # trace_info : t_18378, t_18432, t_18585, t_18639, t_21564, ...
        self.is_first_microbatch = False                                       # trace_info : t_18379, t_18433, t_18586, t_18640, t_21565, ...

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:                                                # trace_info : t_18380, t_18434, t_18587, t_18641, t_21566, ...
            return out                                                         # trace_info : t_18381, t_18435, t_18588, t_18642, t_21567, ...
        return out, None


class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        self.config = config                                                   # trace_info : t_10234, t_10487, t_11012, t_11265

        if gather_output:                                                      # trace_info : t_10235, t_10488, t_11013, t_11266
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        if is_expert:                                                          # trace_info : t_10236, t_10489, t_11014, t_11267
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        if skip_weight_param_allocation:                                       # trace_info : t_10237, t_10490, t_11015, t_11268
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias                           # trace_info : t_10238, t_10491, t_11016, t_11269
        self.is_first_microbatch = True                                        # trace_info : t_10239, t_10492, t_11017, t_11270
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache# trace_info : t_10240, t_10493, t_11018, t_11271
        extra_kwargs = _get_extra_te_kwargs(config)                            # trace_info : t_10241, t_10494, t_11019, t_11272

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        if _te_version >= packaging.version.Version("0.11.0"):                 # trace_info : t_10248, t_10501, t_11026, t_11279
            extra_kwargs["normalization"] = self.config.normalization          # trace_info : t_10249, t_10502, t_11027, t_11280
        elif self.config.normalization != "LayerNorm":
            raise ValueError(
                f"Transformer Engine v{_te_version} does not support {self.config.normalization}."
            )

        if _te_version >= packaging.version.Version("0.8.0"):                  # trace_info : t_10250, t_10503, t_11028, t_11281
            if self.config.tp_comm_overlap:                                    # trace_info : t_10251, t_10504, t_11029, t_11282
                extra_kwargs["ub_bulk_wgrad"] = self.config.tp_comm_bulk_wgrad
                extra_kwargs["ub_bulk_dgrad"] = self.config.tp_comm_bulk_dgrad
                if _te_version > packaging.version.Version("1.5.0"):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    if _te_version > packaging.version.Version("1.6.0.dev0"):
                        extra_kwargs["ub_overlap_rs_dgrad"] = (
                            self.config.tp_comm_overlap_rs_dgrad
                            if hasattr(self.config, "tp_comm_overlap_rs_dgrad")
                            else False
                        )
                else:
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                if _te_version > packaging.version.Version("1.0.0"):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        super().__init__(                                                      # trace_info : t_10252, t_10276, t_10278, t_10505, t_10529, ...
            in_features=input_size,                                            # trace_info : t_10253, t_10506, t_11031, t_11284
            out_features=output_size,                                          # trace_info : t_10254, t_10507, t_11032, t_11285
            eps=self.config.layernorm_epsilon,                                 # trace_info : t_10255, t_10508, t_11033, t_11286
            sequence_parallel=self.config.sequence_parallel,                   # trace_info : t_10256, t_10509, t_11034, t_11287
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,  # trace_info : t_10257, t_10510, t_11035, t_11288
            tp_group=get_tensor_model_parallel_group(check_initialized=False), # trace_info : t_10258, t_10511, t_11036, t_11289
            tp_size=self.config.tensor_model_parallel_size,                    # trace_info : t_10261, t_10514, t_11039, t_11292
            get_rng_state_tracker=get_cuda_rng_tracker                         # trace_info : t_10268, t_10521, t_11046, t_11299
            if get_cuda_rng_tracker().is_initialized()                         # trace_info : t_10262, t_10515, t_11040, t_11293
            else None,
            init_method=condition_init_method(config, init_method),            # trace_info : t_10269, t_10522, t_11047, t_11300
            bias=bias,                                                         # trace_info : t_10271, t_10524, t_11049, t_11302
            return_bias=self.te_return_bias,                                   # trace_info : t_10272, t_10525, t_11050, t_11303
            parallel_mode="column",                                            # trace_info : t_10273, t_10526, t_11051, t_11304
            return_layernorm_output=False,                                     # trace_info : t_10274, t_10527, t_11052, t_11305
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,     # trace_info : t_10275, t_10528, t_11053, t_11306
            **extra_kwargs,                                                    # trace_info : t_10277, t_10530, t_11055, t_11308
        )

    def forward(self, x):
        _is_first_microbatch = (                                               # trace_info : t_18277, t_18416, t_18484, t_18623, t_21463, ...
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch# trace_info : t_18276, t_18415, t_18483, t_18622, t_21462, ...
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)     # trace_info : t_18278, t_18417, t_18485, t_18624, t_21464, ...
        self.is_first_microbatch = False                                       # trace_info : t_18279, t_18418, t_18486, t_18625, t_21465, ...

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:                                                # trace_info : t_18280, t_18419, t_18487, t_18626, t_21466, ...
            return out                                                         # trace_info : t_18420, t_18627, t_21606, t_21813, t_24792, ...
        return out, None                                                       # trace_info : t_18281, t_18488, t_21467, t_21674, t_24653, ...

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
    ):
        if not input_is_parallel:                                              # trace_info : t_10124, t_10591, t_10902, t_11369
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )

        if is_expert:                                                          # trace_info : t_10125, t_10592, t_10903, t_11370
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        super().__init__(                                                      # trace_info : t_10126, t_10137, t_10593, t_10604, t_10904, ...
            input_size=input_size,                                             # trace_info : t_10127, t_10594, t_10905, t_11372
            output_size=output_size,                                           # trace_info : t_10128, t_10595, t_10906, t_11373
            parallel_mode="row",                                               # trace_info : t_10129, t_10596, t_10907, t_11374
            config=config,                                                     # trace_info : t_10130, t_10597, t_10908, t_11375
            init_method=condition_init_method(config, init_method),            # trace_info : t_10131, t_10598, t_10909, t_11376
            bias=bias,                                                         # trace_info : t_10133, t_10600, t_10911, t_11378
            skip_bias_add=skip_bias_add,                                       # trace_info : t_10134, t_10601, t_10912, t_11379
            skip_weight_param_allocation=False,  # We don't currently use this for row parallel layers# trace_info : t_10135, t_10602, t_10913, t_11380
            tp_comm_buffer_name=tp_comm_buffer_name,                           # trace_info : t_10136, t_10603, t_10914, t_11381
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        self.config = config                                                   # trace_info : t_10045, t_10824
        self.te_forward_mask_type = False                                      # trace_info : t_10046, t_10825
        self.qkv_format: str = 'sbhd'                                          # trace_info : t_10047, t_10826

        if self.config.apply_query_key_layer_scaling != bool(                  # trace_info : t_10048, t_10050, t_10827, t_10829
            int(os.getenv('NVTE_APPLY_QK_LAYER_SCALING', '0'))                 # trace_info : t_10049, t_10828
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs = {}                                                      # trace_info : t_10051, t_10830
        if _te_version >= packaging.version.Version("0.11.0"):                 # trace_info : t_10052, t_10831
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups      # trace_info : t_10053, t_10832
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{_te_version} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if _te_version >= packaging.version.Version("0.10.0"):                 # trace_info : t_10054, t_10833
            extra_kwargs["attention_type"] = attention_type                    # trace_info : t_10055, t_10834
            # older version don't need attention_type

        if _te_version > packaging.version.Version("0.12.0"):                  # trace_info : t_10056, t_10835
            self.te_forward_mask_type = True                                   # trace_info : t_10057, t_10836

        # Only Transformer-Engine version >= 1.0.0 supports context parallelism
        if _te_version >= packaging.version.Version("1.0.0"):                  # trace_info : t_10058, t_10837
            if getattr(TEDotProductAttention, "cp_stream") is None:            # trace_info : t_10059, t_10838
                TEDotProductAttention.cp_stream = torch.cuda.Stream()          # trace_info : t_10060
            extra_kwargs["cp_group"] = get_context_parallel_group(check_initialized=False)# trace_info : t_10061, t_10839
            extra_kwargs["cp_global_ranks"] = get_context_parallel_global_ranks(# trace_info : t_10064, t_10066, t_10842, t_10844
                check_initialized=False                                        # trace_info : t_10065, t_10843
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream        # trace_info : t_10069, t_10847
        else:
            assert (
                self.config.context_parallel_size == 1
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"

        if config.window_size is not None:                                     # trace_info : t_10070, t_10848
            # Check version
            assert _te_version >= packaging.version.Version(
                "1.2.0"
            ), f"Transformer-Engine version ({str(_te_version)}) must be >= 1.2.0 to support sliding window attention."
            extra_kwargs['window_size'] = config.window_size

        super().__init__(                                                      # trace_info : t_10071, t_10090, t_10092, t_10849, t_10868, ...
            num_attention_heads=self.config.num_attention_heads,               # trace_info : t_10072, t_10850
            kv_channels=self.config.kv_channels,                               # trace_info : t_10073, t_10851
            attention_dropout=self.config.attention_dropout                    # trace_info : t_10075, t_10853
            if attention_dropout is None                                       # trace_info : t_10074, t_10852
            else attention_dropout,
            attn_mask_type=attn_mask_type.name,                                # trace_info : t_10076, t_10854
            sequence_parallel=self.config.sequence_parallel,                   # trace_info : t_10077, t_10855
            tp_size=self.config.tensor_model_parallel_size,                    # trace_info : t_10078, t_10856
            get_rng_state_tracker=get_cuda_rng_tracker                         # trace_info : t_10085, t_10863
            if get_cuda_rng_tracker().is_initialized()                         # trace_info : t_10079, t_10857
            else None,
            tp_group=get_tensor_model_parallel_group(check_initialized=False), # trace_info : t_10086, t_10864
            layer_number=layer_number,                                         # trace_info : t_10089, t_10867
            **extra_kwargs,                                                    # trace_info : t_10091, t_10869
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: PackedSeqParams = None,
    ):
        packed_seq_kwargs = (                                                  # trace_info : t_18326, t_18533, t_21512, t_21719, t_24698, ...
            dataclasses.asdict(packed_seq_params) if packed_seq_params is not None else {}# trace_info : t_18325, t_18532, t_21511, t_21718, t_24697, ...
        )
        # overwrite self.qkv_format depending on self.config.apply_rope_fusion, which can be set after init
        if self.config.apply_rope_fusion and _te_version > packaging.version.Version("0.13.0"):# trace_info : t_18327, t_18534, t_21513, t_21720, t_24699, ...
            self.qkv_format = 'bshd'                                           # trace_info : t_18328, t_18535, t_21514, t_21721, t_24700, ...

        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)      # trace_info : t_18329, t_18536, t_21515, t_21722, t_24701, ...

        if _te_version < packaging.version.Version("1.3.0"):                   # trace_info : t_18330, t_18537, t_21516, t_21723, t_24702, ...
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H copies (#555)
            # These two arguments did not exist prior to 1.3.0
            packed_seq_kwargs.pop("max_seqlen_q", None)
            packed_seq_kwargs.pop("max_seqlen_kv", None)

        if self.config.apply_rope_fusion and qkv_format == 'bshd':             # trace_info : t_18331, t_18538, t_21517, t_21724, t_24703, ...
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]# trace_info : t_18332, t_18539, t_21518, t_21725, t_24704, ...
            # In PyTorch, the following two tensors are in fact the same:
            #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
            #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
            # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
            # can have same shape but different strides.
            # We unify them to the first one to pass the stride check in TE
            if value.shape == key.shape and value.shape[0] == 1 and value.stride() != key.stride():# trace_info : t_18333, t_18540, t_21519, t_21726, t_24705, ...
                value = value.as_strided(value.shape, key.stride())

        if self.te_forward_mask_type:                                          # trace_info : t_18334, t_18541, t_21520, t_21727, t_24706, ...
            core_attn_out = super().forward(                                   # trace_info : t_18335, t_18340, t_18342, t_18344, t_18542, ...
                query,                                                         # trace_info : t_18336, t_18543, t_21522, t_21729, t_24708, ...
                key,                                                           # trace_info : t_18337, t_18544, t_21523, t_21730, t_24709, ...
                value,                                                         # trace_info : t_18338, t_18545, t_21524, t_21731, t_24710, ...
                attention_mask,                                                # trace_info : t_18339, t_18546, t_21525, t_21732, t_24711, ...
                attn_mask_type=attn_mask_type.name,                            # trace_info : t_18341, t_18548, t_21527, t_21734, t_24713, ...
                **packed_seq_kwargs,                                           # trace_info : t_18343, t_18550, t_21529, t_21736, t_24715, ...
            )
        else:
            core_attn_out = super().forward(query, key, value, attention_mask, **packed_seq_kwargs,)

        if self.config.apply_rope_fusion and qkv_format == 'bshd':             # trace_info : t_18372, t_18579, t_21558, t_21765, t_24744, ...
            return core_attn_out.transpose(0, 1)                               # trace_info : t_18373, t_18580, t_21559, t_21766, t_24745, ...
        else:
            return core_attn_out


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        fp8_format: int,
        override_linear_precision: tuple = (False, False, False),
    ):
        extra_kwargs = _get_extra_te_kwargs(config)
        if _te_version >= packaging.version.Version("1.6.0.dev0"):
            extra_kwargs["fp8_dpa"] = config.fp8_dot_product_attention
            extra_kwargs["fp8_mha"] = config.fp8_multi_head_attention

        super().__init__(
            margin=config.fp8_margin,
            interval=config.fp8_interval,
            fp8_format=fp8_format,
            amax_compute_algo=config.fp8_amax_compute_algo,
            amax_history_len=config.fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            **extra_kwargs,
        )


def te_checkpoint(
    forward_func,
    distribute_saved_activations,
    get_rng_state_tracker,
    tp_group,
    hidden_states,
    attention_mask,
    context,
    context_mask,
    rotary_pos_emb,
    packed_seq_params,
):
    from transformer_engine.pytorch.distributed import checkpoint

    if _te_version >= packaging.version.Version("1.5.0"):
        return checkpoint(
            forward_func,
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            packed_seq_params,
            distribute_saved_activations=distribute_saved_activations,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
        )
    else:
        return checkpoint(
            forward_func,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            packed_seq_params,
        )


try:

    from transformer_engine.pytorch.attention import _SplitAlongDim

    SplitAlongDim = _SplitAlongDim.apply

except ImportError:

    SplitAlongDim = None

try:

    from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context

except ImportError:

    get_cpu_offload_context = None
