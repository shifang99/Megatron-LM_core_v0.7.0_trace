# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import re                                                                      # trace_info : t_9875
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDelayedScaling,
    TENorm,
    get_cpu_offload_context,
    te_checkpoint,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer, TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor


def get_num_layers_to_build(config: TransformerConfig) -> int:

    num_layers_per_pipeline_rank = (                                           # trace_info : t_9869
        config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_9864
    )

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:# trace_info : t_9870
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_rank

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.

        num_layers_to_build = num_layers_per_pipeline_rank                     # trace_info : t_9872

    return num_layers_to_build                                                 # trace_info : t_9873


@dataclass
class TransformerBlockSubmodules:
    layer_specs: List[ModuleSpec] = None


def _get_block_submodules(
    config: TransformerConfig, spec: Union[TransformerBlockSubmodules, ModuleSpec],
) -> TransformerBlockSubmodules:

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):                           # trace_info : t_9859
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):                                         # trace_info : t_9860
        if issubclass(spec.module, TransformerBlock):                          # trace_info : t_9861
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):                    # trace_info : t_9862
            num_layers = get_num_layers_to_build(config)                       # trace_info : t_9863
            return TransformerBlockSubmodules(layer_specs=[spec] * num_layers) # trace_info : t_9874
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__(config=config)                                        # trace_info : t_9855

        self.submodules = _get_block_submodules(config, spec)                  # trace_info : t_9858
        self.post_layer_norm = post_layer_norm                                 # trace_info : t_9876
        self.pre_process = pre_process                                         # trace_info : t_9877
        self.post_process = post_process                                       # trace_info : t_9878
        # Dictionary to store CUDA graphs. Number of items in the dictionary = len(self.layers).
        # Item `i` in the dictionary is a list of `N` CUDA graphs for layer 'i' where N is the
        # number of microbatches. Multiple CUDA graphs per layer is required to support
        # pipelining which requires running FWD graph of multiple microbatches before BWD graph.
        self.cuda_graphs = {}                                                  # trace_info : t_9879
        self.current_microbatch = -1                                           # trace_info : t_9880

        # required for pipeline parallel schedules
        self.input_tensor = None                                               # trace_info : t_9881

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'# trace_info : t_9882

        if get_cpu_offload_context is not None:                                # trace_info : t_9883
            (                                                                  # trace_info : t_9890
                self.offload_context,                                          # trace_info : t_9891
                self.group_prefetch_offload_commit_async,                      # trace_info : t_9892
            ) = get_cpu_offload_context(                                       # trace_info : t_9884, t_9889
                self.config.cpu_offloading,                                    # trace_info : t_9885
                self.config.cpu_offloading_num_layers,                         # trace_info : t_9886
                self.config.cpu_offloading_activations,                        # trace_info : t_9887
                self.config.cpu_offloading_weights,                            # trace_info : t_9888
            )
            self.config._cpu_offloading_context = (                            # trace_info : t_9894
                self.offload_context if self.config.cpu_offloading else None   # trace_info : t_9893
            )
        else:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self._build_layers()                                                   # trace_info : t_9895
        self.num_layers_per_pipeline_rank = len(self.layers)                   # trace_info : t_11481

    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):                             # trace_info : t_9896
            return build_module(layer_spec, config=self.config, layer_number=layer_number,)# trace_info : t_9901, t_10680

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(                                     # trace_info : t_9897, t_11458
            [                                                                  # trace_info : t_9898, t_9900
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)    # trace_info : t_9899
            ]
        )

        # # TODO: add back standalone_embedding_stage
        # if self.num_layers == 0:
        #     # When a standalone embedding stage is used (e.g.,
        #     # args.standalone_embedding_stage == True), virtual pipeline ranks
        #     # on pipeline rank 0 will have zero transformer layers assigned to
        #     # them. This results in the model's input and output tensors to be
        #     # the same, which will cause failure for certain output tensor
        #     # optimizations (e.g., pipeline output deallocation). To remedy
        #     # this, we assign a 'no-op' layer on these ranks, which will
        #     # disconnect the input tensor from the output tensor.
        #     self.num_layers = 1
        #     self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        # else:
        #     self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:                         # trace_info : t_11459
            # Final layer norm before output.
            self.final_layernorm = TENorm(                                     # trace_info : t_11460, t_11464
                config=self.config,                                            # trace_info : t_11461
                hidden_size=self.config.hidden_size,                           # trace_info : t_11462
                eps=self.config.layernorm_epsilon,                             # trace_info : t_11463
            )

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(l, l + self.config.recompute_num_layers)
                )

                l += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for l in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    l >= recompute_skip_num_layers
                    and l < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(l, l + 1))
                else:
                    hidden_states, context = custom(l, l + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        packed_seq_params,
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor                                       # trace_info : t_17905, t_21091, t_24277

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

        if not self.pre_process:                                               # trace_info : t_18239, t_21425, t_24611
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(                                  # trace_info : t_18240, t_18242, t_21426, t_21428, t_24612, ...
            inp=hidden_states, requires_grad=True, keep_graph=True,            # trace_info : t_18241, t_21427, t_24613
        )

        if self.config.sequence_parallel:                                      # trace_info : t_18245, t_21431, t_24617
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()                                        # trace_info : t_18246, t_21432, t_24618

        if self.config.fp8:                                                    # trace_info : t_18247, t_21433, t_24619
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()                                        # trace_info : t_18248, t_21434, t_24620

        with rng_context and fp8_context:                                      # trace_info : t_18249, t_18666, t_21435, t_21852, t_24621, ...
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:  # trace_info : t_18250, t_21436, t_24622
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                )
            else:
                for l_no, layer in enumerate(self.layers):                     # trace_info : t_18251, t_18458, t_18665, t_21437, t_21644, ...
                    with self.offload_context:                                 # trace_info : t_18252, t_18453, t_18459, t_18660, t_21438, ...
                        if (len(self.cuda_graphs) == 0) or (not self.training):# trace_info : t_18253, t_18460, t_21439, t_21646, t_24625, ...
                            hidden_states, context = layer(                    # trace_info : t_18254, t_18262, t_18461, t_18469, t_21440, ...
                                hidden_states=hidden_states,                   # trace_info : t_18255, t_18462, t_21441, t_21648, t_24627, ...
                                attention_mask=attention_mask,                 # trace_info : t_18256, t_18463, t_21442, t_21649, t_24628, ...
                                context=context,                               # trace_info : t_18257, t_18464, t_21443, t_21650, t_24629, ...
                                context_mask=context_mask,                     # trace_info : t_18258, t_18465, t_21444, t_21651, t_24630, ...
                                rotary_pos_emb=rotary_pos_emb,                 # trace_info : t_18259, t_18466, t_21445, t_21652, t_24631, ...
                                inference_params=inference_params,             # trace_info : t_18260, t_18467, t_21446, t_21653, t_24632, ...
                                packed_seq_params=packed_seq_params,           # trace_info : t_18261, t_18468, t_21447, t_21654, t_24633, ...
                            )
                            # CUDA graph doesn't output context and is expected to be None
                            assert (
                                (context is None)                              # trace_info : t_18452, t_18659, t_21638, t_21845, t_24824, ...
                                or (not self.config.enable_cuda_graph)
                                or (not self.training)
                            )
                        else:
                            # CUDA graph replay for layer `l_no` and microbatch `self.current_microbatch`
                            # CUDA graph requires positional arguments with the exception of is_first_microbatch.
                            # Also CUDA graph accepts only Tensor inputs and outputs. Hence, the arg list and
                            # returned list is limited to `hidden_states`.
                            assert (len(self.cuda_graphs) > l_no) and (
                                self.current_microbatch < len(self.cuda_graphs[l_no])
                            )
                            hidden_states = self.cuda_graphs[l_no][self.current_microbatch](
                                hidden_states, is_first_microbatch=(self.current_microbatch == 0),
                            )

                    if (                                                       # trace_info : t_18455, t_18457, t_18662, t_18664, t_21641, ...
                        torch.is_grad_enabled()                                # trace_info : t_18454, t_18661, t_21640, t_21847, t_24826, ...
                        and self.config.cpu_offloading                         # trace_info : t_18456, t_18663, t_21642, t_21849, t_24828, ...
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.post_process and self.post_layer_norm:                         # trace_info : t_18667, t_21853, t_25039
            hidden_states = self.final_layernorm(hidden_states)                # trace_info : t_18668, t_21854, t_25040

        return hidden_states                                                   # trace_info : t_18669, t_21855, t_25041

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        assert not sharded_offsets, "Unexpected sharded offsets"
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = layer._get_layer_offset()

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict
