# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from logging import getLogger
from typing import Dict, Optional

import torch

from .. import parallel_state
from ..transformer.module import MegatronModule
from ..transformer.transformer_config import TransformerConfig
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import ParamAndGradBuffer

logger = getLogger(__name__)


class DistributedDataParallel(MegatronModule):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        data_parallel_group: Data-parallel process group.
        expert_data_parallel_group: Optional data-parallel process group for experts in a MoE.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        check_for_nan_in_grad: If true, check if local grad norm is NaN.

    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        disable_bucketing: bool = False,
    ):
        super().__init__(config=config)                                        # trace_info : t_12213
        self.module = module                                                   # trace_info : t_12216

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:                                     # trace_info : t_12217
            dp_size = parallel_state.get_data_parallel_world_size()            # trace_info : t_12218
            ddp_config.bucket_size = max(40000000, 1000000 * dp_size)          # trace_info : t_12226
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:                                 # trace_info : t_12227
            ddp_config.bucket_size = None                                      # trace_info : t_12228

        self.ddp_config = ddp_config                                           # trace_info : t_12229
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_12230
            logger.info(f'Setting up DistributedDataParallel with config {self.ddp_config}')# trace_info : t_12231

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size                         # trace_info : t_12237
        if parallel_state.get_pipeline_model_parallel_rank() > 0:              # trace_info : t_12238
            self.bucket_size = None
        if disable_bucketing:                                                  # trace_info : t_12243
            self.bucket_size = None

        self.module = module                                                   # trace_info : t_12244
        self.param_to_buffer = {}                                              # trace_info : t_12245

        # Group parameters by their gradient type.
        param_to_name = {}                                                     # trace_info : t_12246
        dense_params = []                                                      # trace_info : t_12247
        expert_parallel_params = []                                            # trace_info : t_12248
        for name, param in self.module.named_parameters():                     # trace_info : t_12249, t_12255, t_12261, t_12267, t_12273, ...
            if not param.requires_grad:                                        # trace_info : t_12250, t_12256, t_12262, t_12268, t_12274, ...
                continue

            param.grad_added_to_main_grad = False                              # trace_info : t_12251, t_12257, t_12263, t_12269, t_12275, ...
            param_to_name[param] = name                                        # trace_info : t_12252, t_12258, t_12264, t_12270, t_12276, ...

            if getattr(param, 'allreduce', True):                              # trace_info : t_12253, t_12259, t_12265, t_12271, t_12277, ...
                dense_params.append(param)                                     # trace_info : t_12254, t_12260, t_12266, t_12272, t_12278, ...
            else:
                expert_parallel_params.append(param)

        def allocate_buffers_for_parameters(                                   # trace_info : t_12418
            input_params, data_parallel_group, gradient_scaling_factor,
        ):
            param_and_grad_dtype_to_params = {}                                # trace_info : t_12425, t_14211

            # Group parameters by their gradient type.
            for param in input_params:                                         # trace_info : t_12426, t_12433, t_12440, t_12447, t_12454, ...
                if not param.requires_grad:                                    # trace_info : t_12427, t_12434, t_12441, t_12448, t_12455, ...
                    continue

                param_dtype = param.dtype                                      # trace_info : t_12428, t_12435, t_12442, t_12449, t_12456, ...
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype# trace_info : t_12429, t_12436, t_12443, t_12450, t_12457, ...

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])# trace_info : t_12430, t_12437, t_12444, t_12451, t_12458, ...
                params.append(param)                                           # trace_info : t_12431, t_12438, t_12445, t_12452, t_12459, ...
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params# trace_info : t_12432, t_12439, t_12446, t_12453, t_12460, ...

            # Allocate the grad buffers and map the grads.
            buffers = []                                                       # trace_info : t_12623, t_14213
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():# trace_info : t_12624, t_14204, t_14214
                buffers.append(                                                # trace_info : t_12625, t_14146
                    ParamAndGradBuffer(                                        # trace_info : t_12626, t_12635
                        self.ddp_config,                                       # trace_info : t_12627
                        param_dtype,                                           # trace_info : t_12628
                        grad_dtype,                                            # trace_info : t_12629
                        params,                                                # trace_info : t_12630
                        data_parallel_group,                                   # trace_info : t_12631
                        self.bucket_size,                                      # trace_info : t_12632
                        param_to_name,                                         # trace_info : t_12633
                        gradient_scaling_factor,                               # trace_info : t_12634
                    )
                )
                for param in params:                                           # trace_info : t_14147, t_14149, t_14151, t_14153, t_14155, ...
                    self.param_to_buffer[param] = buffers[-1]                  # trace_info : t_14148, t_14150, t_14152, t_14154, t_14156, ...

            return buffers                                                     # trace_info : t_14205, t_14215

        if config.calculate_per_token_loss:                                    # trace_info : t_12419
            gradient_scaling_factor = 1.0
        else:
            data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)# trace_info : t_12420
            gradient_scaling_factor = 1.0 / data_parallel_world_size           # trace_info : t_12421

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(                        # trace_info : t_12422, t_12424
            dense_params, data_parallel_group, gradient_scaling_factor=gradient_scaling_factor,# trace_info : t_12423
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(        # trace_info : t_14206, t_14210
            expert_parallel_params,                                            # trace_info : t_14207
            expert_data_parallel_group,                                        # trace_info : t_14208
            gradient_scaling_factor=gradient_scaling_factor,                   # trace_info : t_14209
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_14216

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []                                                    # trace_info : t_14217
        for param in self.module.parameters():                                 # trace_info : t_14218, t_14226, t_14234, t_14242, t_14250, ...
            if param.requires_grad:                                            # trace_info : t_14219, t_14227, t_14235, t_14243, t_14251, ...
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)                             # trace_info : t_14220, t_14228, t_14236, t_14244, t_14252, ...
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]              # trace_info : t_14221, t_14229, t_14237, t_14245, t_14253, ...
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))# trace_info : t_14222, t_14230, t_14238, t_14246, t_14254, ...
                self.grad_accs.append(grad_acc)                                # trace_info : t_14225, t_14233, t_14241, t_14249, t_14257, ...

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)                                  # trace_info : t_18312, t_21951, t_89558

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):                                               # trace_info : t_14223, t_14231, t_14239, t_14247, t_14255, ...
            if param.requires_grad:
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    param_to_buffer[param].register_grad_ready(param)

        return param_hook                                                      # trace_info : t_14224, t_14232, t_14240, t_14248, t_14256, ...

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for buffer in self.buffers + self.expert_parallel_buffers:
                buffer.is_last_microbatch = True

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.start_grad_sync()

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_19740, t_19759, t_23377, t_23396, t_90984, ...
            buffer.finish_grad_sync()                                          # trace_info : t_19741, t_23378, t_90985

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():                                 # trace_info : t_17703, t_17706, t_17709, t_17712, t_17715, ...
            if param.requires_grad:                                            # trace_info : t_17704, t_17707, t_17710, t_17713, t_17716, ...
                param.grad_added_to_main_grad = False                          # trace_info : t_17705, t_17708, t_17711, t_17714, t_17717, ...
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_17788, t_17798, t_21371, t_21381, t_88978, ...
            buffer.reset()                                                     # trace_info : t_17789, t_21372, t_88979

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                torch.distributed.broadcast(
                    param.data,
                    src=torch.distributed.get_process_group_ranks(self.expert_data_parallel_group),
                    group=self.expert_data_parallel_group,
                )
            else:
                torch.distributed.broadcast(
                    param.data,
                    src=torch.distributed.get_process_group_ranks(self.data_parallel_group),
                    group=self.data_parallel_group,
                )

    def state_dict(self, prefix='', keep_vars=False):
        """
        Returns a dictionary containing references to the whole state of the
        wrapped module.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names. Parameters and buffers
        set to None are not included.
        """
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """
        Returns wrapped module's state_dict for checkpoint saving.
        """
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        self.module.load_state_dict(state_dict, strict=strict)
