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
        super().__init__(config=config)                                        # trace_info : t_9184
        self.module = module                                                   # trace_info : t_9187

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:                                     # trace_info : t_9188
            dp_size = parallel_state.get_data_parallel_world_size()            # trace_info : t_9189
            ddp_config.bucket_size = max(40000000, 1000000 * dp_size)          # trace_info : t_9197
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:                                 # trace_info : t_9198
            ddp_config.bucket_size = None                                      # trace_info : t_9199

        self.ddp_config = ddp_config                                           # trace_info : t_9200
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_9201
            logger.info(f'Setting up DistributedDataParallel with config {self.ddp_config}')# trace_info : t_9202

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size                         # trace_info : t_9208
        if parallel_state.get_pipeline_model_parallel_rank() > 0:              # trace_info : t_9209
            self.bucket_size = None
        if disable_bucketing:                                                  # trace_info : t_9214
            self.bucket_size = None

        self.module = module                                                   # trace_info : t_9215
        self.param_to_buffer = {}                                              # trace_info : t_9216

        # Group parameters by their gradient type.
        param_to_name = {}                                                     # trace_info : t_9217
        dense_params = []                                                      # trace_info : t_9218
        expert_parallel_params = []                                            # trace_info : t_9219
        for name, param in self.module.named_parameters():                     # trace_info : t_9220, t_9226, t_9232, t_9238, t_9244, ...
            if not param.requires_grad:                                        # trace_info : t_9221, t_9227, t_9233, t_9239, t_9245, ...
                continue

            param.grad_added_to_main_grad = False                              # trace_info : t_9222, t_9228, t_9234, t_9240, t_9246, ...
            param_to_name[param] = name                                        # trace_info : t_9223, t_9229, t_9235, t_9241, t_9247, ...

            if getattr(param, 'allreduce', True):                              # trace_info : t_9224, t_9230, t_9236, t_9242, t_9248, ...
                dense_params.append(param)                                     # trace_info : t_9225, t_9231, t_9237, t_9243, t_9249, ...
            else:
                expert_parallel_params.append(param)

        def allocate_buffers_for_parameters(                                   # trace_info : t_9389
            input_params, data_parallel_group, gradient_scaling_factor,
        ):
            param_and_grad_dtype_to_params = {}                                # trace_info : t_9396, t_11182

            # Group parameters by their gradient type.
            for param in input_params:                                         # trace_info : t_9397, t_9404, t_9411, t_9418, t_9425, ...
                if not param.requires_grad:                                    # trace_info : t_9398, t_9405, t_9412, t_9419, t_9426, ...
                    continue

                param_dtype = param.dtype                                      # trace_info : t_9399, t_9406, t_9413, t_9420, t_9427, ...
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype# trace_info : t_9400, t_9407, t_9414, t_9421, t_9428, ...

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])# trace_info : t_9401, t_9408, t_9415, t_9422, t_9429, ...
                params.append(param)                                           # trace_info : t_9402, t_9409, t_9416, t_9423, t_9430, ...
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params# trace_info : t_9403, t_9410, t_9417, t_9424, t_9431, ...

            # Allocate the grad buffers and map the grads.
            buffers = []                                                       # trace_info : t_9594, t_11184
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():# trace_info : t_9595, t_11175, t_11185
                buffers.append(                                                # trace_info : t_9596, t_11117
                    ParamAndGradBuffer(                                        # trace_info : t_9597, t_9606
                        self.ddp_config,                                       # trace_info : t_9598
                        param_dtype,                                           # trace_info : t_9599
                        grad_dtype,                                            # trace_info : t_9600
                        params,                                                # trace_info : t_9601
                        data_parallel_group,                                   # trace_info : t_9602
                        self.bucket_size,                                      # trace_info : t_9603
                        param_to_name,                                         # trace_info : t_9604
                        gradient_scaling_factor,                               # trace_info : t_9605
                    )
                )
                for param in params:                                           # trace_info : t_11118, t_11120, t_11122, t_11124, t_11126, ...
                    self.param_to_buffer[param] = buffers[-1]                  # trace_info : t_11119, t_11121, t_11123, t_11125, t_11127, ...

            return buffers                                                     # trace_info : t_11176, t_11186

        if config.calculate_per_token_loss:                                    # trace_info : t_9390
            gradient_scaling_factor = 1.0
        else:
            data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)# trace_info : t_9391
            gradient_scaling_factor = 1.0 / data_parallel_world_size           # trace_info : t_9392

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(                        # trace_info : t_9393, t_9395
            dense_params, data_parallel_group, gradient_scaling_factor=gradient_scaling_factor,# trace_info : t_9394
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(        # trace_info : t_11177, t_11181
            expert_parallel_params,                                            # trace_info : t_11178
            expert_data_parallel_group,                                        # trace_info : t_11179
            gradient_scaling_factor=gradient_scaling_factor,                   # trace_info : t_11180
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_11187

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []                                                    # trace_info : t_11188
        for param in self.module.parameters():                                 # trace_info : t_11189, t_11197, t_11205, t_11213, t_11221, ...
            if param.requires_grad:                                            # trace_info : t_11190, t_11198, t_11206, t_11214, t_11222, ...
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)                             # trace_info : t_11191, t_11199, t_11207, t_11215, t_11223, ...
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]              # trace_info : t_11192, t_11200, t_11208, t_11216, t_11224, ...
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))# trace_info : t_11193, t_11201, t_11209, t_11217, t_11225, ...
                self.grad_accs.append(grad_acc)                                # trace_info : t_11196, t_11204, t_11212, t_11220, t_11228, ...

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)                                  # trace_info : t_15167, t_18808, t_22447

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):                                               # trace_info : t_11194, t_11202, t_11210, t_11218, t_11226, ...
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

        return param_hook                                                      # trace_info : t_11195, t_11203, t_11211, t_11219, t_11227, ...

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
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_16612, t_16630, t_20251, t_20269, t_23890, ...
            buffer.finish_grad_sync()                                          # trace_info : t_16613, t_20252, t_23891

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():                                 # trace_info : t_14567, t_14570, t_14573, t_14576, t_14579, ...
            if param.requires_grad:                                            # trace_info : t_14568, t_14571, t_14574, t_14577, t_14580, ...
                param.grad_added_to_main_grad = False                          # trace_info : t_14569, t_14572, t_14575, t_14578, t_14581, ...
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_14652, t_14662, t_18237, t_18247, t_21876, ...
            buffer.reset()                                                     # trace_info : t_14653, t_18238, t_21877

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
