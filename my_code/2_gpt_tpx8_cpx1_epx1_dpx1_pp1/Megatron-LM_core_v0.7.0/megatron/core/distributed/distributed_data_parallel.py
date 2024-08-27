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
        super().__init__(config=config)                                        # trace_info : t_14051
        self.module = module                                                   # trace_info : t_14054

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:                                     # trace_info : t_14055
            dp_size = parallel_state.get_data_parallel_world_size()            # trace_info : t_14056
            ddp_config.bucket_size = max(40000000, 1000000 * dp_size)          # trace_info : t_14064
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:                                 # trace_info : t_14065
            ddp_config.bucket_size = None                                      # trace_info : t_14066

        self.ddp_config = ddp_config                                           # trace_info : t_14067
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_14068
            logger.info(f'Setting up DistributedDataParallel with config {self.ddp_config}')# trace_info : t_14069

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size                         # trace_info : t_14075
        if parallel_state.get_pipeline_model_parallel_rank() > 0:              # trace_info : t_14076
            self.bucket_size = None
        if disable_bucketing:                                                  # trace_info : t_14081
            self.bucket_size = None

        self.module = module                                                   # trace_info : t_14082
        self.param_to_buffer = {}                                              # trace_info : t_14083

        # Group parameters by their gradient type.
        param_to_name = {}                                                     # trace_info : t_14084
        dense_params = []                                                      # trace_info : t_14085
        expert_parallel_params = []                                            # trace_info : t_14086
        for name, param in self.module.named_parameters():                     # trace_info : t_14087, t_14093, t_14099, t_14105, t_14111, ...
            if not param.requires_grad:                                        # trace_info : t_14088, t_14094, t_14100, t_14106, t_14112, ...
                continue

            param.grad_added_to_main_grad = False                              # trace_info : t_14089, t_14095, t_14101, t_14107, t_14113, ...
            param_to_name[param] = name                                        # trace_info : t_14090, t_14096, t_14102, t_14108, t_14114, ...

            if getattr(param, 'allreduce', True):                              # trace_info : t_14091, t_14097, t_14103, t_14109, t_14115, ...
                dense_params.append(param)                                     # trace_info : t_14092, t_14098, t_14104, t_14110, t_14116, ...
            else:
                expert_parallel_params.append(param)

        def allocate_buffers_for_parameters(                                   # trace_info : t_14256
            input_params, data_parallel_group, gradient_scaling_factor,
        ):
            param_and_grad_dtype_to_params = {}                                # trace_info : t_14263, t_16049

            # Group parameters by their gradient type.
            for param in input_params:                                         # trace_info : t_14264, t_14271, t_14278, t_14285, t_14292, ...
                if not param.requires_grad:                                    # trace_info : t_14265, t_14272, t_14279, t_14286, t_14293, ...
                    continue

                param_dtype = param.dtype                                      # trace_info : t_14266, t_14273, t_14280, t_14287, t_14294, ...
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype# trace_info : t_14267, t_14274, t_14281, t_14288, t_14295, ...

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])# trace_info : t_14268, t_14275, t_14282, t_14289, t_14296, ...
                params.append(param)                                           # trace_info : t_14269, t_14276, t_14283, t_14290, t_14297, ...
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params# trace_info : t_14270, t_14277, t_14284, t_14291, t_14298, ...

            # Allocate the grad buffers and map the grads.
            buffers = []                                                       # trace_info : t_14461, t_16051
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():# trace_info : t_14462, t_16042, t_16052
                buffers.append(                                                # trace_info : t_14463, t_15984
                    ParamAndGradBuffer(                                        # trace_info : t_14464, t_14473
                        self.ddp_config,                                       # trace_info : t_14465
                        param_dtype,                                           # trace_info : t_14466
                        grad_dtype,                                            # trace_info : t_14467
                        params,                                                # trace_info : t_14468
                        data_parallel_group,                                   # trace_info : t_14469
                        self.bucket_size,                                      # trace_info : t_14470
                        param_to_name,                                         # trace_info : t_14471
                        gradient_scaling_factor,                               # trace_info : t_14472
                    )
                )
                for param in params:                                           # trace_info : t_15985, t_15987, t_15989, t_15991, t_15993, ...
                    self.param_to_buffer[param] = buffers[-1]                  # trace_info : t_15986, t_15988, t_15990, t_15992, t_15994, ...

            return buffers                                                     # trace_info : t_16043, t_16053

        if config.calculate_per_token_loss:                                    # trace_info : t_14257
            gradient_scaling_factor = 1.0
        else:
            data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)# trace_info : t_14258
            gradient_scaling_factor = 1.0 / data_parallel_world_size           # trace_info : t_14259

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(                        # trace_info : t_14260, t_14262
            dense_params, data_parallel_group, gradient_scaling_factor=gradient_scaling_factor,# trace_info : t_14261
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(        # trace_info : t_16044, t_16048
            expert_parallel_params,                                            # trace_info : t_16045
            expert_data_parallel_group,                                        # trace_info : t_16046
            gradient_scaling_factor=gradient_scaling_factor,                   # trace_info : t_16047
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_16054

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []                                                    # trace_info : t_16055
        for param in self.module.parameters():                                 # trace_info : t_16056, t_16064, t_16072, t_16080, t_16088, ...
            if param.requires_grad:                                            # trace_info : t_16057, t_16065, t_16073, t_16081, t_16089, ...
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)                             # trace_info : t_16058, t_16066, t_16074, t_16082, t_16090, ...
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]              # trace_info : t_16059, t_16067, t_16075, t_16083, t_16091, ...
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))# trace_info : t_16060, t_16068, t_16076, t_16084, t_16092, ...
                self.grad_accs.append(grad_acc)                                # trace_info : t_16063, t_16071, t_16079, t_16087, t_16095, ...

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)                                  # trace_info : t_20033, t_23645, t_27255

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):                                               # trace_info : t_16061, t_16069, t_16077, t_16085, t_16093, ...
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

        return param_hook                                                      # trace_info : t_16062, t_16070, t_16078, t_16086, t_16094, ...

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
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_21445, t_21463, t_25055, t_25073, t_28665, ...
            buffer.finish_grad_sync()                                          # trace_info : t_21446, t_25056, t_28666

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():                                 # trace_info : t_19433, t_19436, t_19439, t_19442, t_19445, ...
            if param.requires_grad:                                            # trace_info : t_19434, t_19437, t_19440, t_19443, t_19446, ...
                param.grad_added_to_main_grad = False                          # trace_info : t_19435, t_19438, t_19441, t_19444, t_19447, ...
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_19518, t_19528, t_23074, t_23084, t_26684, ...
            buffer.reset()                                                     # trace_info : t_19519, t_23075, t_26685

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
