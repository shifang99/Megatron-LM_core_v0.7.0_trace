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
        super().__init__(config=config)                                        # trace_info : t_12043
        self.module = module                                                   # trace_info : t_12046

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:                                     # trace_info : t_12047
            dp_size = parallel_state.get_data_parallel_world_size()            # trace_info : t_12048
            ddp_config.bucket_size = max(40000000, 1000000 * dp_size)          # trace_info : t_12056
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:                                 # trace_info : t_12057
            ddp_config.bucket_size = None                                      # trace_info : t_12058

        self.ddp_config = ddp_config                                           # trace_info : t_12059
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_12060
            logger.info(f'Setting up DistributedDataParallel with config {self.ddp_config}')# trace_info : t_12061

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size                         # trace_info : t_12067
        if parallel_state.get_pipeline_model_parallel_rank() > 0:              # trace_info : t_12068
            self.bucket_size = None
        if disable_bucketing:                                                  # trace_info : t_12073
            self.bucket_size = None

        self.module = module                                                   # trace_info : t_12074
        self.param_to_buffer = {}                                              # trace_info : t_12075

        # Group parameters by their gradient type.
        param_to_name = {}                                                     # trace_info : t_12076
        dense_params = []                                                      # trace_info : t_12077
        expert_parallel_params = []                                            # trace_info : t_12078
        for name, param in self.module.named_parameters():                     # trace_info : t_12079, t_12085, t_12091, t_12097, t_12103, ...
            if not param.requires_grad:                                        # trace_info : t_12080, t_12086, t_12092, t_12098, t_12104, ...
                continue

            param.grad_added_to_main_grad = False                              # trace_info : t_12081, t_12087, t_12093, t_12099, t_12105, ...
            param_to_name[param] = name                                        # trace_info : t_12082, t_12088, t_12094, t_12100, t_12106, ...

            if getattr(param, 'allreduce', True):                              # trace_info : t_12083, t_12089, t_12095, t_12101, t_12107, ...
                dense_params.append(param)                                     # trace_info : t_12084, t_12090, t_12096, t_12102, t_12108, ...
            else:
                expert_parallel_params.append(param)

        def allocate_buffers_for_parameters(                                   # trace_info : t_12248
            input_params, data_parallel_group, gradient_scaling_factor,
        ):
            param_and_grad_dtype_to_params = {}                                # trace_info : t_12255, t_14041

            # Group parameters by their gradient type.
            for param in input_params:                                         # trace_info : t_12256, t_12263, t_12270, t_12277, t_12284, ...
                if not param.requires_grad:                                    # trace_info : t_12257, t_12264, t_12271, t_12278, t_12285, ...
                    continue

                param_dtype = param.dtype                                      # trace_info : t_12258, t_12265, t_12272, t_12279, t_12286, ...
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype# trace_info : t_12259, t_12266, t_12273, t_12280, t_12287, ...

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])# trace_info : t_12260, t_12267, t_12274, t_12281, t_12288, ...
                params.append(param)                                           # trace_info : t_12261, t_12268, t_12275, t_12282, t_12289, ...
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params# trace_info : t_12262, t_12269, t_12276, t_12283, t_12290, ...

            # Allocate the grad buffers and map the grads.
            buffers = []                                                       # trace_info : t_12453, t_14043
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():# trace_info : t_12454, t_14034, t_14044
                buffers.append(                                                # trace_info : t_12455, t_13976
                    ParamAndGradBuffer(                                        # trace_info : t_12456, t_12465
                        self.ddp_config,                                       # trace_info : t_12457
                        param_dtype,                                           # trace_info : t_12458
                        grad_dtype,                                            # trace_info : t_12459
                        params,                                                # trace_info : t_12460
                        data_parallel_group,                                   # trace_info : t_12461
                        self.bucket_size,                                      # trace_info : t_12462
                        param_to_name,                                         # trace_info : t_12463
                        gradient_scaling_factor,                               # trace_info : t_12464
                    )
                )
                for param in params:                                           # trace_info : t_13977, t_13979, t_13981, t_13983, t_13985, ...
                    self.param_to_buffer[param] = buffers[-1]                  # trace_info : t_13978, t_13980, t_13982, t_13984, t_13986, ...

            return buffers                                                     # trace_info : t_14035, t_14045

        if config.calculate_per_token_loss:                                    # trace_info : t_12249
            gradient_scaling_factor = 1.0
        else:
            data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)# trace_info : t_12250
            gradient_scaling_factor = 1.0 / data_parallel_world_size           # trace_info : t_12251

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(                        # trace_info : t_12252, t_12254
            dense_params, data_parallel_group, gradient_scaling_factor=gradient_scaling_factor,# trace_info : t_12253
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(        # trace_info : t_14036, t_14040
            expert_parallel_params,                                            # trace_info : t_14037
            expert_data_parallel_group,                                        # trace_info : t_14038
            gradient_scaling_factor=gradient_scaling_factor,                   # trace_info : t_14039
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_14046

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []                                                    # trace_info : t_14047
        for param in self.module.parameters():                                 # trace_info : t_14048, t_14056, t_14064, t_14072, t_14080, ...
            if param.requires_grad:                                            # trace_info : t_14049, t_14057, t_14065, t_14073, t_14081, ...
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)                             # trace_info : t_14050, t_14058, t_14066, t_14074, t_14082, ...
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]              # trace_info : t_14051, t_14059, t_14067, t_14075, t_14083, ...
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))# trace_info : t_14052, t_14060, t_14068, t_14076, t_14084, ...
                self.grad_accs.append(grad_acc)                                # trace_info : t_14055, t_14063, t_14071, t_14079, t_14087, ...

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)                                  # trace_info : t_18152, t_21338, t_24524

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):                                               # trace_info : t_14053, t_14061, t_14069, t_14077, t_14085, ...
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

        return param_hook                                                      # trace_info : t_14054, t_14062, t_14070, t_14078, t_14086, ...

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
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_19010, t_19029, t_22196, t_22215, t_25382, ...
            buffer.finish_grad_sync()                                          # trace_info : t_19011, t_22197, t_25383

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():                                 # trace_info : t_17425, t_17428, t_17431, t_17434, t_17437, ...
            if param.requires_grad:                                            # trace_info : t_17426, t_17429, t_17432, t_17435, t_17438, ...
                param.grad_added_to_main_grad = False                          # trace_info : t_17427, t_17430, t_17433, t_17436, t_17439, ...
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_17510, t_17520, t_20640, t_20650, t_23826, ...
            buffer.reset()                                                     # trace_info : t_17511, t_20641, t_23827

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
