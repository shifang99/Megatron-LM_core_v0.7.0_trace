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
        super().__init__(config=config)                                        # trace_info : t_12472
        self.module = module                                                   # trace_info : t_12475

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:                                     # trace_info : t_12476
            dp_size = parallel_state.get_data_parallel_world_size()            # trace_info : t_12477
            ddp_config.bucket_size = max(40000000, 1000000 * dp_size)          # trace_info : t_12485
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:                                 # trace_info : t_12486
            ddp_config.bucket_size = None                                      # trace_info : t_12487

        self.ddp_config = ddp_config                                           # trace_info : t_12488
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_12489
            logger.info(f'Setting up DistributedDataParallel with config {self.ddp_config}')# trace_info : t_12490

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size                         # trace_info : t_12496
        if parallel_state.get_pipeline_model_parallel_rank() > 0:              # trace_info : t_12497
            self.bucket_size = None
        if disable_bucketing:                                                  # trace_info : t_12502
            self.bucket_size = None

        self.module = module                                                   # trace_info : t_12503
        self.param_to_buffer = {}                                              # trace_info : t_12504

        # Group parameters by their gradient type.
        param_to_name = {}                                                     # trace_info : t_12505
        dense_params = []                                                      # trace_info : t_12506
        expert_parallel_params = []                                            # trace_info : t_12507
        for name, param in self.module.named_parameters():                     # trace_info : t_12508, t_12514, t_12520, t_12526, t_12532, ...
            if not param.requires_grad:                                        # trace_info : t_12509, t_12515, t_12521, t_12527, t_12533, ...
                continue

            param.grad_added_to_main_grad = False                              # trace_info : t_12510, t_12516, t_12522, t_12528, t_12534, ...
            param_to_name[param] = name                                        # trace_info : t_12511, t_12517, t_12523, t_12529, t_12535, ...

            if getattr(param, 'allreduce', True):                              # trace_info : t_12512, t_12518, t_12524, t_12530, t_12536, ...
                dense_params.append(param)                                     # trace_info : t_12513, t_12519, t_12525, t_12531, t_12537, ...
            else:
                expert_parallel_params.append(param)

        def allocate_buffers_for_parameters(                                   # trace_info : t_12665
            input_params, data_parallel_group, gradient_scaling_factor,
        ):
            param_and_grad_dtype_to_params = {}                                # trace_info : t_12672, t_14344

            # Group parameters by their gradient type.
            for param in input_params:                                         # trace_info : t_12673, t_12680, t_12687, t_12694, t_12701, ...
                if not param.requires_grad:                                    # trace_info : t_12674, t_12681, t_12688, t_12695, t_12702, ...
                    continue

                param_dtype = param.dtype                                      # trace_info : t_12675, t_12682, t_12689, t_12696, t_12703, ...
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype# trace_info : t_12676, t_12683, t_12690, t_12697, t_12704, ...

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])# trace_info : t_12677, t_12684, t_12691, t_12698, t_12705, ...
                params.append(param)                                           # trace_info : t_12678, t_12685, t_12692, t_12699, t_12706, ...
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params# trace_info : t_12679, t_12686, t_12693, t_12700, t_12707, ...

            # Allocate the grad buffers and map the grads.
            buffers = []                                                       # trace_info : t_12856, t_14346
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():# trace_info : t_12857, t_14337, t_14347
                buffers.append(                                                # trace_info : t_12858, t_14283
                    ParamAndGradBuffer(                                        # trace_info : t_12859, t_12868
                        self.ddp_config,                                       # trace_info : t_12860
                        param_dtype,                                           # trace_info : t_12861
                        grad_dtype,                                            # trace_info : t_12862
                        params,                                                # trace_info : t_12863
                        data_parallel_group,                                   # trace_info : t_12864
                        self.bucket_size,                                      # trace_info : t_12865
                        param_to_name,                                         # trace_info : t_12866
                        gradient_scaling_factor,                               # trace_info : t_12867
                    )
                )
                for param in params:                                           # trace_info : t_14284, t_14286, t_14288, t_14290, t_14292, ...
                    self.param_to_buffer[param] = buffers[-1]                  # trace_info : t_14285, t_14287, t_14289, t_14291, t_14293, ...

            return buffers                                                     # trace_info : t_14338, t_14348

        if config.calculate_per_token_loss:                                    # trace_info : t_12666
            gradient_scaling_factor = 1.0
        else:
            data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)# trace_info : t_12667
            gradient_scaling_factor = 1.0 / data_parallel_world_size           # trace_info : t_12668

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(                        # trace_info : t_12669, t_12671
            dense_params, data_parallel_group, gradient_scaling_factor=gradient_scaling_factor,# trace_info : t_12670
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(        # trace_info : t_14339, t_14343
            expert_parallel_params,                                            # trace_info : t_14340
            expert_data_parallel_group,                                        # trace_info : t_14341
            gradient_scaling_factor=gradient_scaling_factor,                   # trace_info : t_14342
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_14349

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []                                                    # trace_info : t_14350
        for param in self.module.parameters():                                 # trace_info : t_14351, t_14359, t_14367, t_14375, t_14383, ...
            if param.requires_grad:                                            # trace_info : t_14352, t_14360, t_14368, t_14376, t_14384, ...
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)                             # trace_info : t_14353, t_14361, t_14369, t_14377, t_14385, ...
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]              # trace_info : t_14354, t_14362, t_14370, t_14378, t_14386, ...
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))# trace_info : t_14355, t_14363, t_14371, t_14379, t_14387, ...
                self.grad_accs.append(grad_acc)                                # trace_info : t_14358, t_14366, t_14374, t_14382, t_14390, ...

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)                                  # trace_info : t_18299, t_22029, t_25757

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):                                               # trace_info : t_14356, t_14364, t_14372, t_14380, t_14388, ...
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

        return param_hook                                                      # trace_info : t_14357, t_14365, t_14373, t_14381, t_14389, ...

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
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_19824, t_19842, t_23552, t_23570, t_27280, ...
            buffer.finish_grad_sync()                                          # trace_info : t_19825, t_23553, t_27281

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():                                 # trace_info : t_17632, t_17635, t_17638, t_17641, t_17644, ...
            if param.requires_grad:                                            # trace_info : t_17633, t_17636, t_17639, t_17642, t_17645, ...
                param.grad_added_to_main_grad = False                          # trace_info : t_17634, t_17637, t_17640, t_17643, t_17646, ...
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_17711, t_17721, t_21389, t_21399, t_25117, ...
            buffer.reset()                                                     # trace_info : t_17712, t_21390, t_25118

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
