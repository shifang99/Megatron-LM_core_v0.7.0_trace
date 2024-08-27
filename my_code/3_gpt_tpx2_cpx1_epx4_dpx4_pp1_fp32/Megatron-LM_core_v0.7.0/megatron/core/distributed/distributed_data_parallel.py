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
        super().__init__(config=config)                                        # trace_info : t_12433
        self.module = module                                                   # trace_info : t_12436

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:                                     # trace_info : t_12437
            dp_size = parallel_state.get_data_parallel_world_size()            # trace_info : t_12438
            ddp_config.bucket_size = max(40000000, 1000000 * dp_size)          # trace_info : t_12446
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:                                 # trace_info : t_12447
            ddp_config.bucket_size = None                                      # trace_info : t_12448

        self.ddp_config = ddp_config                                           # trace_info : t_12449
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:# trace_info : t_12450
            logger.info(f'Setting up DistributedDataParallel with config {self.ddp_config}')# trace_info : t_12451

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size                         # trace_info : t_12457
        if parallel_state.get_pipeline_model_parallel_rank() > 0:              # trace_info : t_12458
            self.bucket_size = None
        if disable_bucketing:                                                  # trace_info : t_12463
            self.bucket_size = None

        self.module = module                                                   # trace_info : t_12464
        self.param_to_buffer = {}                                              # trace_info : t_12465

        # Group parameters by their gradient type.
        param_to_name = {}                                                     # trace_info : t_12466
        dense_params = []                                                      # trace_info : t_12467
        expert_parallel_params = []                                            # trace_info : t_12468
        for name, param in self.module.named_parameters():                     # trace_info : t_12469, t_12475, t_12481, t_12487, t_12493, ...
            if not param.requires_grad:                                        # trace_info : t_12470, t_12476, t_12482, t_12488, t_12494, ...
                continue

            param.grad_added_to_main_grad = False                              # trace_info : t_12471, t_12477, t_12483, t_12489, t_12495, ...
            param_to_name[param] = name                                        # trace_info : t_12472, t_12478, t_12484, t_12490, t_12496, ...

            if getattr(param, 'allreduce', True):                              # trace_info : t_12473, t_12479, t_12485, t_12491, t_12497, ...
                dense_params.append(param)                                     # trace_info : t_12474, t_12480, t_12486, t_12492, t_12498, ...
            else:
                expert_parallel_params.append(param)                           # trace_info : t_12540, t_12546, t_12552, t_12558, t_12618, ...

        def allocate_buffers_for_parameters(                                   # trace_info : t_12650
            input_params, data_parallel_group, gradient_scaling_factor,
        ):
            param_and_grad_dtype_to_params = {}                                # trace_info : t_12657, t_14095

            # Group parameters by their gradient type.
            for param in input_params:                                         # trace_info : t_12658, t_12665, t_12672, t_12679, t_12686, ...
                if not param.requires_grad:                                    # trace_info : t_12659, t_12666, t_12673, t_12680, t_12687, ...
                    continue

                param_dtype = param.dtype                                      # trace_info : t_12660, t_12667, t_12674, t_12681, t_12688, ...
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype# trace_info : t_12661, t_12668, t_12675, t_12682, t_12689, ...

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])# trace_info : t_12662, t_12669, t_12676, t_12683, t_12690, ...
                params.append(param)                                           # trace_info : t_12663, t_12670, t_12677, t_12684, t_12691, ...
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params# trace_info : t_12664, t_12671, t_12678, t_12685, t_12692, ...

            # Allocate the grad buffers and map the grads.
            buffers = []                                                       # trace_info : t_12813, t_14153
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():# trace_info : t_12814, t_14088, t_14154, t_14714
                buffers.append(                                                # trace_info : t_12815, t_14042, t_14155, t_14696
                    ParamAndGradBuffer(                                        # trace_info : t_12816, t_12825, t_14156, t_14165
                        self.ddp_config,                                       # trace_info : t_12817, t_14157
                        param_dtype,                                           # trace_info : t_12818, t_14158
                        grad_dtype,                                            # trace_info : t_12819, t_14159
                        params,                                                # trace_info : t_12820, t_14160
                        data_parallel_group,                                   # trace_info : t_12821, t_14161
                        self.bucket_size,                                      # trace_info : t_12822, t_14162
                        param_to_name,                                         # trace_info : t_12823, t_14163
                        gradient_scaling_factor,                               # trace_info : t_12824, t_14164
                    )
                )
                for param in params:                                           # trace_info : t_14043, t_14045, t_14047, t_14049, t_14051, ...
                    self.param_to_buffer[param] = buffers[-1]                  # trace_info : t_14044, t_14046, t_14048, t_14050, t_14052, ...

            return buffers                                                     # trace_info : t_14089, t_14715

        if config.calculate_per_token_loss:                                    # trace_info : t_12651
            gradient_scaling_factor = 1.0
        else:
            data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)# trace_info : t_12652
            gradient_scaling_factor = 1.0 / data_parallel_world_size           # trace_info : t_12653

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(                        # trace_info : t_12654, t_12656
            dense_params, data_parallel_group, gradient_scaling_factor=gradient_scaling_factor,# trace_info : t_12655
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(        # trace_info : t_14090, t_14094
            expert_parallel_params,                                            # trace_info : t_14091
            expert_data_parallel_group,                                        # trace_info : t_14092
            gradient_scaling_factor=gradient_scaling_factor,                   # trace_info : t_14093
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_14716

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []                                                    # trace_info : t_14717
        for param in self.module.parameters():                                 # trace_info : t_14718, t_14726, t_14734, t_14742, t_14750, ...
            if param.requires_grad:                                            # trace_info : t_14719, t_14727, t_14735, t_14743, t_14751, ...
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)                             # trace_info : t_14720, t_14728, t_14736, t_14744, t_14752, ...
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]              # trace_info : t_14721, t_14729, t_14737, t_14745, t_14753, ...
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))# trace_info : t_14722, t_14730, t_14738, t_14746, t_14754, ...
                self.grad_accs.append(grad_acc)                                # trace_info : t_14725, t_14733, t_14741, t_14749, t_14757, ...

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)                                  # trace_info : t_18210, t_22563, t_26908

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):                                               # trace_info : t_14723, t_14731, t_14739, t_14747, t_14755, ...
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

        return param_hook                                                      # trace_info : t_14724, t_14732, t_14740, t_14748, t_14756, ...

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
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_20192, t_20214, t_20236, t_24537, t_24559, ...
            buffer.finish_grad_sync()                                          # trace_info : t_20193, t_20215, t_24538, t_24560, t_28883, ...

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():                                 # trace_info : t_17634, t_17637, t_17640, t_17643, t_17646, ...
            if param.requires_grad:                                            # trace_info : t_17635, t_17638, t_17641, t_17644, t_17647, ...
                param.grad_added_to_main_grad = False                          # trace_info : t_17636, t_17639, t_17642, t_17645, t_17648, ...
        for buffer in self.buffers + self.expert_parallel_buffers:             # trace_info : t_17725, t_17735, t_17745, t_22018, t_22028, ...
            buffer.reset()                                                     # trace_info : t_17726, t_17736, t_22019, t_22029, t_26364, ...

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
