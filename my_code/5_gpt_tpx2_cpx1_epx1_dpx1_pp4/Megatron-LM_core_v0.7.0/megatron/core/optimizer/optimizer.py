# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron optimizer."""

import math
from abc import ABC, abstractmethod
from itertools import chain
from logging import getLogger
from typing import Any, Callable, List, Optional, Tuple

import amp_C
import torch
from apex.multi_tensor_apply import multi_tensor_applier

from .. import parallel_state, tensor_parallel
from ..dist_checkpointing.mapping import ShardedStateDict
from ..dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from ..dist_checkpointing.utils import add_prefix_for_sharding
from ..transformer.module import param_is_not_shared
from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32
from .grad_scaler import MegatronGradScaler
from .optimizer_config import OptimizerConfig

logger = getLogger(__name__)


def _zero_grad_group_helper(group: List[torch.nn.Parameter], set_to_none: bool):
    """
    Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer.
    """
    for param in group:                                                        # trace_info : t_17726, t_17728, t_17730, t_17732, t_17734, ...
        if param.grad is not None:                                             # trace_info : t_17727, t_17729, t_17731, t_17733, t_17735, ...
            if set_to_none:                                                    # trace_info : t_21465, t_21469, t_21473, t_21477, t_21481, ...
                param.grad = None                                              # trace_info : t_21466, t_21470, t_21474, t_21478, t_21482, ...
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(
    this: List[torch.Tensor], that: List[torch.Tensor], overflow_buf: Optional[torch.Tensor] = None
):
    """
    Use multi-tensor-applier to copy values from one list to another.
    We don't have a bfloat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16.
    """
    if overflow_buf:                                                           # trace_info : t_20994, t_24722, t_28450
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):                                   # trace_info : t_20995, t_20997, t_20999, t_21001, t_21003, ...
            that_.copy_(this_)                                                 # trace_info : t_20996, t_20998, t_21000, t_21002, t_21004, ...


class MegatronOptimizer(ABC):
    """
    Base class for all Megatron optimizers.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable = lambda x: None,
    ):

        """Input optimizer is the base optimizer (e.g., Adam)."""
        self.optimizer = optimizer                                             # trace_info : t_15221
        assert self.optimizer, 'no optimizer is provided.'                     # trace_info : t_15222
        self.config = config                                                   # trace_info : t_15223
        self.init_state_fn = init_state_fn                                     # trace_info : t_15224

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []                                                            # trace_info : t_20257, t_20318, t_23985, t_24046, t_27713, ...
        for param_group in self.optimizer.param_groups:                        # trace_info : t_20258, t_20280, t_20314, t_20319, t_20341, ...
            for param in param_group['params']:                                # trace_info : t_20259, t_20261, t_20263, t_20265, t_20267, ...
                params.append(param)                                           # trace_info : t_20260, t_20262, t_20264, t_20266, t_20268, ...
        return params                                                          # trace_info : t_20315, t_20376, t_24043, t_24104, t_27771, ...

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()                                         # trace_info : t_20317, t_24045, t_27773
        grads_for_norm = []                                                    # trace_info : t_20377, t_24105, t_27833
        for param in params:                                                   # trace_info : t_20378, t_20387, t_20403, t_20412, t_20421, ...
            grad = param.grad                                                  # trace_info : t_20379, t_20388, t_20404, t_20413, t_20422, ...
            grad_not_none = grad is not None                                   # trace_info : t_20380, t_20389, t_20405, t_20414, t_20423, ...
            is_not_shared = param_is_not_shared(param)                         # trace_info : t_20381, t_20390, t_20406, t_20415, t_20424, ...
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)# trace_info : t_20383, t_20392, t_20408, t_20417, t_20426, ...
            if grad_not_none and is_not_shared and is_not_tp_duplicate:        # trace_info : t_20385, t_20401, t_20410, t_20419, t_20428, ...
                grads_for_norm.append(grad)                                    # trace_info : t_20386, t_20402, t_20411, t_20420, t_20429, ...

        return grads_for_norm                                                  # trace_info : t_20704, t_24432, t_28160

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Default returned here, but the distributed optimizer overrides this."""
        return parallel_state.get_model_parallel_group()                       # trace_info : t_20223, t_20707, t_23951, t_24435, t_27679, ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm."""
        params = self.get_parameters()                                         # trace_info : t_20256, t_23984, t_27712
        grads_for_norm = self.get_main_grads_for_grad_norm()                   # trace_info : t_20316, t_24044, t_27772
        return clip_grad_norm_fp32(                                            # trace_info : t_20705, t_20710, t_24433, t_24438, t_28161, ...
            params, grads_for_norm, clip_grad, model_parallel_group=self.get_model_parallel_group(),# trace_info : t_20706, t_24434, t_28162
        )

    def count_zeros(self) -> float:
        """Count number of zeros in model's gradients."""
        params = self.get_parameters()
        return count_zeros_fp32(params, model_parallel_group=self.get_model_parallel_group())

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True):
        pass

    @abstractmethod
    def get_loss_scale(self) -> torch.Tensor:
        """
        Get current loss scale factor.
        NOTE: The output should be a CUDA tensor of size 1.
        """
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Simple scaling."""
        return self.get_loss_scale() * loss

    def finish_param_sync(self, model_index: int):
        """
        Finish parameter synchronization for all optimizers.
        This is a no-op for all non-distributed optimizers.
        """
        pass

    @abstractmethod
    def reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups                                     # trace_info : t_15937, t_21083, t_21192, t_24811, t_24920, ...

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    @abstractmethod
    def step(self):
        """Step the optimizer."""
        pass

    @abstractmethod
    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ) -> ShardedStateDict:
        """ Builds sharded state dict for the optimizer, based on model's sharded state dict.

        Args:
            model_sharded_state_dict (ShardedStateDict): sharded state dict of the model
            is_loading (bool, optional): flag indicating whether the state dict will be used to save or load the optimizer state.
                Defaults to False.

        Returns: optimizer sharded state dict
        """


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Base class for both the float-16 and the distributed optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: Optional[MegatronGradScaler],
        init_state_fn: Callable,
    ):

        super().__init__(                                                      # trace_info : t_15218, t_15220
            optimizer, config, init_state_fn,                                  # trace_info : t_15219
        )
        self.grad_scaler = grad_scaler                                         # trace_info : t_15225

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:                                           # trace_info : t_15226
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:                                                   # trace_info : t_15227
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')# trace_info : t_15228

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:                                                   # trace_info : t_15229
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_15230

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:                                           # trace_info : t_15231
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:                                           # trace_info : t_21183, t_24911, t_28639
            return self._scale_one
        return self.grad_scaler.scale                                          # trace_info : t_21184, t_24912, t_28640

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()              # trace_info : t_20125, t_23853, t_27581

        # Reset found inf.
        self.found_inf.fill_(0.0)                                              # trace_info : t_20216, t_23944, t_27672

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(                      # trace_info : t_20217, t_20220, t_23945, t_23948, t_27673, ...
            main_grads, self.found_inf, self.grad_scaler.inv_scale             # trace_info : t_20218, t_23946, t_27674
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(                                          # trace_info : t_20221, t_20226, t_23949, t_23954, t_27677, ...
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()# trace_info : t_20222, t_23950, t_27678
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0                             # trace_info : t_20227, t_23955, t_27683

        return found_inf_flag                                                  # trace_info : t_20228, t_23956, t_27684

    @torch.no_grad()
    def step(self):

        timers = self.config.timers                                            # trace_info : t_19978, t_23706, t_27434

        # Copy gradients from model params to main params.
        if timers is not None:                                                 # trace_info : t_19979, t_23707, t_27435
            timers('optimizer-copy-to-main-grad', log_level=1).start(          # trace_info : t_19980, t_19987, t_23708, t_23715, t_27436, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_19986, t_23714, t_27442
            )
        self._copy_model_grads_to_main_grads()                                 # trace_info : t_19989, t_23717, t_27445
        if timers is not None:                                                 # trace_info : t_20104, t_23832, t_27560
            timers('optimizer-copy-to-main-grad').stop()                       # trace_info : t_20105, t_23833, t_27561

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:                                                   # trace_info : t_20113, t_23841, t_27569

            # Unscale and check for inf/nan.
            if timers is not None:                                             # trace_info : t_20114, t_23842, t_27570
                timers('optimizer-unscale-and-check-inf', log_level=1).start(  # trace_info : t_20115, t_20122, t_23843, t_23850, t_27571, ...
                    barrier=self.config.barrier_with_L1_time                   # trace_info : t_20121, t_23849, t_27577
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()      # trace_info : t_20124, t_23852, t_27580
            if timers is not None:                                             # trace_info : t_20229, t_23957, t_27685
                timers('optimizer-unscale-and-check-inf').stop()               # trace_info : t_20230, t_23958, t_27686

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)                            # trace_info : t_20238, t_23966, t_27694

            # If we found inf/nan, skip the update.
            if found_inf_flag:                                                 # trace_info : t_20242, t_23970, t_27698
                return False, None, None

        # Clip the main gradients.
        if timers is not None:                                                 # trace_info : t_20243, t_23971, t_27699
            timers('optimizer-clip-main-grad', log_level=1).start(             # trace_info : t_20244, t_20251, t_23972, t_23979, t_27700, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20250, t_23978, t_27706
            )
        grad_norm = None                                                       # trace_info : t_20253, t_23981, t_27709
        if self.config.clip_grad > 0.0:                                        # trace_info : t_20254, t_23982, t_27710
            grad_norm = self.clip_grad_norm(self.config.clip_grad)             # trace_info : t_20255, t_23983, t_27711
        if timers is not None:                                                 # trace_info : t_20844, t_24572, t_28300
            timers('optimizer-clip-main-grad').stop()                          # trace_info : t_20845, t_24573, t_28301

        # Count the zeros in the grads.
        if timers is not None:                                                 # trace_info : t_20853, t_24581, t_28309
            timers('optimizer-count-zeros', log_level=1).start(                # trace_info : t_20854, t_20861, t_24582, t_24589, t_28310, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20860, t_24588, t_28316
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None# trace_info : t_20863, t_24591, t_28319
        if timers is not None:                                                 # trace_info : t_20864, t_24592, t_28320
            timers('optimizer-count-zeros').stop()                             # trace_info : t_20865, t_24593, t_28321

        # Step the optimizer.
        if timers is not None:                                                 # trace_info : t_20873, t_24601, t_28329
            timers('optimizer-inner-step', log_level=1).start(                 # trace_info : t_20874, t_20881, t_24602, t_24609, t_28330, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20880, t_24608, t_28336
            )
        self.optimizer.step()                                                  # trace_info : t_20883, t_24611, t_28339
        if timers is not None:                                                 # trace_info : t_20884, t_24612, t_28340
            timers('optimizer-inner-step').stop()                              # trace_info : t_20885, t_24613, t_28341

        # Update params from main params.
        if timers is not None:                                                 # trace_info : t_20893, t_24621, t_28349
            timers('optimizer-copy-main-to-model-params', log_level=1).start(  # trace_info : t_20894, t_20901, t_24622, t_24629, t_28350, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20900, t_24628, t_28356
            )
        self._copy_main_params_to_model_params()                               # trace_info : t_20903, t_24631, t_28359
        if timers is not None:                                                 # trace_info : t_21048, t_24776, t_28504
            timers('optimizer-copy-main-to-model-params').stop()               # trace_info : t_21049, t_24777, t_28505

        # Successful update.
        return True, grad_norm, num_zeros_in_grad                              # trace_info : t_21057, t_24785, t_28513


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Callable,
    ):

        super().__init__(                                                      # trace_info : t_15215, t_15217
            optimizer, config, grad_scaler, init_state_fn,                     # trace_info : t_15216
        )

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []                                               # trace_info : t_15232
        self.fp32_from_float16_groups = []                                     # trace_info : t_15233
        self.fp32_from_fp32_groups = []                                        # trace_info : t_15234

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:                        # trace_info : t_15235, t_15483, t_15875
            float16_params_this_group = []                                     # trace_info : t_15236, t_15484
            fp32_params_this_group = []                                        # trace_info : t_15237, t_15485
            fp32_from_float16_params_this_group = []                           # trace_info : t_15238, t_15486
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):                  # trace_info : t_15239, t_15263, t_15287, t_15311, t_15335, ...
                if param.requires_grad:                                        # trace_info : t_15240, t_15264, t_15288, t_15312, t_15336, ...

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:# trace_info : t_15241, t_15265, t_15289, t_15313, t_15337, ...
                        float16_params_this_group.append(param)                # trace_info : t_15242, t_15266, t_15290, t_15314, t_15338, ...
                        # Create a copy
                        main_param = param.detach().clone().float()            # trace_info : t_15243, t_15267, t_15291, t_15315, t_15339, ...
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)# trace_info : t_15244, t_15268, t_15292, t_15316, t_15340, ...
                        if hasattr(param, 'shared'):                           # trace_info : t_15259, t_15283, t_15307, t_15331, t_15355, ...
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param                  # trace_info : t_15260, t_15284, t_15308, t_15332, t_15356, ...

                        fp32_from_float16_params_this_group.append(main_param) # trace_info : t_15261, t_15285, t_15309, t_15333, t_15357, ...
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:                      # trace_info : t_15262, t_15286, t_15310, t_15334, t_15358, ...
                            self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        raise TypeError(
                            'Wrapped parameters must be one of '
                            'torch.cuda.FloatTensor,  '
                            'torch.cuda.HalfTensor, or '
                            'torch.cuda.BFloat16Tensor. '
                            'Received {}'.format(param.type())
                        )

            self.float16_groups.append(float16_params_this_group)              # trace_info : t_15480, t_15872
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)# trace_info : t_15481, t_15873
            self.fp32_from_fp32_groups.append(fp32_params_this_group)          # trace_info : t_15482, t_15874

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:                                      # trace_info : t_17724, t_17747, t_17782, t_21402, t_21425, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17725, t_17748, t_21403, t_21426, t_25131, ...
        for group in self.fp32_from_float16_groups:                            # trace_info : t_17783, t_17806, t_17841, t_21461, t_21504, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17784, t_17807, t_21462, t_21505, t_25190, ...
        for group in self.fp32_from_fp32_groups:                               # trace_info : t_17842, t_17845, t_17848, t_21572, t_21575, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17843, t_17846, t_21573, t_21576, t_25301, ...

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []                                                        # trace_info : t_20126, t_23854, t_27582

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:                       # trace_info : t_20127, t_20159, t_20209, t_23855, t_23887, ...
            for main_param in main_group:                                      # trace_info : t_20128, t_20131, t_20134, t_20137, t_20140, ...
                if main_param.grad is not None:                                # trace_info : t_20129, t_20132, t_20135, t_20138, t_20141, ...
                    main_grads.append(main_param.grad.data)                    # trace_info : t_20130, t_20133, t_20136, t_20139, t_20142, ...

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:                          # trace_info : t_20210, t_20212, t_20214, t_23938, t_23940, ...
            for main_param in main_group:                                      # trace_info : t_20211, t_20213, t_23939, t_23941, t_27667, ...
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads                                                      # trace_info : t_20215, t_23943, t_27671

    def _get_model_and_main_params_data_float16(self):
        model_data = []                                                        # trace_info : t_20905, t_24633, t_28361
        main_data = []                                                         # trace_info : t_20906, t_24634, t_28362
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_20907, t_20939, t_20989, t_24635, t_24667, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_20908, t_20911, t_20914, t_20917, t_20920, ...
                model_data.append(model_param.data)                            # trace_info : t_20909, t_20912, t_20915, t_20918, t_20921, ...
                main_data.append(main_param.data)                              # trace_info : t_20910, t_20913, t_20916, t_20919, t_20922, ...
        return model_data, main_data                                           # trace_info : t_20990, t_24718, t_28446

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_19990, t_20032, t_20098, t_23718, t_23760, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_19991, t_19995, t_19999, t_20003, t_20007, ...
                if hasattr(model_param, 'main_grad'):                          # trace_info : t_19992, t_19996, t_20000, t_20004, t_20008, ...
                    main_param.grad = model_param.main_grad.float()            # trace_info : t_19993, t_19997, t_20001, t_20005, t_20009, ...
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None                                        # trace_info : t_19994, t_19998, t_20002, t_20006, t_20010, ...

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:                         # trace_info : t_20099, t_20101, t_20103, t_23827, t_23829, ...
            for model_param in model_group:                                    # trace_info : t_20100, t_20102, t_23828, t_23830, t_27556, ...
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16() # trace_info : t_20904, t_24632, t_28360
        _multi_tensor_copy_this_to_that(                                       # trace_info : t_20991, t_20993, t_24719, t_24721, t_28447, ...
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf# trace_info : t_20992, t_24720, t_28448
        )

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        if is_loading:
            self.init_state_fn(self.optimizer)

        state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, chain.from_iterable(g for g in self.float16_groups)
        )

        # Convert fp32_from_fp16_params
        assert len(state_dict['fp32_from_fp16_params']) == len(
            state_dict['optimizer']['param_groups']
        )
        state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix=f'optimizer.state.fp32_param',
                )
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict['fp32_from_fp16_params'], state_dict['optimizer']['param_groups']
            )
        ]

        # Convert regular optimizer state
        optim_state_to_sharding_state(state_dict['optimizer'], id_to_sharded_param_map)
        return state_dict

    def load_state_dict(self, state_dict):
        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            logger.info('***WARNING*** loading optimizer from ' 'an old checkpoint ...')
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info(
                    '***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...'
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        # Copy data for the main params.
        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
            self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


class FP32Optimizer(MegatronOptimizer):
    """Float32 optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, config: OptimizerConfig, init_state_fn: Callable,
    ):

        super(FP32Optimizer, self).__init__(
            optimizer, config, init_state_fn,
        )

        self._scale = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""

        timers = self.config.timers

        # Copy main_grads to grads.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad = param.main_grad
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        # Clip gradients.
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        # Update parameters.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        if is_loading:
            self.init_state_fn(self.optimizer)

        state_dict = self.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, self.get_parameters()
        )
        optim_state_to_sharding_state(state_dict, id_to_sharded_param_map)

        return state_dict


class ProxyDict:
    """
    A dictionary-like object that proxies to a list of dictionaries.

    e.g., ProxyDict([{'a': 1}, {'b': 2}]) behaves like:
    {
        (0, 'a'): 1,
        (1, 'b'): 2,
    }
    We use tuples as keys to avoid ambiguity with the keys of the inner dicts.
    """

    def __init__(self, inner_dicts: List[dict]):
        self._inner_dicts = inner_dicts

    def __getitem__(self, key: Tuple[int, str]):
        idx, inner_key = key
        return self._inner_dicts[idx].get(inner_key)

    def __setitem__(self, key: Tuple[int, str], value: Any):
        idx, inner_key = key
        self._inner_dicts[idx][inner_key] = value

    def __len__(self) -> int:
        return sum([len(inner_dict) for inner_dict in self._inner_dicts])

    def __iter__(self):
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key in inner_dict:
                yield (idx, inner_key)

    def items(self):
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key, value in inner_dict.items():
                yield (idx, inner_key), value


class ChainedOptimizer(MegatronOptimizer):
    """ChainedOptimizer is designed for a collection of optimizers.

    These optimizers are responsible for different parts of multiple models for
    a training task and will be executed one-by-one when the model is updated.

    Args:
        chained_optimizers: a list of optimizers.
    """

    def __init__(self, chained_optimizers: List[MegatronOptimizer]):
        self.chained_optimizers = chained_optimizers

    @property
    def param_groups(self) -> List[dict]:
        param_groups = []
        for optimizer in self.chained_optimizers:
            param_groups += optimizer.param_groups
        return param_groups

    @property
    def state(self) -> ProxyDict:
        """
        Return optimizer state with tuple keys, where the first element is the
        index of the optimizer in the list of chained optimizers.
        """
        return ProxyDict([opt.state for opt in self.chained_optimizers])

    def zero_grad(self, set_to_none=True):
        for optimizer in self.chained_optimizers:
            optimizer.zero_grad(set_to_none)

    def get_loss_scale(self):
        return self.chained_optimizers[0].get_loss_scale()

    def reload_model_params(self):
        for optimizer in self.chained_optimizers:
            optimizer.reload_model_params()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.chained_optimizers]

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        sharded_state_dict = {}
        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            optim_state_dict = optimizer.sharded_state_dict(
                model_sharded_state_dict, is_loading, **kwargs
            )
            add_prefix_for_sharding(optim_state_dict, f'chained_{optimizer_idx}.')
            sharded_state_dict[optimizer_idx] = optim_state_dict
        return sharded_state_dict

    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) != len(state_dict):
            raise RuntimeError(
                f'Expected {len(self.chained_optimizers)} entries'
                f' in state dict, but got {len(state_dict)}.'
            )
        if isinstance(state_dict, dict):
            state_dict = (v for k, v in sorted(state_dict.items()))
        for optimizer, state in zip(self.chained_optimizers, state_dict):
            optimizer.load_state_dict(state)

    def disable_pre_hook(self):
        for optimizer in self.chained_optimizers:
            if (
                not optimizer.config.use_distributed_optimizer
                or not optimizer.config.overlap_param_gather
            ):
                raise ValueError(
                    "disable_pre_hook should only be called with 'use_distributed_optimizer' "
                    "and 'overlap_param_gather' both enabled."
                )
            optimizer.disable_pre_hook()

    def enable_pre_hook(self):
        for optimizer in self.chained_optimizers:
            if (
                not optimizer.config.use_distributed_optimizer
                or not optimizer.config.overlap_param_gather
            ):
                raise ValueError(
                    "enable_pre_hook should only be called with 'use_distributed_optimizer' "
                    "and 'overlap_param_gather' both enabled."
                )
            optimizer.enable_pre_hook()

    def step(self):
        """ChainedOptimizer will step all optimizers one by one.
        """

        update_successful, grad_norm, num_zeros_in_grad = True, 0, 0
        grad_norms = []
        for optimizer in self.chained_optimizers:
            _update_successful, _grad_norm, _num_zeros_in_grad = optimizer.step()
            update_successful &= _update_successful
            grad_norms += [_grad_norm if _grad_norm else 0.0]
            num_zeros_in_grad += _num_zeros_in_grad if _num_zeros_in_grad else 0
        grad_norm = math.sqrt(sum([x ** 2 for x in grad_norms]))

        return update_successful, grad_norm, num_zeros_in_grad

    def save_parameter_state(self, filename: str):
        """Save the distributed parameter states of all optimizers to a file.

        Args:
            filename (str): path to save parameter state to.
        """
        save_states = False
        states = []
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'get_parameter_state_dp_zero'):
                state_dict = optimizer.get_parameter_state_dp_zero()

                # Save checkpoint economically, only when DP rank = 0, state dict
                # needs to be saved.
                if torch.distributed.get_rank(optimizer.data_parallel_group) == 0:
                    states.append(state_dict)
                    save_states = True
                else:
                    states.append(None)
            else:
                states.append(None)

        if save_states:
            torch.save(states, filename)

    def load_parameter_state(self, filename: str):
        """Load the distributed parameter states of all optimizers from a file.

        Args:
            filename (str): path to load parameter state from.
        """
        states = None
        for idx, optimizer in enumerate(self.chained_optimizers):
            if not hasattr(optimizer, 'load_parameter_state_from_dp_zero'):
                continue

            # Lazy loading checkpoint, state dict is needed only when DP rank = 0.
            if torch.distributed.get_rank(optimizer.data_parallel_group) == 0 and states is None:
                states = torch.load(filename)

            state_dict = states[idx] if states else None
            optimizer.load_parameter_state_from_dp_zero(state_dict)

    def finish_param_sync(self, model_index: int):
        """Finish parameter synchronization for all optimizers.
        """
        for optimizer in self.chained_optimizers:
            optimizer.finish_param_sync(model_index)
