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
    for param in group:                                                        # trace_info : t_14667, t_14669, t_14671, t_14673, t_14675, ...
        if param.grad is not None:                                             # trace_info : t_14668, t_14670, t_14672, t_14674, t_14676, ...
            if set_to_none:                                                    # trace_info : t_18317, t_18321, t_18325, t_18329, t_18333, ...
                param.grad = None                                              # trace_info : t_18318, t_18322, t_18326, t_18330, t_18334, ...
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
    if overflow_buf:                                                           # trace_info : t_17813, t_21452, t_25091
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):                                   # trace_info : t_17814, t_17816, t_17818, t_17820, t_17822, ...
            that_.copy_(this_)                                                 # trace_info : t_17815, t_17817, t_17819, t_17821, t_17823, ...


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
        self.optimizer = optimizer                                             # trace_info : t_12107
        assert self.optimizer, 'no optimizer is provided.'                     # trace_info : t_12108
        self.config = config                                                   # trace_info : t_12109
        self.init_state_fn = init_state_fn                                     # trace_info : t_12110

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []                                                            # trace_info : t_17022, t_17087, t_20661, t_20726, t_24300, ...
        for param_group in self.optimizer.param_groups:                        # trace_info : t_17023, t_17045, t_17083, t_17088, t_17110, ...
            for param in param_group['params']:                                # trace_info : t_17024, t_17026, t_17028, t_17030, t_17032, ...
                params.append(param)                                           # trace_info : t_17025, t_17027, t_17029, t_17031, t_17033, ...
        return params                                                          # trace_info : t_17084, t_17149, t_20723, t_20788, t_24362, ...

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()                                         # trace_info : t_17086, t_20725, t_24364
        grads_for_norm = []                                                    # trace_info : t_17150, t_20789, t_24428
        for param in params:                                                   # trace_info : t_17151, t_17160, t_17176, t_17185, t_17194, ...
            grad = param.grad                                                  # trace_info : t_17152, t_17161, t_17177, t_17186, t_17195, ...
            grad_not_none = grad is not None                                   # trace_info : t_17153, t_17162, t_17178, t_17187, t_17196, ...
            is_not_shared = param_is_not_shared(param)                         # trace_info : t_17154, t_17163, t_17179, t_17188, t_17197, ...
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)# trace_info : t_17156, t_17165, t_17181, t_17190, t_17199, ...
            if grad_not_none and is_not_shared and is_not_tp_duplicate:        # trace_info : t_17158, t_17174, t_17183, t_17192, t_17201, ...
                grads_for_norm.append(grad)                                    # trace_info : t_17159, t_17175, t_17184, t_17193, t_17202, ...

        return grads_for_norm                                                  # trace_info : t_17509, t_21148, t_24787

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Default returned here, but the distributed optimizer overrides this."""
        return parallel_state.get_model_parallel_group()                       # trace_info : t_16988, t_17512, t_20627, t_21151, t_24266, ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm."""
        params = self.get_parameters()                                         # trace_info : t_17021, t_20660, t_24299
        grads_for_norm = self.get_main_grads_for_grad_norm()                   # trace_info : t_17085, t_20724, t_24363
        return clip_grad_norm_fp32(                                            # trace_info : t_17510, t_17515, t_21149, t_21154, t_24788, ...
            params, grads_for_norm, clip_grad, model_parallel_group=self.get_model_parallel_group(),# trace_info : t_17511, t_21150, t_24789
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
        return self.get_loss_scale() * loss                                    # trace_info : t_16544, t_20183, t_23822

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
        return self.optimizer.param_groups                                     # trace_info : t_12871, t_17906, t_18027, t_21545, t_21666, ...

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

        super().__init__(                                                      # trace_info : t_12104, t_12106
            optimizer, config, init_state_fn,                                  # trace_info : t_12105
        )
        self.grad_scaler = grad_scaler                                         # trace_info : t_12111

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:                                           # trace_info : t_12112
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:                                                   # trace_info : t_12113
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')# trace_info : t_12114

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:                                                   # trace_info : t_12115
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_12116

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:                                           # trace_info : t_12117
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:                                           # trace_info : t_16545, t_18018, t_20184, t_21657, t_23823, ...
            return self._scale_one
        return self.grad_scaler.scale                                          # trace_info : t_16546, t_18019, t_20185, t_21658, t_23824, ...

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()              # trace_info : t_16884, t_20523, t_24162

        # Reset found inf.
        self.found_inf.fill_(0.0)                                              # trace_info : t_16981, t_20620, t_24259

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(                      # trace_info : t_16982, t_16985, t_20621, t_20624, t_24260, ...
            main_grads, self.found_inf, self.grad_scaler.inv_scale             # trace_info : t_16983, t_20622, t_24261
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(                                          # trace_info : t_16986, t_16991, t_20625, t_20630, t_24264, ...
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()# trace_info : t_16987, t_20626, t_24265
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0                             # trace_info : t_16992, t_20631, t_24270

        return found_inf_flag                                                  # trace_info : t_16993, t_20632, t_24271

    @torch.no_grad()
    def step(self):

        timers = self.config.timers                                            # trace_info : t_16729, t_20368, t_24007

        # Copy gradients from model params to main params.
        if timers is not None:                                                 # trace_info : t_16730, t_20369, t_24008
            timers('optimizer-copy-to-main-grad', log_level=1).start(          # trace_info : t_16731, t_16738, t_20370, t_20377, t_24009, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_16737, t_20376, t_24015
            )
        self._copy_model_grads_to_main_grads()                                 # trace_info : t_16740, t_20379, t_24018
        if timers is not None:                                                 # trace_info : t_16863, t_20502, t_24141
            timers('optimizer-copy-to-main-grad').stop()                       # trace_info : t_16864, t_20503, t_24142

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:                                                   # trace_info : t_16872, t_20511, t_24150

            # Unscale and check for inf/nan.
            if timers is not None:                                             # trace_info : t_16873, t_20512, t_24151
                timers('optimizer-unscale-and-check-inf', log_level=1).start(  # trace_info : t_16874, t_16881, t_20513, t_20520, t_24152, ...
                    barrier=self.config.barrier_with_L1_time                   # trace_info : t_16880, t_20519, t_24158
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()      # trace_info : t_16883, t_20522, t_24161
            if timers is not None:                                             # trace_info : t_16994, t_20633, t_24272
                timers('optimizer-unscale-and-check-inf').stop()               # trace_info : t_16995, t_20634, t_24273

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)                            # trace_info : t_17003, t_20642, t_24281

            # If we found inf/nan, skip the update.
            if found_inf_flag:                                                 # trace_info : t_17007, t_20646, t_24285
                return False, None, None

        # Clip the main gradients.
        if timers is not None:                                                 # trace_info : t_17008, t_20647, t_24286
            timers('optimizer-clip-main-grad', log_level=1).start(             # trace_info : t_17009, t_17016, t_20648, t_20655, t_24287, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_17015, t_20654, t_24293
            )
        grad_norm = None                                                       # trace_info : t_17018, t_20657, t_24296
        if self.config.clip_grad > 0.0:                                        # trace_info : t_17019, t_20658, t_24297
            grad_norm = self.clip_grad_norm(self.config.clip_grad)             # trace_info : t_17020, t_20659, t_24298
        if timers is not None:                                                 # trace_info : t_17657, t_21296, t_24935
            timers('optimizer-clip-main-grad').stop()                          # trace_info : t_17658, t_21297, t_24936

        # Count the zeros in the grads.
        if timers is not None:                                                 # trace_info : t_17666, t_21305, t_24944
            timers('optimizer-count-zeros', log_level=1).start(                # trace_info : t_17667, t_17674, t_21306, t_21313, t_24945, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_17673, t_21312, t_24951
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None# trace_info : t_17676, t_21315, t_24954
        if timers is not None:                                                 # trace_info : t_17677, t_21316, t_24955
            timers('optimizer-count-zeros').stop()                             # trace_info : t_17678, t_21317, t_24956

        # Step the optimizer.
        if timers is not None:                                                 # trace_info : t_17686, t_21325, t_24964
            timers('optimizer-inner-step', log_level=1).start(                 # trace_info : t_17687, t_17694, t_21326, t_21333, t_24965, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_17693, t_21332, t_24971
            )
        self.optimizer.step()                                                  # trace_info : t_17696, t_21335, t_24974
        if timers is not None:                                                 # trace_info : t_17697, t_21336, t_24975
            timers('optimizer-inner-step').stop()                              # trace_info : t_17698, t_21337, t_24976

        # Update params from main params.
        if timers is not None:                                                 # trace_info : t_17706, t_21345, t_24984
            timers('optimizer-copy-main-to-model-params', log_level=1).start(  # trace_info : t_17707, t_17714, t_21346, t_21353, t_24985, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_17713, t_21352, t_24991
            )
        self._copy_main_params_to_model_params()                               # trace_info : t_17716, t_21355, t_24994
        if timers is not None:                                                 # trace_info : t_17871, t_21510, t_25149
            timers('optimizer-copy-main-to-model-params').stop()               # trace_info : t_17872, t_21511, t_25150

        # Successful update.
        return True, grad_norm, num_zeros_in_grad                              # trace_info : t_17880, t_21519, t_25158


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

        super().__init__(                                                      # trace_info : t_12101, t_12103
            optimizer, config, grad_scaler, init_state_fn,                     # trace_info : t_12102
        )

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []                                               # trace_info : t_12118
        self.fp32_from_float16_groups = []                                     # trace_info : t_12119
        self.fp32_from_fp32_groups = []                                        # trace_info : t_12120

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:                        # trace_info : t_12121, t_12369, t_12809
            float16_params_this_group = []                                     # trace_info : t_12122, t_12370
            fp32_params_this_group = []                                        # trace_info : t_12123, t_12371
            fp32_from_float16_params_this_group = []                           # trace_info : t_12124, t_12372
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):                  # trace_info : t_12125, t_12149, t_12173, t_12197, t_12221, ...
                if param.requires_grad:                                        # trace_info : t_12126, t_12150, t_12174, t_12198, t_12222, ...

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:# trace_info : t_12127, t_12151, t_12175, t_12199, t_12223, ...
                        float16_params_this_group.append(param)                # trace_info : t_12128, t_12152, t_12176, t_12200, t_12224, ...
                        # Create a copy
                        main_param = param.detach().clone().float()            # trace_info : t_12129, t_12153, t_12177, t_12201, t_12225, ...
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)# trace_info : t_12130, t_12154, t_12178, t_12202, t_12226, ...
                        if hasattr(param, 'shared'):                           # trace_info : t_12145, t_12169, t_12193, t_12217, t_12241, ...
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param                  # trace_info : t_12146, t_12170, t_12194, t_12218, t_12242, ...

                        fp32_from_float16_params_this_group.append(main_param) # trace_info : t_12147, t_12171, t_12195, t_12219, t_12243, ...
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:                      # trace_info : t_12148, t_12172, t_12196, t_12220, t_12244, ...
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

            self.float16_groups.append(float16_params_this_group)              # trace_info : t_12366, t_12806
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)# trace_info : t_12367, t_12807
            self.fp32_from_fp32_groups.append(fp32_params_this_group)          # trace_info : t_12368, t_12808

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:                                      # trace_info : t_14665, t_14688, t_14727, t_18250, t_18273, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_14666, t_14689, t_18251, t_18274, t_21890, ...
        for group in self.fp32_from_float16_groups:                            # trace_info : t_14728, t_14751, t_14790, t_18313, t_18356, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_14729, t_14752, t_18314, t_18357, t_21953, ...
        for group in self.fp32_from_fp32_groups:                               # trace_info : t_14791, t_14794, t_14797, t_18432, t_18435, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_14792, t_14795, t_18433, t_18436, t_22072, ...

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []                                                        # trace_info : t_16885, t_20524, t_24163

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:                       # trace_info : t_16886, t_16918, t_16974, t_20525, t_20557, ...
            for main_param in main_group:                                      # trace_info : t_16887, t_16890, t_16893, t_16896, t_16899, ...
                if main_param.grad is not None:                                # trace_info : t_16888, t_16891, t_16894, t_16897, t_16900, ...
                    main_grads.append(main_param.grad.data)                    # trace_info : t_16889, t_16892, t_16895, t_16898, t_16901, ...

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:                          # trace_info : t_16975, t_16977, t_16979, t_20614, t_20616, ...
            for main_param in main_group:                                      # trace_info : t_16976, t_16978, t_20615, t_20617, t_24254, ...
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads                                                      # trace_info : t_16980, t_20619, t_24258

    def _get_model_and_main_params_data_float16(self):
        model_data = []                                                        # trace_info : t_17718, t_21357, t_24996
        main_data = []                                                         # trace_info : t_17719, t_21358, t_24997
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_17720, t_17752, t_17808, t_21359, t_21391, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_17721, t_17724, t_17727, t_17730, t_17733, ...
                model_data.append(model_param.data)                            # trace_info : t_17722, t_17725, t_17728, t_17731, t_17734, ...
                main_data.append(main_param.data)                              # trace_info : t_17723, t_17726, t_17729, t_17732, t_17735, ...
        return model_data, main_data                                           # trace_info : t_17809, t_21448, t_25087

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_16741, t_16783, t_16857, t_20380, t_20422, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_16742, t_16746, t_16750, t_16754, t_16758, ...
                if hasattr(model_param, 'main_grad'):                          # trace_info : t_16743, t_16747, t_16751, t_16755, t_16759, ...
                    main_param.grad = model_param.main_grad.float()            # trace_info : t_16744, t_16748, t_16752, t_16756, t_16760, ...
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None                                        # trace_info : t_16745, t_16749, t_16753, t_16757, t_16761, ...

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:                         # trace_info : t_16858, t_16860, t_16862, t_20497, t_20499, ...
            for model_param in model_group:                                    # trace_info : t_16859, t_16861, t_20498, t_20500, t_24137, ...
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16() # trace_info : t_17717, t_21356, t_24995
        _multi_tensor_copy_this_to_that(                                       # trace_info : t_17810, t_17812, t_21449, t_21451, t_25088, ...
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf# trace_info : t_17811, t_21450, t_25089
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
