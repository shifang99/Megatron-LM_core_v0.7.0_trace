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
    for param in group:                                                        # trace_info : t_17525, t_17527, t_17529, t_17531, t_17533, ...
        if param.grad is not None:                                             # trace_info : t_17526, t_17528, t_17530, t_17532, t_17534, ...
            if set_to_none:                                                    # trace_info : t_20720, t_20724, t_20728, t_20732, t_20736, ...
                param.grad = None                                              # trace_info : t_20721, t_20725, t_20729, t_20733, t_20737, ...
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
    if overflow_buf:                                                           # trace_info : t_20216, t_23402, t_26588
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):                                   # trace_info : t_20217, t_20219, t_20221, t_20223, t_20225, ...
            that_.copy_(this_)                                                 # trace_info : t_20218, t_20220, t_20222, t_20224, t_20226, ...


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
        self.optimizer = optimizer                                             # trace_info : t_14966
        assert self.optimizer, 'no optimizer is provided.'                     # trace_info : t_14967
        self.config = config                                                   # trace_info : t_14968
        self.init_state_fn = init_state_fn                                     # trace_info : t_14969

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []                                                            # trace_info : t_19425, t_19490, t_22611, t_22676, t_25797, ...
        for param_group in self.optimizer.param_groups:                        # trace_info : t_19426, t_19448, t_19486, t_19491, t_19513, ...
            for param in param_group['params']:                                # trace_info : t_19427, t_19429, t_19431, t_19433, t_19435, ...
                params.append(param)                                           # trace_info : t_19428, t_19430, t_19432, t_19434, t_19436, ...
        return params                                                          # trace_info : t_19487, t_19552, t_22673, t_22738, t_25859, ...

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()                                         # trace_info : t_19489, t_22675, t_25861
        grads_for_norm = []                                                    # trace_info : t_19553, t_22739, t_25925
        for param in params:                                                   # trace_info : t_19554, t_19563, t_19579, t_19588, t_19597, ...
            grad = param.grad                                                  # trace_info : t_19555, t_19564, t_19580, t_19589, t_19598, ...
            grad_not_none = grad is not None                                   # trace_info : t_19556, t_19565, t_19581, t_19590, t_19599, ...
            is_not_shared = param_is_not_shared(param)                         # trace_info : t_19557, t_19566, t_19582, t_19591, t_19600, ...
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)# trace_info : t_19559, t_19568, t_19584, t_19593, t_19602, ...
            if grad_not_none and is_not_shared and is_not_tp_duplicate:        # trace_info : t_19561, t_19577, t_19586, t_19595, t_19604, ...
                grads_for_norm.append(grad)                                    # trace_info : t_19562, t_19578, t_19587, t_19596, t_19605, ...

        return grads_for_norm                                                  # trace_info : t_19912, t_23098, t_26284

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Default returned here, but the distributed optimizer overrides this."""
        return parallel_state.get_model_parallel_group()                       # trace_info : t_19391, t_19915, t_22577, t_23101, t_25763, ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm."""
        params = self.get_parameters()                                         # trace_info : t_19424, t_22610, t_25796
        grads_for_norm = self.get_main_grads_for_grad_norm()                   # trace_info : t_19488, t_22674, t_25860
        return clip_grad_norm_fp32(                                            # trace_info : t_19913, t_19918, t_23099, t_23104, t_26285, ...
            params, grads_for_norm, clip_grad, model_parallel_group=self.get_model_parallel_group(),# trace_info : t_19914, t_23100, t_26286
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
        return self.get_loss_scale() * loss                                    # trace_info : t_18942, t_22128, t_25314

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
        return self.optimizer.param_groups                                     # trace_info : t_15730, t_20309, t_20430, t_23495, t_23616, ...

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

        super().__init__(                                                      # trace_info : t_14963, t_14965
            optimizer, config, init_state_fn,                                  # trace_info : t_14964
        )
        self.grad_scaler = grad_scaler                                         # trace_info : t_14970

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:                                           # trace_info : t_14971
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:                                                   # trace_info : t_14972
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')# trace_info : t_14973

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:                                                   # trace_info : t_14974
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_14975

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:                                           # trace_info : t_14976
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:                                           # trace_info : t_18943, t_20421, t_22129, t_23607, t_25315, ...
            return self._scale_one
        return self.grad_scaler.scale                                          # trace_info : t_18944, t_20422, t_22130, t_23608, t_25316, ...

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()              # trace_info : t_19287, t_22473, t_25659

        # Reset found inf.
        self.found_inf.fill_(0.0)                                              # trace_info : t_19384, t_22570, t_25756

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(                      # trace_info : t_19385, t_19388, t_22571, t_22574, t_25757, ...
            main_grads, self.found_inf, self.grad_scaler.inv_scale             # trace_info : t_19386, t_22572, t_25758
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(                                          # trace_info : t_19389, t_19394, t_22575, t_22580, t_25761, ...
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()# trace_info : t_19390, t_22576, t_25762
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0                             # trace_info : t_19395, t_22581, t_25767

        return found_inf_flag                                                  # trace_info : t_19396, t_22582, t_25768

    @torch.no_grad()
    def step(self):

        timers = self.config.timers                                            # trace_info : t_19132, t_22318, t_25504

        # Copy gradients from model params to main params.
        if timers is not None:                                                 # trace_info : t_19133, t_22319, t_25505
            timers('optimizer-copy-to-main-grad', log_level=1).start(          # trace_info : t_19134, t_19141, t_22320, t_22327, t_25506, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_19140, t_22326, t_25512
            )
        self._copy_model_grads_to_main_grads()                                 # trace_info : t_19143, t_22329, t_25515
        if timers is not None:                                                 # trace_info : t_19266, t_22452, t_25638
            timers('optimizer-copy-to-main-grad').stop()                       # trace_info : t_19267, t_22453, t_25639

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:                                                   # trace_info : t_19275, t_22461, t_25647

            # Unscale and check for inf/nan.
            if timers is not None:                                             # trace_info : t_19276, t_22462, t_25648
                timers('optimizer-unscale-and-check-inf', log_level=1).start(  # trace_info : t_19277, t_19284, t_22463, t_22470, t_25649, ...
                    barrier=self.config.barrier_with_L1_time                   # trace_info : t_19283, t_22469, t_25655
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()      # trace_info : t_19286, t_22472, t_25658
            if timers is not None:                                             # trace_info : t_19397, t_22583, t_25769
                timers('optimizer-unscale-and-check-inf').stop()               # trace_info : t_19398, t_22584, t_25770

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)                            # trace_info : t_19406, t_22592, t_25778

            # If we found inf/nan, skip the update.
            if found_inf_flag:                                                 # trace_info : t_19410, t_22596, t_25782
                return False, None, None

        # Clip the main gradients.
        if timers is not None:                                                 # trace_info : t_19411, t_22597, t_25783
            timers('optimizer-clip-main-grad', log_level=1).start(             # trace_info : t_19412, t_19419, t_22598, t_22605, t_25784, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_19418, t_22604, t_25790
            )
        grad_norm = None                                                       # trace_info : t_19421, t_22607, t_25793
        if self.config.clip_grad > 0.0:                                        # trace_info : t_19422, t_22608, t_25794
            grad_norm = self.clip_grad_norm(self.config.clip_grad)             # trace_info : t_19423, t_22609, t_25795
        if timers is not None:                                                 # trace_info : t_20060, t_23246, t_26432
            timers('optimizer-clip-main-grad').stop()                          # trace_info : t_20061, t_23247, t_26433

        # Count the zeros in the grads.
        if timers is not None:                                                 # trace_info : t_20069, t_23255, t_26441
            timers('optimizer-count-zeros', log_level=1).start(                # trace_info : t_20070, t_20077, t_23256, t_23263, t_26442, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20076, t_23262, t_26448
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None# trace_info : t_20079, t_23265, t_26451
        if timers is not None:                                                 # trace_info : t_20080, t_23266, t_26452
            timers('optimizer-count-zeros').stop()                             # trace_info : t_20081, t_23267, t_26453

        # Step the optimizer.
        if timers is not None:                                                 # trace_info : t_20089, t_23275, t_26461
            timers('optimizer-inner-step', log_level=1).start(                 # trace_info : t_20090, t_20097, t_23276, t_23283, t_26462, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20096, t_23282, t_26468
            )
        self.optimizer.step()                                                  # trace_info : t_20099, t_23285, t_26471
        if timers is not None:                                                 # trace_info : t_20100, t_23286, t_26472
            timers('optimizer-inner-step').stop()                              # trace_info : t_20101, t_23287, t_26473

        # Update params from main params.
        if timers is not None:                                                 # trace_info : t_20109, t_23295, t_26481
            timers('optimizer-copy-main-to-model-params', log_level=1).start(  # trace_info : t_20110, t_20117, t_23296, t_23303, t_26482, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20116, t_23302, t_26488
            )
        self._copy_main_params_to_model_params()                               # trace_info : t_20119, t_23305, t_26491
        if timers is not None:                                                 # trace_info : t_20274, t_23460, t_26646
            timers('optimizer-copy-main-to-model-params').stop()               # trace_info : t_20275, t_23461, t_26647

        # Successful update.
        return True, grad_norm, num_zeros_in_grad                              # trace_info : t_20283, t_23469, t_26655


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

        super().__init__(                                                      # trace_info : t_14960, t_14962
            optimizer, config, grad_scaler, init_state_fn,                     # trace_info : t_14961
        )

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []                                               # trace_info : t_14977
        self.fp32_from_float16_groups = []                                     # trace_info : t_14978
        self.fp32_from_fp32_groups = []                                        # trace_info : t_14979

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:                        # trace_info : t_14980, t_15228, t_15668
            float16_params_this_group = []                                     # trace_info : t_14981, t_15229
            fp32_params_this_group = []                                        # trace_info : t_14982, t_15230
            fp32_from_float16_params_this_group = []                           # trace_info : t_14983, t_15231
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):                  # trace_info : t_14984, t_15008, t_15032, t_15056, t_15080, ...
                if param.requires_grad:                                        # trace_info : t_14985, t_15009, t_15033, t_15057, t_15081, ...

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:# trace_info : t_14986, t_15010, t_15034, t_15058, t_15082, ...
                        float16_params_this_group.append(param)                # trace_info : t_14987, t_15011, t_15035, t_15059, t_15083, ...
                        # Create a copy
                        main_param = param.detach().clone().float()            # trace_info : t_14988, t_15012, t_15036, t_15060, t_15084, ...
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)# trace_info : t_14989, t_15013, t_15037, t_15061, t_15085, ...
                        if hasattr(param, 'shared'):                           # trace_info : t_15004, t_15028, t_15052, t_15076, t_15100, ...
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param                  # trace_info : t_15005, t_15029, t_15053, t_15077, t_15101, ...

                        fp32_from_float16_params_this_group.append(main_param) # trace_info : t_15006, t_15030, t_15054, t_15078, t_15102, ...
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:                      # trace_info : t_15007, t_15031, t_15055, t_15079, t_15103, ...
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

            self.float16_groups.append(float16_params_this_group)              # trace_info : t_15225, t_15665
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)# trace_info : t_15226, t_15666
            self.fp32_from_fp32_groups.append(fp32_params_this_group)          # trace_info : t_15227, t_15667

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:                                      # trace_info : t_17523, t_17546, t_17585, t_20653, t_20676, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17524, t_17547, t_20654, t_20677, t_23840, ...
        for group in self.fp32_from_float16_groups:                            # trace_info : t_17586, t_17609, t_17648, t_20716, t_20759, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17587, t_17610, t_20717, t_20760, t_23903, ...
        for group in self.fp32_from_fp32_groups:                               # trace_info : t_17649, t_17652, t_17655, t_20835, t_20838, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17650, t_17653, t_20836, t_20839, t_24022, ...

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []                                                        # trace_info : t_19288, t_22474, t_25660

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:                       # trace_info : t_19289, t_19321, t_19377, t_22475, t_22507, ...
            for main_param in main_group:                                      # trace_info : t_19290, t_19293, t_19296, t_19299, t_19302, ...
                if main_param.grad is not None:                                # trace_info : t_19291, t_19294, t_19297, t_19300, t_19303, ...
                    main_grads.append(main_param.grad.data)                    # trace_info : t_19292, t_19295, t_19298, t_19301, t_19304, ...

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:                          # trace_info : t_19378, t_19380, t_19382, t_22564, t_22566, ...
            for main_param in main_group:                                      # trace_info : t_19379, t_19381, t_22565, t_22567, t_25751, ...
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads                                                      # trace_info : t_19383, t_22569, t_25755

    def _get_model_and_main_params_data_float16(self):
        model_data = []                                                        # trace_info : t_20121, t_23307, t_26493
        main_data = []                                                         # trace_info : t_20122, t_23308, t_26494
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_20123, t_20155, t_20211, t_23309, t_23341, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_20124, t_20127, t_20130, t_20133, t_20136, ...
                model_data.append(model_param.data)                            # trace_info : t_20125, t_20128, t_20131, t_20134, t_20137, ...
                main_data.append(main_param.data)                              # trace_info : t_20126, t_20129, t_20132, t_20135, t_20138, ...
        return model_data, main_data                                           # trace_info : t_20212, t_23398, t_26584

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_19144, t_19186, t_19260, t_22330, t_22372, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_19145, t_19149, t_19153, t_19157, t_19161, ...
                if hasattr(model_param, 'main_grad'):                          # trace_info : t_19146, t_19150, t_19154, t_19158, t_19162, ...
                    main_param.grad = model_param.main_grad.float()            # trace_info : t_19147, t_19151, t_19155, t_19159, t_19163, ...
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None                                        # trace_info : t_19148, t_19152, t_19156, t_19160, t_19164, ...

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:                         # trace_info : t_19261, t_19263, t_19265, t_22447, t_22449, ...
            for model_param in model_group:                                    # trace_info : t_19262, t_19264, t_22448, t_22450, t_25634, ...
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16() # trace_info : t_20120, t_23306, t_26492
        _multi_tensor_copy_this_to_that(                                       # trace_info : t_20213, t_20215, t_23399, t_23401, t_26585, ...
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf# trace_info : t_20214, t_23400, t_26586
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
