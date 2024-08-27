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
    for param in group:                                                        # trace_info : t_17803, t_17805, t_17807, t_17809, t_17811, ...
        if param.grad is not None:                                             # trace_info : t_17804, t_17806, t_17808, t_17810, t_17812, ...
            if set_to_none:                                                    # trace_info : t_21451, t_21455, t_21459, t_21463, t_21467, ...
                param.grad = None                                              # trace_info : t_21452, t_21456, t_21460, t_21464, t_21468, ...
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
    if overflow_buf:                                                           # trace_info : t_20946, t_24583, t_92190
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):                                   # trace_info : t_20947, t_20949, t_20951, t_20953, t_20955, ...
            that_.copy_(this_)                                                 # trace_info : t_20948, t_20950, t_20952, t_20954, t_20956, ...


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
        self.optimizer = optimizer                                             # trace_info : t_15136
        assert self.optimizer, 'no optimizer is provided.'                     # trace_info : t_15137
        self.config = config                                                   # trace_info : t_15138
        self.init_state_fn = init_state_fn                                     # trace_info : t_15139

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []                                                            # trace_info : t_20155, t_20220, t_23792, t_23857, t_91399, ...
        for param_group in self.optimizer.param_groups:                        # trace_info : t_20156, t_20178, t_20216, t_20221, t_20243, ...
            for param in param_group['params']:                                # trace_info : t_20157, t_20159, t_20161, t_20163, t_20165, ...
                params.append(param)                                           # trace_info : t_20158, t_20160, t_20162, t_20164, t_20166, ...
        return params                                                          # trace_info : t_20217, t_20282, t_23854, t_23919, t_91461, ...

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()                                         # trace_info : t_20219, t_23856, t_91463
        grads_for_norm = []                                                    # trace_info : t_20283, t_23920, t_91527
        for param in params:                                                   # trace_info : t_20284, t_20293, t_20309, t_20318, t_20327, ...
            grad = param.grad                                                  # trace_info : t_20285, t_20294, t_20310, t_20319, t_20328, ...
            grad_not_none = grad is not None                                   # trace_info : t_20286, t_20295, t_20311, t_20320, t_20329, ...
            is_not_shared = param_is_not_shared(param)                         # trace_info : t_20287, t_20296, t_20312, t_20321, t_20330, ...
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)# trace_info : t_20289, t_20298, t_20314, t_20323, t_20332, ...
            if grad_not_none and is_not_shared and is_not_tp_duplicate:        # trace_info : t_20291, t_20307, t_20316, t_20325, t_20334, ...
                grads_for_norm.append(grad)                                    # trace_info : t_20292, t_20308, t_20317, t_20326, t_20335, ...

        return grads_for_norm                                                  # trace_info : t_20642, t_24279, t_91886

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Default returned here, but the distributed optimizer overrides this."""
        return parallel_state.get_model_parallel_group()                       # trace_info : t_20121, t_20645, t_23758, t_24282, t_91365, ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm."""
        params = self.get_parameters()                                         # trace_info : t_20154, t_23791, t_91398
        grads_for_norm = self.get_main_grads_for_grad_norm()                   # trace_info : t_20218, t_23855, t_91462
        return clip_grad_norm_fp32(                                            # trace_info : t_20643, t_20648, t_24280, t_24285, t_91887, ...
            params, grads_for_norm, clip_grad, model_parallel_group=self.get_model_parallel_group(),# trace_info : t_20644, t_24281, t_91888
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
        return self.get_loss_scale() * loss                                    # trace_info : t_19672, t_23309, t_90916

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
        return self.optimizer.param_groups                                     # trace_info : t_15900, t_21039, t_21160, t_24676, t_24797, ...

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

        super().__init__(                                                      # trace_info : t_15133, t_15135
            optimizer, config, init_state_fn,                                  # trace_info : t_15134
        )
        self.grad_scaler = grad_scaler                                         # trace_info : t_15140

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:                                           # trace_info : t_15141
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:                                                   # trace_info : t_15142
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')# trace_info : t_15143

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:                                                   # trace_info : t_15144
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_15145

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:                                           # trace_info : t_15146
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:                                           # trace_info : t_19673, t_21151, t_23310, t_24788, t_90917, ...
            return self._scale_one
        return self.grad_scaler.scale                                          # trace_info : t_19674, t_21152, t_23311, t_24789, t_90918, ...

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()              # trace_info : t_20017, t_23654, t_91261

        # Reset found inf.
        self.found_inf.fill_(0.0)                                              # trace_info : t_20114, t_23751, t_91358

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(                      # trace_info : t_20115, t_20118, t_23752, t_23755, t_91359, ...
            main_grads, self.found_inf, self.grad_scaler.inv_scale             # trace_info : t_20116, t_23753, t_91360
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(                                          # trace_info : t_20119, t_20124, t_23756, t_23761, t_91363, ...
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()# trace_info : t_20120, t_23757, t_91364
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0                             # trace_info : t_20125, t_23762, t_91369

        return found_inf_flag                                                  # trace_info : t_20126, t_23763, t_91370

    @torch.no_grad()
    def step(self):

        timers = self.config.timers                                            # trace_info : t_19862, t_23499, t_91106

        # Copy gradients from model params to main params.
        if timers is not None:                                                 # trace_info : t_19863, t_23500, t_91107
            timers('optimizer-copy-to-main-grad', log_level=1).start(          # trace_info : t_19864, t_19871, t_23501, t_23508, t_91108, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_19870, t_23507, t_91114
            )
        self._copy_model_grads_to_main_grads()                                 # trace_info : t_19873, t_23510, t_91117
        if timers is not None:                                                 # trace_info : t_19996, t_23633, t_91240
            timers('optimizer-copy-to-main-grad').stop()                       # trace_info : t_19997, t_23634, t_91241

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:                                                   # trace_info : t_20005, t_23642, t_91249

            # Unscale and check for inf/nan.
            if timers is not None:                                             # trace_info : t_20006, t_23643, t_91250
                timers('optimizer-unscale-and-check-inf', log_level=1).start(  # trace_info : t_20007, t_20014, t_23644, t_23651, t_91251, ...
                    barrier=self.config.barrier_with_L1_time                   # trace_info : t_20013, t_23650, t_91257
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()      # trace_info : t_20016, t_23653, t_91260
            if timers is not None:                                             # trace_info : t_20127, t_23764, t_91371
                timers('optimizer-unscale-and-check-inf').stop()               # trace_info : t_20128, t_23765, t_91372

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)                            # trace_info : t_20136, t_23773, t_91380

            # If we found inf/nan, skip the update.
            if found_inf_flag:                                                 # trace_info : t_20140, t_23777, t_91384
                return False, None, None

        # Clip the main gradients.
        if timers is not None:                                                 # trace_info : t_20141, t_23778, t_91385
            timers('optimizer-clip-main-grad', log_level=1).start(             # trace_info : t_20142, t_20149, t_23779, t_23786, t_91386, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20148, t_23785, t_91392
            )
        grad_norm = None                                                       # trace_info : t_20151, t_23788, t_91395
        if self.config.clip_grad > 0.0:                                        # trace_info : t_20152, t_23789, t_91396
            grad_norm = self.clip_grad_norm(self.config.clip_grad)             # trace_info : t_20153, t_23790, t_91397
        if timers is not None:                                                 # trace_info : t_20790, t_24427, t_92034
            timers('optimizer-clip-main-grad').stop()                          # trace_info : t_20791, t_24428, t_92035

        # Count the zeros in the grads.
        if timers is not None:                                                 # trace_info : t_20799, t_24436, t_92043
            timers('optimizer-count-zeros', log_level=1).start(                # trace_info : t_20800, t_20807, t_24437, t_24444, t_92044, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20806, t_24443, t_92050
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None# trace_info : t_20809, t_24446, t_92053
        if timers is not None:                                                 # trace_info : t_20810, t_24447, t_92054
            timers('optimizer-count-zeros').stop()                             # trace_info : t_20811, t_24448, t_92055

        # Step the optimizer.
        if timers is not None:                                                 # trace_info : t_20819, t_24456, t_92063
            timers('optimizer-inner-step', log_level=1).start(                 # trace_info : t_20820, t_20827, t_24457, t_24464, t_92064, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20826, t_24463, t_92070
            )
        self.optimizer.step()                                                  # trace_info : t_20829, t_24466, t_92073
        if timers is not None:                                                 # trace_info : t_20830, t_24467, t_92074
            timers('optimizer-inner-step').stop()                              # trace_info : t_20831, t_24468, t_92075

        # Update params from main params.
        if timers is not None:                                                 # trace_info : t_20839, t_24476, t_92083
            timers('optimizer-copy-main-to-model-params', log_level=1).start(  # trace_info : t_20840, t_20847, t_24477, t_24484, t_92084, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20846, t_24483, t_92090
            )
        self._copy_main_params_to_model_params()                               # trace_info : t_20849, t_24486, t_92093
        if timers is not None:                                                 # trace_info : t_21004, t_24641, t_92248
            timers('optimizer-copy-main-to-model-params').stop()               # trace_info : t_21005, t_24642, t_92249

        # Successful update.
        return True, grad_norm, num_zeros_in_grad                              # trace_info : t_21013, t_24650, t_92257


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

        super().__init__(                                                      # trace_info : t_15130, t_15132
            optimizer, config, grad_scaler, init_state_fn,                     # trace_info : t_15131
        )

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []                                               # trace_info : t_15147
        self.fp32_from_float16_groups = []                                     # trace_info : t_15148
        self.fp32_from_fp32_groups = []                                        # trace_info : t_15149

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:                        # trace_info : t_15150, t_15398, t_15838
            float16_params_this_group = []                                     # trace_info : t_15151, t_15399
            fp32_params_this_group = []                                        # trace_info : t_15152, t_15400
            fp32_from_float16_params_this_group = []                           # trace_info : t_15153, t_15401
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):                  # trace_info : t_15154, t_15178, t_15202, t_15226, t_15250, ...
                if param.requires_grad:                                        # trace_info : t_15155, t_15179, t_15203, t_15227, t_15251, ...

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:# trace_info : t_15156, t_15180, t_15204, t_15228, t_15252, ...
                        float16_params_this_group.append(param)                # trace_info : t_15157, t_15181, t_15205, t_15229, t_15253, ...
                        # Create a copy
                        main_param = param.detach().clone().float()            # trace_info : t_15158, t_15182, t_15206, t_15230, t_15254, ...
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)# trace_info : t_15159, t_15183, t_15207, t_15231, t_15255, ...
                        if hasattr(param, 'shared'):                           # trace_info : t_15174, t_15198, t_15222, t_15246, t_15270, ...
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param                  # trace_info : t_15175, t_15199, t_15223, t_15247, t_15271, ...

                        fp32_from_float16_params_this_group.append(main_param) # trace_info : t_15176, t_15200, t_15224, t_15248, t_15272, ...
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:                      # trace_info : t_15177, t_15201, t_15225, t_15249, t_15273, ...
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

            self.float16_groups.append(float16_params_this_group)              # trace_info : t_15395, t_15835
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)# trace_info : t_15396, t_15836
            self.fp32_from_fp32_groups.append(fp32_params_this_group)          # trace_info : t_15397, t_15837

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:                                      # trace_info : t_17801, t_17824, t_17863, t_21384, t_21407, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17802, t_17825, t_21385, t_21408, t_88992, ...
        for group in self.fp32_from_float16_groups:                            # trace_info : t_17864, t_17887, t_17926, t_21447, t_21490, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17865, t_17888, t_21448, t_21491, t_89055, ...
        for group in self.fp32_from_fp32_groups:                               # trace_info : t_17927, t_17930, t_17933, t_21566, t_21569, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_17928, t_17931, t_21567, t_21570, t_89174, ...

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []                                                        # trace_info : t_20018, t_23655, t_91262

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:                       # trace_info : t_20019, t_20051, t_20107, t_23656, t_23688, ...
            for main_param in main_group:                                      # trace_info : t_20020, t_20023, t_20026, t_20029, t_20032, ...
                if main_param.grad is not None:                                # trace_info : t_20021, t_20024, t_20027, t_20030, t_20033, ...
                    main_grads.append(main_param.grad.data)                    # trace_info : t_20022, t_20025, t_20028, t_20031, t_20034, ...

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:                          # trace_info : t_20108, t_20110, t_20112, t_23745, t_23747, ...
            for main_param in main_group:                                      # trace_info : t_20109, t_20111, t_23746, t_23748, t_91353, ...
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads                                                      # trace_info : t_20113, t_23750, t_91357

    def _get_model_and_main_params_data_float16(self):
        model_data = []                                                        # trace_info : t_20851, t_24488, t_92095
        main_data = []                                                         # trace_info : t_20852, t_24489, t_92096
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_20853, t_20885, t_20941, t_24490, t_24522, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_20854, t_20857, t_20860, t_20863, t_20866, ...
                model_data.append(model_param.data)                            # trace_info : t_20855, t_20858, t_20861, t_20864, t_20867, ...
                main_data.append(main_param.data)                              # trace_info : t_20856, t_20859, t_20862, t_20865, t_20868, ...
        return model_data, main_data                                           # trace_info : t_20942, t_24579, t_92186

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_19874, t_19916, t_19990, t_23511, t_23553, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_19875, t_19879, t_19883, t_19887, t_19891, ...
                if hasattr(model_param, 'main_grad'):                          # trace_info : t_19876, t_19880, t_19884, t_19888, t_19892, ...
                    main_param.grad = model_param.main_grad.float()            # trace_info : t_19877, t_19881, t_19885, t_19889, t_19893, ...
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None                                        # trace_info : t_19878, t_19882, t_19886, t_19890, t_19894, ...

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:                         # trace_info : t_19991, t_19993, t_19995, t_23628, t_23630, ...
            for model_param in model_group:                                    # trace_info : t_19992, t_19994, t_23629, t_23631, t_91236, ...
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16() # trace_info : t_20850, t_24487, t_92094
        _multi_tensor_copy_this_to_that(                                       # trace_info : t_20943, t_20945, t_24580, t_24582, t_92187, ...
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf# trace_info : t_20944, t_24581, t_92188
        )

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def state_dict(self):
        state_dict = {}                                                        # trace_info : t_29108, t_96701
        state_dict['optimizer'] = self.optimizer.state_dict()                  # trace_info : t_29109, t_96702
        if self.grad_scaler:                                                   # trace_info : t_29110, t_96703
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()          # trace_info : t_29111, t_96704
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups    # trace_info : t_29117, t_96710
        return state_dict                                                      # trace_info : t_29118, t_96711

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        if is_loading:                                                         # trace_info : t_29106, t_96699
            self.init_state_fn(self.optimizer)

        state_dict = self.state_dict()                                         # trace_info : t_29107, t_96700

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(           # trace_info : t_29119, t_29121, t_96712, t_96714
            model_sharded_state_dict, chain.from_iterable(g for g in self.float16_groups)# trace_info : t_29120, t_29344, t_29375, t_29430, t_96713, ...
        )

        # Convert fp32_from_fp16_params
        assert len(state_dict['fp32_from_fp16_params']) == len(                # trace_info : t_29610, t_29612, t_97203, t_97205
            state_dict['optimizer']['param_groups']                            # trace_info : t_29611, t_97204
        )
        state_dict['fp32_from_fp16_params'] = [                                # trace_info : t_29613, t_29617, t_97206, t_97210
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix=f'optimizer.state.fp32_param',
                )
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(                                # trace_info : t_29614, t_29616, t_97207, t_97209
                state_dict['fp32_from_fp16_params'], state_dict['optimizer']['param_groups']# trace_info : t_29615, t_97208
            )
        ]

        # Convert regular optimizer state
        optim_state_to_sharding_state(state_dict['optimizer'], id_to_sharded_param_map)# trace_info : t_30066, t_97659
        return state_dict                                                      # trace_info : t_31394, t_98987

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
