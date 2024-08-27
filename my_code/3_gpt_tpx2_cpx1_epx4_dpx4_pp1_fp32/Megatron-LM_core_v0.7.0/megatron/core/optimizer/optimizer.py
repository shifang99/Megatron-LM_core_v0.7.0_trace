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
    for param in group:                                                        # trace_info : t_17752, t_17754, t_17756, t_17758, t_17760, ...
        if param.grad is not None:                                             # trace_info : t_17753, t_17755, t_17757, t_17759, t_17761, ...
            if set_to_none:                                                    # trace_info : t_22047, t_22051, t_22055, t_22059, t_22063, ...
                param.grad = None                                              # trace_info : t_22048, t_22052, t_22056, t_22060, t_22064, ...
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
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


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
        self.optimizer = optimizer                                             # trace_info : t_15675, t_15728
        assert self.optimizer, 'no optimizer is provided.'                     # trace_info : t_15676, t_15729
        self.config = config                                                   # trace_info : t_15677, t_15730
        self.init_state_fn = init_state_fn                                     # trace_info : t_15678, t_15731

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []                                                            # trace_info : t_20686, t_20739, t_21329, t_21354, t_25031, ...
        for param_group in self.optimizer.param_groups:                        # trace_info : t_20687, t_20705, t_20735, t_20740, t_20758, ...
            for param in param_group['params']:                                # trace_info : t_20688, t_20690, t_20692, t_20694, t_20696, ...
                params.append(param)                                           # trace_info : t_20689, t_20691, t_20693, t_20695, t_20697, ...
        return params                                                          # trace_info : t_20736, t_20789, t_21351, t_21376, t_25081, ...

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()                                         # trace_info : t_20738, t_21353, t_25083, t_25698, t_29428, ...
        grads_for_norm = []                                                    # trace_info : t_20790, t_21377, t_25135, t_25722, t_29480, ...
        for param in params:                                                   # trace_info : t_20791, t_20800, t_20816, t_20825, t_20834, ...
            grad = param.grad                                                  # trace_info : t_20792, t_20801, t_20817, t_20826, t_20835, ...
            grad_not_none = grad is not None                                   # trace_info : t_20793, t_20802, t_20818, t_20827, t_20836, ...
            is_not_shared = param_is_not_shared(param)                         # trace_info : t_20794, t_20803, t_20819, t_20828, t_20837, ...
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)# trace_info : t_20796, t_20805, t_20821, t_20830, t_20839, ...
            if grad_not_none and is_not_shared and is_not_tp_duplicate:        # trace_info : t_20798, t_20814, t_20823, t_20832, t_20848, ...
                grads_for_norm.append(grad)                                    # trace_info : t_20799, t_20815, t_20824, t_20833, t_20849, ...

        return grads_for_norm                                                  # trace_info : t_21095, t_21465, t_25440, t_25810, t_29785, ...

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Default returned here, but the distributed optimizer overrides this."""
        return parallel_state.get_model_parallel_group()                       # trace_info : t_21098, t_21468, t_25443, t_25813, t_29788, ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm."""
        params = self.get_parameters()                                         # trace_info : t_20685, t_21328, t_25030, t_25673, t_29375, ...
        grads_for_norm = self.get_main_grads_for_grad_norm()                   # trace_info : t_20737, t_21352, t_25082, t_25697, t_29427, ...
        return clip_grad_norm_fp32(                                            # trace_info : t_21096, t_21101, t_21466, t_21471, t_25441, ...
            params, grads_for_norm, clip_grad, model_parallel_group=self.get_model_parallel_group(),# trace_info : t_21097, t_21467, t_25442, t_25812, t_29787, ...
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
        return self.get_loss_scale() * loss                                    # trace_info : t_20071, t_20125, t_24416, t_24470, t_28761, ...

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
        return self.optimizer.param_groups                                     # trace_info : t_15796, t_15799, t_21617, t_21620, t_21775, ...

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

        super().__init__(
            optimizer, config, init_state_fn,
        )
        self.grad_scaler = grad_scaler

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0

        return found_inf_flag

    @torch.no_grad()
    def step(self):

        timers = self.config.timers

        # Copy gradients from model params to main params.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self._copy_model_grads_to_main_grads()
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            if timers is not None:
                timers('optimizer-unscale-and-check-inf', log_level=1).start(
                    barrier=self.config.barrier_with_L1_time
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        # Clip the main gradients.
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

        # Step the optimizer.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        # Update params from main params.
        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self._copy_main_params_to_model_params()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad


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

        super().__init__(
            optimizer, config, grad_scaler, init_state_fn,
        )

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)
                        if hasattr(param, 'shared'):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param

                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
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

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:
            for model_param in model_group:
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
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

        super(FP32Optimizer, self).__init__(                                   # trace_info : t_15672, t_15674, t_15725, t_15727
            optimizer, config, init_state_fn,                                  # trace_info : t_15673, t_15726
        )

        self._scale = torch.tensor([1.0], dtype=torch.float, device='cuda')    # trace_info : t_15679, t_15732

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:                              # trace_info : t_17750, t_17769, t_17800, t_17803, t_17814, ...
            _zero_grad_group_helper(group['params'], set_to_none)              # trace_info : t_17751, t_17770, t_17804, t_17815, t_22044, ...

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale                                                     # trace_info : t_20073, t_20127, t_21765, t_24418, t_24472, ...

    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""

        timers = self.config.timers                                            # trace_info : t_20603, t_21274, t_24948, t_25619, t_29293, ...

        # Copy main_grads to grads.
        if timers is not None:                                                 # trace_info : t_20604, t_21275, t_24949, t_25620, t_29294, ...
            timers('optimizer-copy-to-main-grad', log_level=1).start(          # trace_info : t_20605, t_20612, t_21276, t_21283, t_24950, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20611, t_21282, t_24956, t_25627, t_29301, ...
            )
        for param_group in self.optimizer.param_groups:                        # trace_info : t_20614, t_20632, t_20662, t_21285, t_21295, ...
            for param in param_group['params']:                                # trace_info : t_20615, t_20617, t_20619, t_20621, t_20623, ...
                param.grad = param.main_grad                                   # trace_info : t_20616, t_20618, t_20620, t_20622, t_20624, ...
        if timers is not None:                                                 # trace_info : t_20663, t_21306, t_25008, t_25651, t_29353, ...
            timers('optimizer-copy-to-main-grad').stop()                       # trace_info : t_20664, t_21307, t_25009, t_25652, t_29354, ...

        # Clip gradients.
        if timers is not None:                                                 # trace_info : t_20672, t_21315, t_25017, t_25660, t_29362, ...
            timers('optimizer-clip-main-grad', log_level=1).start(             # trace_info : t_20673, t_20680, t_21316, t_21323, t_25018, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_20679, t_21322, t_25024, t_25667, t_29369, ...
            )
        grad_norm = None                                                       # trace_info : t_20682, t_21325, t_25027, t_25670, t_29372, ...
        if self.config.clip_grad > 0.0:                                        # trace_info : t_20683, t_21326, t_25028, t_25671, t_29373, ...
            grad_norm = self.clip_grad_norm(self.config.clip_grad)             # trace_info : t_20684, t_21327, t_25029, t_25672, t_29374, ...
        if timers is not None:                                                 # trace_info : t_21219, t_21533, t_25564, t_25878, t_29909, ...
            timers('optimizer-clip-main-grad').stop()                          # trace_info : t_21220, t_21534, t_25565, t_25879, t_29910, ...

        # Count the zeros in the grads.
        if timers is not None:                                                 # trace_info : t_21228, t_21542, t_25573, t_25887, t_29918, ...
            timers('optimizer-count-zeros', log_level=1).start(                # trace_info : t_21229, t_21236, t_21543, t_21550, t_25574, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_21235, t_21549, t_25580, t_25894, t_29925, ...
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None# trace_info : t_21238, t_21552, t_25583, t_25897, t_29928, ...
        if timers is not None:                                                 # trace_info : t_21239, t_21553, t_25584, t_25898, t_29929, ...
            timers('optimizer-count-zeros').stop()                             # trace_info : t_21240, t_21554, t_25585, t_25899, t_29930, ...

        # Update parameters.
        if timers is not None:                                                 # trace_info : t_21248, t_21562, t_25593, t_25907, t_29938, ...
            timers('optimizer-inner-step', log_level=1).start(                 # trace_info : t_21249, t_21256, t_21563, t_21570, t_25594, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_21255, t_21569, t_25600, t_25914, t_29945, ...
            )
        self.optimizer.step()                                                  # trace_info : t_21258, t_21572, t_25603, t_25917, t_29948, ...
        if timers is not None:                                                 # trace_info : t_21259, t_21573, t_25604, t_25918, t_29949, ...
            timers('optimizer-inner-step').stop()                              # trace_info : t_21260, t_21574, t_25605, t_25919, t_29950, ...

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad                              # trace_info : t_21268, t_21582, t_25613, t_25927, t_29958, ...

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
        self.chained_optimizers = chained_optimizers                           # trace_info : t_15736

    @property
    def param_groups(self) -> List[dict]:
        param_groups = []                                                      # trace_info : t_15793, t_21614, t_21772, t_25959, t_26117, ...
        for optimizer in self.chained_optimizers:                              # trace_info : t_15794, t_15797, t_15800, t_21615, t_21618, ...
            param_groups += optimizer.param_groups                             # trace_info : t_15795, t_15798, t_21616, t_21619, t_21774, ...
        return param_groups                                                    # trace_info : t_15801, t_21622, t_21780, t_25967, t_26125, ...

    @property
    def state(self) -> ProxyDict:
        """
        Return optimizer state with tuple keys, where the first element is the
        index of the optimizer in the list of chained optimizers.
        """
        return ProxyDict([opt.state for opt in self.chained_optimizers])

    def zero_grad(self, set_to_none=True):
        for optimizer in self.chained_optimizers:                              # trace_info : t_17748, t_17801, t_17826, t_22041, t_22138, ...
            optimizer.zero_grad(set_to_none)                                   # trace_info : t_17749, t_17802, t_22042, t_22139, t_26387, ...

    def get_loss_scale(self):
        return self.chained_optimizers[0].get_loss_scale()                     # trace_info : t_20072, t_20126, t_21764, t_24417, t_24471, ...

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

        update_successful, grad_norm, num_zeros_in_grad = True, 0, 0           # trace_info : t_20599, t_24944, t_29289
        grad_norms = []                                                        # trace_info : t_20600, t_24945, t_29290
        for optimizer in self.chained_optimizers:                              # trace_info : t_20601, t_21272, t_21586, t_24946, t_25617, ...
            _update_successful, _grad_norm, _num_zeros_in_grad = optimizer.step()# trace_info : t_20602, t_21273, t_24947, t_25618, t_29292, ...
            update_successful &= _update_successful                            # trace_info : t_21269, t_21583, t_25614, t_25928, t_29959, ...
            grad_norms += [_grad_norm if _grad_norm else 0.0]                  # trace_info : t_21270, t_21584, t_25615, t_25929, t_29960, ...
            num_zeros_in_grad += _num_zeros_in_grad if _num_zeros_in_grad else 0# trace_info : t_21271, t_21585, t_25616, t_25930, t_29961, ...
        grad_norm = math.sqrt(sum([x ** 2 for x in grad_norms]))               # trace_info : t_21587, t_25932, t_30277

        return update_successful, grad_norm, num_zeros_in_grad                 # trace_info : t_21588, t_25933, t_30278

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
