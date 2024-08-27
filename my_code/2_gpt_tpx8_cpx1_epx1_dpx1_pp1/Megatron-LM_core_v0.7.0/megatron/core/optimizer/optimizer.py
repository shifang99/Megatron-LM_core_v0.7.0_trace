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
    for param in group:                                                        # trace_info : t_19533, t_19535, t_19537, t_19539, t_19541, ...
        if param.grad is not None:                                             # trace_info : t_19534, t_19536, t_19538, t_19540, t_19542, ...
            if set_to_none:                                                    # trace_info : t_23154, t_23158, t_23162, t_23166, t_23170, ...
                param.grad = None                                              # trace_info : t_23155, t_23159, t_23163, t_23167, t_23171, ...
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
    if overflow_buf:                                                           # trace_info : t_22650, t_26260, t_29870
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):                                   # trace_info : t_22651, t_22653, t_22655, t_22657, t_22659, ...
            that_.copy_(this_)                                                 # trace_info : t_22652, t_22654, t_22656, t_22658, t_22660, ...


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
        self.optimizer = optimizer                                             # trace_info : t_16974
        assert self.optimizer, 'no optimizer is provided.'                     # trace_info : t_16975
        self.config = config                                                   # trace_info : t_16976
        self.init_state_fn = init_state_fn                                     # trace_info : t_16977

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []                                                            # trace_info : t_21859, t_21924, t_25469, t_25534, t_29079, ...
        for param_group in self.optimizer.param_groups:                        # trace_info : t_21860, t_21882, t_21920, t_21925, t_21947, ...
            for param in param_group['params']:                                # trace_info : t_21861, t_21863, t_21865, t_21867, t_21869, ...
                params.append(param)                                           # trace_info : t_21862, t_21864, t_21866, t_21868, t_21870, ...
        return params                                                          # trace_info : t_21921, t_21986, t_25531, t_25596, t_29141, ...

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()                                         # trace_info : t_21923, t_25533, t_29143
        grads_for_norm = []                                                    # trace_info : t_21987, t_25597, t_29207
        for param in params:                                                   # trace_info : t_21988, t_21997, t_22013, t_22022, t_22031, ...
            grad = param.grad                                                  # trace_info : t_21989, t_21998, t_22014, t_22023, t_22032, ...
            grad_not_none = grad is not None                                   # trace_info : t_21990, t_21999, t_22015, t_22024, t_22033, ...
            is_not_shared = param_is_not_shared(param)                         # trace_info : t_21991, t_22000, t_22016, t_22025, t_22034, ...
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)# trace_info : t_21993, t_22002, t_22018, t_22027, t_22036, ...
            if grad_not_none and is_not_shared and is_not_tp_duplicate:        # trace_info : t_21995, t_22011, t_22020, t_22029, t_22038, ...
                grads_for_norm.append(grad)                                    # trace_info : t_21996, t_22012, t_22021, t_22030, t_22039, ...

        return grads_for_norm                                                  # trace_info : t_22346, t_25956, t_29566

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Default returned here, but the distributed optimizer overrides this."""
        return parallel_state.get_model_parallel_group()                       # trace_info : t_21825, t_22349, t_25435, t_25959, t_29045, ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm."""
        params = self.get_parameters()                                         # trace_info : t_21858, t_25468, t_29078
        grads_for_norm = self.get_main_grads_for_grad_norm()                   # trace_info : t_21922, t_25532, t_29142
        return clip_grad_norm_fp32(                                            # trace_info : t_22347, t_22352, t_25957, t_25962, t_29567, ...
            params, grads_for_norm, clip_grad, model_parallel_group=self.get_model_parallel_group(),# trace_info : t_22348, t_25958, t_29568
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
        return self.get_loss_scale() * loss                                    # trace_info : t_21377, t_24987, t_28597

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
        return self.optimizer.param_groups                                     # trace_info : t_17738, t_22743, t_22864, t_26353, t_26474, ...

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

        super().__init__(                                                      # trace_info : t_16971, t_16973
            optimizer, config, init_state_fn,                                  # trace_info : t_16972
        )
        self.grad_scaler = grad_scaler                                         # trace_info : t_16978

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:                                           # trace_info : t_16979
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:                                                   # trace_info : t_16980
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')# trace_info : t_16981

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:                                                   # trace_info : t_16982
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')# trace_info : t_16983

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:                                           # trace_info : t_16984
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:                                           # trace_info : t_21378, t_22855, t_24988, t_26465, t_28598, ...
            return self._scale_one
        return self.grad_scaler.scale                                          # trace_info : t_21379, t_22856, t_24989, t_26466, t_28599, ...

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()              # trace_info : t_21721, t_25331, t_28941

        # Reset found inf.
        self.found_inf.fill_(0.0)                                              # trace_info : t_21818, t_25428, t_29038

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(                      # trace_info : t_21819, t_21822, t_25429, t_25432, t_29039, ...
            main_grads, self.found_inf, self.grad_scaler.inv_scale             # trace_info : t_21820, t_25430, t_29040
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(                                          # trace_info : t_21823, t_21828, t_25433, t_25438, t_29043, ...
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()# trace_info : t_21824, t_25434, t_29044
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0                             # trace_info : t_21829, t_25439, t_29049

        return found_inf_flag                                                  # trace_info : t_21830, t_25440, t_29050

    @torch.no_grad()
    def step(self):

        timers = self.config.timers                                            # trace_info : t_21566, t_25176, t_28786

        # Copy gradients from model params to main params.
        if timers is not None:                                                 # trace_info : t_21567, t_25177, t_28787
            timers('optimizer-copy-to-main-grad', log_level=1).start(          # trace_info : t_21568, t_21575, t_25178, t_25185, t_28788, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_21574, t_25184, t_28794
            )
        self._copy_model_grads_to_main_grads()                                 # trace_info : t_21577, t_25187, t_28797
        if timers is not None:                                                 # trace_info : t_21700, t_25310, t_28920
            timers('optimizer-copy-to-main-grad').stop()                       # trace_info : t_21701, t_25311, t_28921

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:                                                   # trace_info : t_21709, t_25319, t_28929

            # Unscale and check for inf/nan.
            if timers is not None:                                             # trace_info : t_21710, t_25320, t_28930
                timers('optimizer-unscale-and-check-inf', log_level=1).start(  # trace_info : t_21711, t_21718, t_25321, t_25328, t_28931, ...
                    barrier=self.config.barrier_with_L1_time                   # trace_info : t_21717, t_25327, t_28937
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()      # trace_info : t_21720, t_25330, t_28940
            if timers is not None:                                             # trace_info : t_21831, t_25441, t_29051
                timers('optimizer-unscale-and-check-inf').stop()               # trace_info : t_21832, t_25442, t_29052

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)                            # trace_info : t_21840, t_25450, t_29060

            # If we found inf/nan, skip the update.
            if found_inf_flag:                                                 # trace_info : t_21844, t_25454, t_29064
                return False, None, None

        # Clip the main gradients.
        if timers is not None:                                                 # trace_info : t_21845, t_25455, t_29065
            timers('optimizer-clip-main-grad', log_level=1).start(             # trace_info : t_21846, t_21853, t_25456, t_25463, t_29066, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_21852, t_25462, t_29072
            )
        grad_norm = None                                                       # trace_info : t_21855, t_25465, t_29075
        if self.config.clip_grad > 0.0:                                        # trace_info : t_21856, t_25466, t_29076
            grad_norm = self.clip_grad_norm(self.config.clip_grad)             # trace_info : t_21857, t_25467, t_29077
        if timers is not None:                                                 # trace_info : t_22494, t_26104, t_29714
            timers('optimizer-clip-main-grad').stop()                          # trace_info : t_22495, t_26105, t_29715

        # Count the zeros in the grads.
        if timers is not None:                                                 # trace_info : t_22503, t_26113, t_29723
            timers('optimizer-count-zeros', log_level=1).start(                # trace_info : t_22504, t_22511, t_26114, t_26121, t_29724, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_22510, t_26120, t_29730
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None# trace_info : t_22513, t_26123, t_29733
        if timers is not None:                                                 # trace_info : t_22514, t_26124, t_29734
            timers('optimizer-count-zeros').stop()                             # trace_info : t_22515, t_26125, t_29735

        # Step the optimizer.
        if timers is not None:                                                 # trace_info : t_22523, t_26133, t_29743
            timers('optimizer-inner-step', log_level=1).start(                 # trace_info : t_22524, t_22531, t_26134, t_26141, t_29744, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_22530, t_26140, t_29750
            )
        self.optimizer.step()                                                  # trace_info : t_22533, t_26143, t_29753
        if timers is not None:                                                 # trace_info : t_22534, t_26144, t_29754
            timers('optimizer-inner-step').stop()                              # trace_info : t_22535, t_26145, t_29755

        # Update params from main params.
        if timers is not None:                                                 # trace_info : t_22543, t_26153, t_29763
            timers('optimizer-copy-main-to-model-params', log_level=1).start(  # trace_info : t_22544, t_22551, t_26154, t_26161, t_29764, ...
                barrier=self.config.barrier_with_L1_time                       # trace_info : t_22550, t_26160, t_29770
            )
        self._copy_main_params_to_model_params()                               # trace_info : t_22553, t_26163, t_29773
        if timers is not None:                                                 # trace_info : t_22708, t_26318, t_29928
            timers('optimizer-copy-main-to-model-params').stop()               # trace_info : t_22709, t_26319, t_29929

        # Successful update.
        return True, grad_norm, num_zeros_in_grad                              # trace_info : t_22717, t_26327, t_29937


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

        super().__init__(                                                      # trace_info : t_16968, t_16970
            optimizer, config, grad_scaler, init_state_fn,                     # trace_info : t_16969
        )

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []                                               # trace_info : t_16985
        self.fp32_from_float16_groups = []                                     # trace_info : t_16986
        self.fp32_from_fp32_groups = []                                        # trace_info : t_16987

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:                        # trace_info : t_16988, t_17236, t_17676
            float16_params_this_group = []                                     # trace_info : t_16989, t_17237
            fp32_params_this_group = []                                        # trace_info : t_16990, t_17238
            fp32_from_float16_params_this_group = []                           # trace_info : t_16991, t_17239
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):                  # trace_info : t_16992, t_17016, t_17040, t_17064, t_17088, ...
                if param.requires_grad:                                        # trace_info : t_16993, t_17017, t_17041, t_17065, t_17089, ...

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:# trace_info : t_16994, t_17018, t_17042, t_17066, t_17090, ...
                        float16_params_this_group.append(param)                # trace_info : t_16995, t_17019, t_17043, t_17067, t_17091, ...
                        # Create a copy
                        main_param = param.detach().clone().float()            # trace_info : t_16996, t_17020, t_17044, t_17068, t_17092, ...
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)# trace_info : t_16997, t_17021, t_17045, t_17069, t_17093, ...
                        if hasattr(param, 'shared'):                           # trace_info : t_17012, t_17036, t_17060, t_17084, t_17108, ...
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param                  # trace_info : t_17013, t_17037, t_17061, t_17085, t_17109, ...

                        fp32_from_float16_params_this_group.append(main_param) # trace_info : t_17014, t_17038, t_17062, t_17086, t_17110, ...
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:                      # trace_info : t_17015, t_17039, t_17063, t_17087, t_17111, ...
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

            self.float16_groups.append(float16_params_this_group)              # trace_info : t_17233, t_17673
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)# trace_info : t_17234, t_17674
            self.fp32_from_fp32_groups.append(fp32_params_this_group)          # trace_info : t_17235, t_17675

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:                                      # trace_info : t_19531, t_19554, t_19593, t_23087, t_23110, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_19532, t_19555, t_23088, t_23111, t_26698, ...
        for group in self.fp32_from_float16_groups:                            # trace_info : t_19594, t_19617, t_19656, t_23150, t_23193, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_19595, t_19618, t_23151, t_23194, t_26761, ...
        for group in self.fp32_from_fp32_groups:                               # trace_info : t_19657, t_19660, t_19663, t_23269, t_23272, ...
            _zero_grad_group_helper(group, set_to_none)                        # trace_info : t_19658, t_19661, t_23270, t_23273, t_26880, ...

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []                                                        # trace_info : t_21722, t_25332, t_28942

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:                       # trace_info : t_21723, t_21755, t_21811, t_25333, t_25365, ...
            for main_param in main_group:                                      # trace_info : t_21724, t_21727, t_21730, t_21733, t_21736, ...
                if main_param.grad is not None:                                # trace_info : t_21725, t_21728, t_21731, t_21734, t_21737, ...
                    main_grads.append(main_param.grad.data)                    # trace_info : t_21726, t_21729, t_21732, t_21735, t_21738, ...

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:                          # trace_info : t_21812, t_21814, t_21816, t_25422, t_25424, ...
            for main_param in main_group:                                      # trace_info : t_21813, t_21815, t_25423, t_25425, t_29033, ...
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads                                                      # trace_info : t_21817, t_25427, t_29037

    def _get_model_and_main_params_data_float16(self):
        model_data = []                                                        # trace_info : t_22555, t_26165, t_29775
        main_data = []                                                         # trace_info : t_22556, t_26166, t_29776
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_22557, t_22589, t_22645, t_26167, t_26199, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_22558, t_22561, t_22564, t_22567, t_22570, ...
                model_data.append(model_param.data)                            # trace_info : t_22559, t_22562, t_22565, t_22568, t_22571, ...
                main_data.append(main_param.data)                              # trace_info : t_22560, t_22563, t_22566, t_22569, t_22572, ...
        return model_data, main_data                                           # trace_info : t_22646, t_26256, t_29866

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):# trace_info : t_21578, t_21620, t_21694, t_25188, t_25230, ...
            for model_param, main_param in zip(model_group, main_group):       # trace_info : t_21579, t_21583, t_21587, t_21591, t_21595, ...
                if hasattr(model_param, 'main_grad'):                          # trace_info : t_21580, t_21584, t_21588, t_21592, t_21596, ...
                    main_param.grad = model_param.main_grad.float()            # trace_info : t_21581, t_21585, t_21589, t_21593, t_21597, ...
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None                                        # trace_info : t_21582, t_21586, t_21590, t_21594, t_21598, ...

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:                         # trace_info : t_21695, t_21697, t_21699, t_25305, t_25307, ...
            for model_param in model_group:                                    # trace_info : t_21696, t_21698, t_25306, t_25308, t_28916, ...
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16() # trace_info : t_22554, t_26164, t_29774
        _multi_tensor_copy_this_to_that(                                       # trace_info : t_22647, t_22649, t_26257, t_26259, t_29867, ...
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf# trace_info : t_22648, t_26258, t_29868
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
