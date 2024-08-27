# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron grad scaler."""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class MegatronGradScaler(ABC):
    def __init__(self, initial_scale: float):
        """Initialize scale value with the input initial scale."""
        assert initial_scale > 0.0                                             # trace_info : t_15107
        self._scale = torch.tensor([initial_scale], dtype=torch.float, device='cuda')# trace_info : t_15108

    @property
    def scale(self):
        return self._scale                                                     # trace_info : t_19675, t_21153, t_23312, t_24790, t_90919, ...

    @property
    def inv_scale(self):
        return self._scale.double().reciprocal().float()                       # trace_info : t_20117, t_23754, t_91361

    @abstractmethod
    def update(self, found_inf: bool):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict):
        pass


class ConstantGradScaler(MegatronGradScaler):
    """
    Constant grad scaler (loss scale is never adjusted regardless of NaNs seen in gradients).
    """

    def update(self, found_inf: bool):
        pass

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass


class DynamicGradScaler(MegatronGradScaler):
    """
    Grad scaler with dynamic scale that gets adjusted during training.

    Reduces loss scale by `backoff_factor` if `hysteresis` number of NaNs are seen in a row. Increases
    loss scale by `growth_factor` if NaNs are not seen for `growth_interval` iterations.
    """

    def __init__(
        self,
        initial_scale: float,
        min_scale: float,
        growth_factor: float,
        backoff_factor: float,
        growth_interval: int,
        hysteresis: int,
    ):
        """
        Grad scaler with dynamic scale that gets adjusted during training.

        Args:
            initial_scale (float): Initial loss scale value.
            min_scale (float): Minimum loss scale value.
            growth_factor (float): Factor to grow loss scale by if NaNs are not seen in `growth_interval`
                training iterations. Must be greater than 1.
            backoff_factor (float): Factor to decrease loss scale by if NaNs are seen in `hysteresis`
                consecutive training iterations. Must be between 0 and 1.
            growth_interval (int): Number of training iterations of no NaNs before loss scale is increased.
            hysteresis (int): Number of training iterations of consecutive NaNs before loss scale is decreased.
        """
        super(DynamicGradScaler, self).__init__(initial_scale)                 # trace_info : t_15106

        # Lower bound on the scale.
        assert min_scale > 0.0                                                 # trace_info : t_15109
        assert min_scale <= initial_scale                                      # trace_info : t_15110
        self.min_scale = torch.tensor([min_scale], dtype=torch.float, device='cuda')# trace_info : t_15111
        # Growth and backoff factors for the scale.
        assert growth_factor > 1.0                                             # trace_info : t_15112
        self.growth_factor = torch.tensor([growth_factor], dtype=torch.float, device='cuda')# trace_info : t_15113
        assert backoff_factor < 1.0                                            # trace_info : t_15114
        assert backoff_factor > 0.0                                            # trace_info : t_15115
        self.backoff_factor = torch.tensor([backoff_factor], dtype=torch.float, device='cuda')# trace_info : t_15116
        # Interval over which if we don't see any inf/nan,
        # we will scale the grad scale by the growth factor.
        assert growth_interval > 0                                             # trace_info : t_15117
        self.growth_interval = growth_interval                                 # trace_info : t_15118
        # Number of inf/nans we should see before scaling down
        # the grad scale by the backoff factor.
        assert hysteresis > 0                                                  # trace_info : t_15119
        self.hysteresis = hysteresis                                           # trace_info : t_15120

        # Trackers.
        self._growth_tracker = 0                                               # trace_info : t_15121
        self._hysteresis_tracker = self.hysteresis                             # trace_info : t_15122

    def update(self, found_inf: bool):
        """
        Updates internal state in grad scaler based on whether NaNs are seen in grads or not.
        """

        # If we have an inf/nan, growth tracker is set to 0
        # and hysterisis tracker is reduced by 1.
        if found_inf:                                                          # trace_info : t_20137, t_23774, t_91381
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            # Now if we are out of hysteresis count, scale down the loss.
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(self._scale * self.backoff_factor, self.min_scale)
        else:
            # If there is no nan/inf, increment the growth tracker.
            self._growth_tracker += 1                                          # trace_info : t_20138, t_23775, t_91382
            # If we have had enough consequitive intervals with no nan/inf:
            if self._growth_tracker == self.growth_interval:                   # trace_info : t_20139, t_23776, t_91383
                # Reset the tracker and hysteresis trackers,
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                # and scale up the loss scale.
                self._scale = self._scale * self.growth_factor

    def state_dict(self):
        state_dict = {}                                                        # trace_info : t_29112, t_96705
        state_dict['scale'] = self._scale                                      # trace_info : t_29113, t_96706
        state_dict['growth_tracker'] = self._growth_tracker                    # trace_info : t_29114, t_96707
        state_dict['hysteresis_tracker'] = self._hysteresis_tracker            # trace_info : t_29115, t_96708
        return state_dict                                                      # trace_info : t_29116, t_96709

    def load_state_dict(self, state_dict: Dict):
        self._scale = state_dict['scale'].cuda(torch.cuda.current_device())
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']
