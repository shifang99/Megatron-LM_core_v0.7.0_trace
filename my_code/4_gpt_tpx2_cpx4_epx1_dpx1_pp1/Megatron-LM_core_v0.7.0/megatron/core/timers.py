# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

import time
from abc import ABC, abstractmethod
from typing import List

import torch


class TimerBase(ABC):
    def __init__(self, name):
        self.name = name                                                       # trace_info : t_4310, t_9074, t_15807, t_17350

    @abstractmethod
    def start(self, barrier=False):
        pass

    @abstractmethod
    def stop(self, barrier=False):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def elapsed(self, reset=True, barrier=False):
        pass


class DummyTimer(TimerBase):
    def __init__(self):
        super().__init__('dummy timer')                                        # trace_info : t_4309

    def start(self, barrier=False):
        return                                                                 # trace_info : t_17697, t_17747, t_17925, t_18928, t_19007, ...

    def stop(self, barrier=False):
        return                                                                 # trace_info : t_18144, t_18888, t_18985, t_19039, t_19069, ...

    def reset(self):
        return

    def elapsed(self, reset=True, barrier=False):
        raise Exception('dummy timer should not be used to calculate elapsed time')


class Timer(TimerBase):
    """
    Timer class with ability to start/stop.

    Comment on using `barrier`: If this flag is passed, then all
    the caller processes will wait till all reach the timing routine.
    It is up to the user to make sure all the ranks in `barrier_group`
    call it otherwise, it will result in a hang.
    Comment on `barrier_group`: By default it is set to None which
    in torch distributed land, it will result in the global communicator.
    """

    def __init__(self, name):
        """Initialize Timer.

        Args:
            name (str): Name of the timer.
        """
        super().__init__(name)                                                 # trace_info : t_9073, t_15806, t_17349
        self._elapsed = 0.0                                                    # trace_info : t_9075, t_15808, t_17351
        self._active_time = 0.0                                                # trace_info : t_9076, t_15809, t_17352
        self._started = False                                                  # trace_info : t_9077, t_15810, t_17353
        # Note that None will default to the global process group
        self._barrier_group = None                                             # trace_info : t_9078, t_15811, t_17354
        self._start_time = time.time()                                         # trace_info : t_9079, t_15812, t_17355

    def set_barrier_group(self, barrier_group):
        """Sets barrier group.

        Args:
            barrier_group (ProcessGroup): Torch ProcessGroup for barrier.
        """
        self._barrier_group = barrier_group

    def start(self, barrier=False):
        """Start the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before starting. Defaults to False.
        """
        assert not self._started, 'timer has already been started'             # trace_info : t_9082, t_15817, t_17358
        if barrier:                                                            # trace_info : t_9083, t_15818, t_17359
            torch.distributed.barrier(group=self._barrier_group)               # trace_info : t_9084, t_15819, t_17360
        torch.cuda.synchronize()                                               # trace_info : t_9085, t_15820, t_17361
        self._start_time = time.time()                                         # trace_info : t_9086, t_15821, t_17362
        self._started = True                                                   # trace_info : t_9087, t_15822, t_17363

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, 'timer is not started'                           # trace_info : t_15777, t_17190
        if barrier:                                                            # trace_info : t_15778, t_17191
            torch.distributed.barrier(group=self._barrier_group)
        torch.cuda.synchronize()                                               # trace_info : t_15779, t_17192
        elapsed = time.time() - self._start_time                               # trace_info : t_15780, t_17193
        self._elapsed += elapsed                                               # trace_info : t_15781, t_17194
        self._active_time += elapsed                                           # trace_info : t_15782, t_17195
        self._started = False                                                  # trace_info : t_15783, t_17196

    def reset(self):
        """Reset timer.
        """
        # Don't reset _active_time
        self._elapsed = 0.0                                                    # trace_info : t_17240, t_17252
        self._started = False                                                  # trace_info : t_17241, t_17253

    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started                                               # trace_info : t_17235, t_17247
        # If the timing in progress, end it first.
        if self._started:                                                      # trace_info : t_17236, t_17248
            self.stop(barrier=barrier)
        # Get the elapsed time.
        _elapsed = self._elapsed                                               # trace_info : t_17237, t_17249
        # Reset the elapsed time
        if reset:                                                              # trace_info : t_17238, t_17250
            self.reset()                                                       # trace_info : t_17239, t_17251
        # If timing was in progress, set it back.
        if _started:                                                           # trace_info : t_17242, t_17254
            self.start(barrier=barrier)
        return _elapsed                                                        # trace_info : t_17243, t_17255

    def active_time(self):
        return self._active_time


class Timers:
    """Class for a group of Timers.
    """

    def __init__(self, log_level, log_option):
        """Initialize group of timers.

        Args:
            log_level (int): Log level to control what timers are enabled.            
            log_option (str): Setting for logging statistics over ranks for all the timers. Allowed: ['max', 'minmax', 'all'].
        """
        self._log_level = log_level                                            # trace_info : t_4302
        allowed_log_options = set(['max', 'minmax', 'all'])                    # trace_info : t_4303
        assert (
            log_option in allowed_log_options                                  # trace_info : t_4304
        ), 'input log option {} is invalid. It must be one of {}'.format(
            log_option, allowed_log_options
        )
        self._log_option = log_option                                          # trace_info : t_4305
        self._timers = {}                                                      # trace_info : t_4306
        self._log_levels = {}                                                  # trace_info : t_4307
        self._dummy_timer = DummyTimer()                                       # trace_info : t_4308
        self._max_log_level = 2                                                # trace_info : t_4311

    def __call__(self, name, log_level=None):
        """Call timer with name and log level."""
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:                                               # trace_info : t_9068, t_15774, t_15801, t_17187, t_17344, ...
            if log_level is not None:                                          # trace_info : t_15775, t_17188
                assert log_level == self._log_levels[name], (
                    'input log level {} does not match already existing '
                    'log level {} for {} timer'.format(log_level, self._log_levels[name], name)
                )
            return self._timers[name]                                          # trace_info : t_15776, t_17189
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:                                                  # trace_info : t_9069, t_15802, t_17345, t_17693, t_17743, ...
            log_level = self._max_log_level                                    # trace_info : t_18140, t_18884, t_18981, t_19035, t_19065, ...
        assert (
            log_level <= self._max_log_level                                   # trace_info : t_9070, t_15803, t_17346, t_17694, t_17744, ...
        ), 'log level {} is larger than max supported log level {}'.format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:                                        # trace_info : t_9071, t_15804, t_17347, t_17695, t_17745, ...
            return self._dummy_timer                                           # trace_info : t_17696, t_17746, t_17924, t_18143, t_18887, ...
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)                                       # trace_info : t_9072, t_15805, t_17348
        self._log_levels[name] = log_level                                     # trace_info : t_9080, t_15813, t_17356
        return self._timers[name]                                              # trace_info : t_9081, t_15814, t_17357

    def _get_elapsed_time_all_ranks(self, names, reset, barrier):
        """Returns elapsed times of timers in names.
        Assumptions:
            - All the ranks call this function.
            - `names` are identical on all ranks.
        If the above assumptions are not met, calling this function will
        result in hang.

        Args:
            names (List[str]): list of timer names
            reset (bool): reset the timer after recording the elapsed time
            barrier (bool): if set, do a global barrier before time measurments

        Returns:
            torch.tensor: Tensor of size [world_size, len(names)] with times in float.
        """

        # First make sure all the callers are in sync.
        if barrier:                                                            # trace_info : t_17225
            torch.distributed.barrier()                                        # trace_info : t_17226

        world_size = torch.distributed.get_world_size()                        # trace_info : t_17227
        rank = torch.distributed.get_rank()                                    # trace_info : t_17228

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros(                                       # trace_info : t_17229, t_17231
            (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()# trace_info : t_17230
        )
        for i, name in enumerate(names):                                       # trace_info : t_17232, t_17244, t_17256
            if name in self._timers:                                           # trace_info : t_17233, t_17245
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)# trace_info : t_17234, t_17246

        # See the note above for why we are not using gather.
        torch.distributed._all_gather_base(                                    # trace_info : t_17257, t_17259
            rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1)    # trace_info : t_17258
        )

        return rank_name_to_time                                               # trace_info : t_17260

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)# trace_info : t_17224
        name_to_min_max_time = {}                                              # trace_info : t_17261
        for i, name in enumerate(names):                                       # trace_info : t_17262, t_17269, t_17276
            rank_to_time = rank_name_to_time[:, i]                             # trace_info : t_17263, t_17270
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]                    # trace_info : t_17264, t_17271
            # If the timer exists:
            if rank_to_time.numel() > 0:                                       # trace_info : t_17265, t_17272
                name_to_min_max_time[name] = (                                 # trace_info : t_17268, t_17275
                    rank_to_time.min().item() / normalizer,                    # trace_info : t_17266, t_17273
                    rank_to_time.max().item() / normalizer,                    # trace_info : t_17267, t_17274
                )
        return name_to_min_max_time                                            # trace_info : t_17277

    def _get_global_min_max_time_string(self, names, reset, barrier, normalizer, max_only):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)# trace_info : t_17223
        if not name_to_min_max_time:                                           # trace_info : t_17278
            return None
        if max_only:                                                           # trace_info : t_17279
            output_string = 'max time across ranks (ms):'
        else:
            output_string = '(min, max) time across ranks (ms):'               # trace_info : t_17280
        for name in name_to_min_max_time:                                      # trace_info : t_17281, t_17287, t_17293
            min_time, max_time = name_to_min_max_time[name]                    # trace_info : t_17282, t_17288
            if max_only:                                                       # trace_info : t_17283, t_17289
                output_string += '\n    {}: {:.2f}'.format((name + ' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(          # trace_info : t_17284, t_17286, t_17290, t_17292
                    (name + ' ').ljust(48, '.'), min_time, max_time            # trace_info : t_17285, t_17291
                )
        return output_string                                                   # trace_info : t_17294

    def _get_all_ranks_time_string(self, names, reset, barrier, normalizer):
        """Report times across all ranks."""
        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)

        output_string = 'times across ranks (ms):'
        no_reported_timing = True
        for i, name in enumerate(names):
            not_yet_found = True
            for rank in range(torch.distributed.get_world_size()):
                if rank_name_to_time[rank, i] > 0:
                    no_reported_timing = False
                    if not_yet_found:
                        not_yet_found = False
                        output_string += '\n  {}:'.format(name)
                    output_string += '\n     rank {:2d}: {:.2f}'.format(
                        rank, rank_name_to_time[rank, i] / normalizer
                    )
        if no_reported_timing:
            return None
        return output_string

    def get_all_timers_string(
        self,
        names: List[str] = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Returns the output string with logged timer values according to configured options.

        Args:
            names (List[str]): Names of the timers to log. If None, all registered timers are fetched. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.

        Raises:
            Exception: Raises if log option is invalid.

        Returns:
            str: Formatted string with the timer values.
        """

        if names == None:  # get all registered timers                         # trace_info : t_17215
            names = self._timers.keys()

        assert normalizer > 0.0                                                # trace_info : t_17216
        if self._log_option in ['max', 'minmax']:                              # trace_info : t_17217
            max_only = False                                                   # trace_info : t_17218
            if self._log_option == 'max':                                      # trace_info : t_17219
                max_only = True
            output_string = self._get_global_min_max_time_string(              # trace_info : t_17220, t_17222
                names, reset, barrier, normalizer / 1000.0, max_only           # trace_info : t_17221
            )
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception('unknown timing log option {}'.format(self._log_option))
        return output_string                                                   # trace_info : t_17295

    def log(
        self,
        names: List[str],
        rank: int = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """logs the timers passed in names to stdout. Example usage is to log average per step value for timer 'foo',
          this function can be called with normalizer factor set to logging interval. 

        Args:
            names (List[str]): Names of the timers to log.
            rank (int, optional): logs the timers to a specific rank. If set to None, logs to the last rank. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.
        """

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)# trace_info : t_17214
        # If no input rank is provided, log on last rank.
        if rank is None:                                                       # trace_info : t_17296
            rank = torch.distributed.get_world_size() - 1                      # trace_info : t_17297
        if rank == torch.distributed.get_rank() and output_string is not None: # trace_info : t_17298
            print(output_string, flush=True)

    def write(
        self,
        names: List[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Write timers to a tensorboard writer. Note that we only report maximum time across ranks to tensorboard.

        Args:
            names (List[str]): Names of the timers to log.
            writer (SummaryWriter): Tensorboard SummaryWriter object
            iteration (int): Current iteration.
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                writer.add_scalar(name + '-time', max_time, iteration)
