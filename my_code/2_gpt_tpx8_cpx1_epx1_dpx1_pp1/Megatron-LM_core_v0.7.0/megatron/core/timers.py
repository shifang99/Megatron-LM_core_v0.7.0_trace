# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

import time
from abc import ABC, abstractmethod
from typing import List

import torch


class TimerBase(ABC):
    def __init__(self, name):
        self.name = name                                                       # trace_info : t_5846, t_10631, t_17815, t_19358

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
        super().__init__('dummy timer')                                        # trace_info : t_5845

    def start(self, barrier=False):
        return                                                                 # trace_info : t_19705, t_19755, t_19913, t_21363, t_21442, ...

    def stop(self, barrier=False):
        return                                                                 # trace_info : t_20025, t_21323, t_21420, t_21473, t_21503, ...

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
        super().__init__(name)                                                 # trace_info : t_10630, t_17814, t_19357
        self._elapsed = 0.0                                                    # trace_info : t_10632, t_17816, t_19359
        self._active_time = 0.0                                                # trace_info : t_10633, t_17817, t_19360
        self._started = False                                                  # trace_info : t_10634, t_17818, t_19361
        # Note that None will default to the global process group
        self._barrier_group = None                                             # trace_info : t_10635, t_17819, t_19362
        self._start_time = time.time()                                         # trace_info : t_10636, t_17820, t_19363

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
        assert not self._started, 'timer has already been started'             # trace_info : t_10639, t_17825, t_19366
        if barrier:                                                            # trace_info : t_10640, t_17826, t_19367
            torch.distributed.barrier(group=self._barrier_group)               # trace_info : t_10641, t_17827, t_19368
        torch.cuda.synchronize()                                               # trace_info : t_10642, t_17828, t_19369
        self._start_time = time.time()                                         # trace_info : t_10643, t_17829, t_19370
        self._started = True                                                   # trace_info : t_10644, t_17830, t_19371

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, 'timer is not started'                           # trace_info : t_17785, t_19198
        if barrier:                                                            # trace_info : t_17786, t_19199
            torch.distributed.barrier(group=self._barrier_group)
        torch.cuda.synchronize()                                               # trace_info : t_17787, t_19200
        elapsed = time.time() - self._start_time                               # trace_info : t_17788, t_19201
        self._elapsed += elapsed                                               # trace_info : t_17789, t_19202
        self._active_time += elapsed                                           # trace_info : t_17790, t_19203
        self._started = False                                                  # trace_info : t_17791, t_19204

    def reset(self):
        """Reset timer.
        """
        # Don't reset _active_time
        self._elapsed = 0.0                                                    # trace_info : t_19248, t_19260
        self._started = False                                                  # trace_info : t_19249, t_19261

    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started                                               # trace_info : t_19243, t_19255
        # If the timing in progress, end it first.
        if self._started:                                                      # trace_info : t_19244, t_19256
            self.stop(barrier=barrier)
        # Get the elapsed time.
        _elapsed = self._elapsed                                               # trace_info : t_19245, t_19257
        # Reset the elapsed time
        if reset:                                                              # trace_info : t_19246, t_19258
            self.reset()                                                       # trace_info : t_19247, t_19259
        # If timing was in progress, set it back.
        if _started:                                                           # trace_info : t_19250, t_19262
            self.start(barrier=barrier)
        return _elapsed                                                        # trace_info : t_19251, t_19263

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
        self._log_level = log_level                                            # trace_info : t_5838
        allowed_log_options = set(['max', 'minmax', 'all'])                    # trace_info : t_5839
        assert (
            log_option in allowed_log_options                                  # trace_info : t_5840
        ), 'input log option {} is invalid. It must be one of {}'.format(
            log_option, allowed_log_options
        )
        self._log_option = log_option                                          # trace_info : t_5841
        self._timers = {}                                                      # trace_info : t_5842
        self._log_levels = {}                                                  # trace_info : t_5843
        self._dummy_timer = DummyTimer()                                       # trace_info : t_5844
        self._max_log_level = 2                                                # trace_info : t_5847

    def __call__(self, name, log_level=None):
        """Call timer with name and log level."""
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:                                               # trace_info : t_10625, t_17782, t_17809, t_19195, t_19352, ...
            if log_level is not None:                                          # trace_info : t_17783, t_19196
                assert log_level == self._log_levels[name], (
                    'input log level {} does not match already existing '
                    'log level {} for {} timer'.format(log_level, self._log_levels[name], name)
                )
            return self._timers[name]                                          # trace_info : t_17784, t_19197
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:                                                  # trace_info : t_10626, t_17810, t_19353, t_19701, t_19751, ...
            log_level = self._max_log_level                                    # trace_info : t_20021, t_21319, t_21416, t_21469, t_21499, ...
        assert (
            log_level <= self._max_log_level                                   # trace_info : t_10627, t_17811, t_19354, t_19702, t_19752, ...
        ), 'log level {} is larger than max supported log level {}'.format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:                                        # trace_info : t_10628, t_17812, t_19355, t_19703, t_19753, ...
            return self._dummy_timer                                           # trace_info : t_19704, t_19754, t_19912, t_20024, t_21322, ...
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)                                       # trace_info : t_10629, t_17813, t_19356
        self._log_levels[name] = log_level                                     # trace_info : t_10637, t_17821, t_19364
        return self._timers[name]                                              # trace_info : t_10638, t_17822, t_19365

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
        if barrier:                                                            # trace_info : t_19233
            torch.distributed.barrier()                                        # trace_info : t_19234

        world_size = torch.distributed.get_world_size()                        # trace_info : t_19235
        rank = torch.distributed.get_rank()                                    # trace_info : t_19236

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros(                                       # trace_info : t_19237, t_19239
            (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()# trace_info : t_19238
        )
        for i, name in enumerate(names):                                       # trace_info : t_19240, t_19252, t_19264
            if name in self._timers:                                           # trace_info : t_19241, t_19253
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)# trace_info : t_19242, t_19254

        # See the note above for why we are not using gather.
        torch.distributed._all_gather_base(                                    # trace_info : t_19265, t_19267
            rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1)    # trace_info : t_19266
        )

        return rank_name_to_time                                               # trace_info : t_19268

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)# trace_info : t_19232
        name_to_min_max_time = {}                                              # trace_info : t_19269
        for i, name in enumerate(names):                                       # trace_info : t_19270, t_19277, t_19284
            rank_to_time = rank_name_to_time[:, i]                             # trace_info : t_19271, t_19278
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]                    # trace_info : t_19272, t_19279
            # If the timer exists:
            if rank_to_time.numel() > 0:                                       # trace_info : t_19273, t_19280
                name_to_min_max_time[name] = (                                 # trace_info : t_19276, t_19283
                    rank_to_time.min().item() / normalizer,                    # trace_info : t_19274, t_19281
                    rank_to_time.max().item() / normalizer,                    # trace_info : t_19275, t_19282
                )
        return name_to_min_max_time                                            # trace_info : t_19285

    def _get_global_min_max_time_string(self, names, reset, barrier, normalizer, max_only):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)# trace_info : t_19231
        if not name_to_min_max_time:                                           # trace_info : t_19286
            return None
        if max_only:                                                           # trace_info : t_19287
            output_string = 'max time across ranks (ms):'
        else:
            output_string = '(min, max) time across ranks (ms):'               # trace_info : t_19288
        for name in name_to_min_max_time:                                      # trace_info : t_19289, t_19295, t_19301
            min_time, max_time = name_to_min_max_time[name]                    # trace_info : t_19290, t_19296
            if max_only:                                                       # trace_info : t_19291, t_19297
                output_string += '\n    {}: {:.2f}'.format((name + ' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(          # trace_info : t_19292, t_19294, t_19298, t_19300
                    (name + ' ').ljust(48, '.'), min_time, max_time            # trace_info : t_19293, t_19299
                )
        return output_string                                                   # trace_info : t_19302

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

        if names == None:  # get all registered timers                         # trace_info : t_19223
            names = self._timers.keys()

        assert normalizer > 0.0                                                # trace_info : t_19224
        if self._log_option in ['max', 'minmax']:                              # trace_info : t_19225
            max_only = False                                                   # trace_info : t_19226
            if self._log_option == 'max':                                      # trace_info : t_19227
                max_only = True
            output_string = self._get_global_min_max_time_string(              # trace_info : t_19228, t_19230
                names, reset, barrier, normalizer / 1000.0, max_only           # trace_info : t_19229
            )
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception('unknown timing log option {}'.format(self._log_option))
        return output_string                                                   # trace_info : t_19303

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

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)# trace_info : t_19222
        # If no input rank is provided, log on last rank.
        if rank is None:                                                       # trace_info : t_19304
            rank = torch.distributed.get_world_size() - 1                      # trace_info : t_19305
        if rank == torch.distributed.get_rank() and output_string is not None: # trace_info : t_19306
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
