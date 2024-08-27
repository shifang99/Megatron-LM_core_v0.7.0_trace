# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

import time
from abc import ABC, abstractmethod
from typing import List

import torch


class TimerBase(ABC):
    def __init__(self, name):
        self.name = name                                                       # trace_info : t_4055, t_5764, t_12948, t_14492

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
        super().__init__('dummy timer')                                        # trace_info : t_4054

    def start(self, barrier=False):
        return                                                                 # trace_info : t_14839, t_14889, t_15047, t_16530, t_16609, ...

    def stop(self, barrier=False):
        return                                                                 # trace_info : t_15159, t_16490, t_16587, t_16640, t_16666, ...

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
        super().__init__(name)                                                 # trace_info : t_5763, t_12947, t_14491
        self._elapsed = 0.0                                                    # trace_info : t_5765, t_12949, t_14493
        self._active_time = 0.0                                                # trace_info : t_5766, t_12950, t_14494
        self._started = False                                                  # trace_info : t_5767, t_12951, t_14495
        # Note that None will default to the global process group
        self._barrier_group = None                                             # trace_info : t_5768, t_12952, t_14496
        self._start_time = time.time()                                         # trace_info : t_5769, t_12953, t_14497

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
        assert not self._started, 'timer has already been started'             # trace_info : t_5772, t_12958, t_14500
        if barrier:                                                            # trace_info : t_5773, t_12959, t_14501
            torch.distributed.barrier(group=self._barrier_group)               # trace_info : t_5774, t_12960, t_14502
        torch.cuda.synchronize()                                               # trace_info : t_5775, t_12961, t_14503
        self._start_time = time.time()                                         # trace_info : t_5776, t_12962, t_14504
        self._started = True                                                   # trace_info : t_5777, t_12963, t_14505

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, 'timer is not started'                           # trace_info : t_12918, t_14331
        if barrier:                                                            # trace_info : t_12919, t_14332
            torch.distributed.barrier(group=self._barrier_group)
        torch.cuda.synchronize()                                               # trace_info : t_12920, t_14333
        elapsed = time.time() - self._start_time                               # trace_info : t_12921, t_14334
        self._elapsed += elapsed                                               # trace_info : t_12922, t_14335
        self._active_time += elapsed                                           # trace_info : t_12923, t_14336
        self._started = False                                                  # trace_info : t_12924, t_14337

    def reset(self):
        """Reset timer.
        """
        # Don't reset _active_time
        self._elapsed = 0.0                                                    # trace_info : t_14381, t_14393
        self._started = False                                                  # trace_info : t_14382, t_14394

    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started                                               # trace_info : t_14376, t_14388
        # If the timing in progress, end it first.
        if self._started:                                                      # trace_info : t_14377, t_14389
            self.stop(barrier=barrier)
        # Get the elapsed time.
        _elapsed = self._elapsed                                               # trace_info : t_14378, t_14390
        # Reset the elapsed time
        if reset:                                                              # trace_info : t_14379, t_14391
            self.reset()                                                       # trace_info : t_14380, t_14392
        # If timing was in progress, set it back.
        if _started:                                                           # trace_info : t_14383, t_14395
            self.start(barrier=barrier)
        return _elapsed                                                        # trace_info : t_14384, t_14396

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
        self._log_level = log_level                                            # trace_info : t_4047
        allowed_log_options = set(['max', 'minmax', 'all'])                    # trace_info : t_4048
        assert (
            log_option in allowed_log_options                                  # trace_info : t_4049
        ), 'input log option {} is invalid. It must be one of {}'.format(
            log_option, allowed_log_options
        )
        self._log_option = log_option                                          # trace_info : t_4050
        self._timers = {}                                                      # trace_info : t_4051
        self._log_levels = {}                                                  # trace_info : t_4052
        self._dummy_timer = DummyTimer()                                       # trace_info : t_4053
        self._max_log_level = 2                                                # trace_info : t_4056

    def __call__(self, name, log_level=None):
        """Call timer with name and log level."""
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:                                               # trace_info : t_5758, t_12915, t_12942, t_14328, t_14486, ...
            if log_level is not None:                                          # trace_info : t_12916, t_14329
                assert log_level == self._log_levels[name], (
                    'input log level {} does not match already existing '
                    'log level {} for {} timer'.format(log_level, self._log_levels[name], name)
                )
            return self._timers[name]                                          # trace_info : t_12917, t_14330
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:                                                  # trace_info : t_5759, t_12943, t_14487, t_14835, t_14885, ...
            log_level = self._max_log_level                                    # trace_info : t_15155, t_16486, t_16583, t_16636, t_16662, ...
        assert (
            log_level <= self._max_log_level                                   # trace_info : t_5760, t_12944, t_14488, t_14836, t_14886, ...
        ), 'log level {} is larger than max supported log level {}'.format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:                                        # trace_info : t_5761, t_12945, t_14489, t_14837, t_14887, ...
            return self._dummy_timer                                           # trace_info : t_14838, t_14888, t_15046, t_15158, t_16489, ...
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)                                       # trace_info : t_5762, t_12946, t_14490
        self._log_levels[name] = log_level                                     # trace_info : t_5770, t_12954, t_14498
        return self._timers[name]                                              # trace_info : t_5771, t_12955, t_14499

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
        if barrier:                                                            # trace_info : t_14366
            torch.distributed.barrier()                                        # trace_info : t_14367

        world_size = torch.distributed.get_world_size()                        # trace_info : t_14368
        rank = torch.distributed.get_rank()                                    # trace_info : t_14369

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros(                                       # trace_info : t_14370, t_14372
            (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()# trace_info : t_14371
        )
        for i, name in enumerate(names):                                       # trace_info : t_14373, t_14385, t_14397
            if name in self._timers:                                           # trace_info : t_14374, t_14386
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)# trace_info : t_14375, t_14387

        # See the note above for why we are not using gather.
        torch.distributed._all_gather_base(                                    # trace_info : t_14398, t_14400
            rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1)    # trace_info : t_14399
        )

        return rank_name_to_time                                               # trace_info : t_14401

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)# trace_info : t_14365
        name_to_min_max_time = {}                                              # trace_info : t_14402
        for i, name in enumerate(names):                                       # trace_info : t_14403, t_14410, t_14417
            rank_to_time = rank_name_to_time[:, i]                             # trace_info : t_14404, t_14411
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]                    # trace_info : t_14405, t_14412
            # If the timer exists:
            if rank_to_time.numel() > 0:                                       # trace_info : t_14406, t_14413
                name_to_min_max_time[name] = (                                 # trace_info : t_14409, t_14416
                    rank_to_time.min().item() / normalizer,                    # trace_info : t_14407, t_14414
                    rank_to_time.max().item() / normalizer,                    # trace_info : t_14408, t_14415
                )
        return name_to_min_max_time                                            # trace_info : t_14418

    def _get_global_min_max_time_string(self, names, reset, barrier, normalizer, max_only):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)# trace_info : t_14364
        if not name_to_min_max_time:                                           # trace_info : t_14419
            return None
        if max_only:                                                           # trace_info : t_14420
            output_string = 'max time across ranks (ms):'
        else:
            output_string = '(min, max) time across ranks (ms):'               # trace_info : t_14421
        for name in name_to_min_max_time:                                      # trace_info : t_14422, t_14428, t_14434
            min_time, max_time = name_to_min_max_time[name]                    # trace_info : t_14423, t_14429
            if max_only:                                                       # trace_info : t_14424, t_14430
                output_string += '\n    {}: {:.2f}'.format((name + ' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(          # trace_info : t_14425, t_14427, t_14431, t_14433
                    (name + ' ').ljust(48, '.'), min_time, max_time            # trace_info : t_14426, t_14432
                )
        return output_string                                                   # trace_info : t_14435

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

        if names == None:  # get all registered timers                         # trace_info : t_14356
            names = self._timers.keys()

        assert normalizer > 0.0                                                # trace_info : t_14357
        if self._log_option in ['max', 'minmax']:                              # trace_info : t_14358
            max_only = False                                                   # trace_info : t_14359
            if self._log_option == 'max':                                      # trace_info : t_14360
                max_only = True
            output_string = self._get_global_min_max_time_string(              # trace_info : t_14361, t_14363
                names, reset, barrier, normalizer / 1000.0, max_only           # trace_info : t_14362
            )
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception('unknown timing log option {}'.format(self._log_option))
        return output_string                                                   # trace_info : t_14436

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

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)# trace_info : t_14355
        # If no input rank is provided, log on last rank.
        if rank is None:                                                       # trace_info : t_14437
            rank = torch.distributed.get_world_size() - 1                      # trace_info : t_14438
        if rank == torch.distributed.get_rank() and output_string is not None: # trace_info : t_14439
            print(output_string, flush=True)                                   # trace_info : t_14440

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
