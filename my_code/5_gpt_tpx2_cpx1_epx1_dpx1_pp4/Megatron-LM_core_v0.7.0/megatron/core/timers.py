# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

import time
from abc import ABC, abstractmethod
from typing import List

import torch


class TimerBase(ABC):
    def __init__(self, name):
        self.name = name                                                       # trace_info : t_4311, t_9128, t_16014, t_17557

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
        super().__init__('dummy timer')                                        # trace_info : t_4310

    def start(self, barrier=False):
        return                                                                 # trace_info : t_17893, t_18034, t_18186, t_19504, t_19627, ...

    def stop(self, barrier=False):
        return                                                                 # trace_info : t_18291, t_19440, t_19576, t_19707, t_19782, ...

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
        super().__init__(name)                                                 # trace_info : t_9127, t_16013, t_17556
        self._elapsed = 0.0                                                    # trace_info : t_9129, t_16015, t_17558
        self._active_time = 0.0                                                # trace_info : t_9130, t_16016, t_17559
        self._started = False                                                  # trace_info : t_9131, t_16017, t_17560
        # Note that None will default to the global process group
        self._barrier_group = None                                             # trace_info : t_9132, t_16018, t_17561
        self._start_time = time.time()                                         # trace_info : t_9133, t_16019, t_17562

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
        assert not self._started, 'timer has already been started'             # trace_info : t_9136, t_16024, t_17565
        if barrier:                                                            # trace_info : t_9137, t_16025, t_17566
            torch.distributed.barrier(group=self._barrier_group)               # trace_info : t_9138, t_16026, t_17567
        torch.cuda.synchronize()                                               # trace_info : t_9139, t_16027, t_17568
        self._start_time = time.time()                                         # trace_info : t_9140, t_16028, t_17569
        self._started = True                                                   # trace_info : t_9141, t_16029, t_17570

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, 'timer is not started'                           # trace_info : t_15984, t_17397
        if barrier:                                                            # trace_info : t_15985, t_17398
            torch.distributed.barrier(group=self._barrier_group)
        torch.cuda.synchronize()                                               # trace_info : t_15986, t_17399
        elapsed = time.time() - self._start_time                               # trace_info : t_15987, t_17400
        self._elapsed += elapsed                                               # trace_info : t_15988, t_17401
        self._active_time += elapsed                                           # trace_info : t_15989, t_17402
        self._started = False                                                  # trace_info : t_15990, t_17403

    def reset(self):
        """Reset timer.
        """
        # Don't reset _active_time
        self._elapsed = 0.0                                                    # trace_info : t_17447, t_17459
        self._started = False                                                  # trace_info : t_17448, t_17460

    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started                                               # trace_info : t_17442, t_17454
        # If the timing in progress, end it first.
        if self._started:                                                      # trace_info : t_17443, t_17455
            self.stop(barrier=barrier)
        # Get the elapsed time.
        _elapsed = self._elapsed                                               # trace_info : t_17444, t_17456
        # Reset the elapsed time
        if reset:                                                              # trace_info : t_17445, t_17457
            self.reset()                                                       # trace_info : t_17446, t_17458
        # If timing was in progress, set it back.
        if _started:                                                           # trace_info : t_17449, t_17461
            self.start(barrier=barrier)
        return _elapsed                                                        # trace_info : t_17450, t_17462

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
        self._log_level = log_level                                            # trace_info : t_4303
        allowed_log_options = set(['max', 'minmax', 'all'])                    # trace_info : t_4304
        assert (
            log_option in allowed_log_options                                  # trace_info : t_4305
        ), 'input log option {} is invalid. It must be one of {}'.format(
            log_option, allowed_log_options
        )
        self._log_option = log_option                                          # trace_info : t_4306
        self._timers = {}                                                      # trace_info : t_4307
        self._log_levels = {}                                                  # trace_info : t_4308
        self._dummy_timer = DummyTimer()                                       # trace_info : t_4309
        self._max_log_level = 2                                                # trace_info : t_4312

    def __call__(self, name, log_level=None):
        """Call timer with name and log level."""
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:                                               # trace_info : t_9122, t_15981, t_16008, t_17394, t_17551, ...
            if log_level is not None:                                          # trace_info : t_15982, t_17395
                assert log_level == self._log_levels[name], (
                    'input log level {} does not match already existing '
                    'log level {} for {} timer'.format(log_level, self._log_levels[name], name)
                )
            return self._timers[name]                                          # trace_info : t_15983, t_17396
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:                                                  # trace_info : t_9123, t_16009, t_17552, t_17889, t_18030, ...
            log_level = self._max_log_level                                    # trace_info : t_18287, t_19436, t_19572, t_19703, t_19778, ...
        assert (
            log_level <= self._max_log_level                                   # trace_info : t_9124, t_16010, t_17553, t_17890, t_18031, ...
        ), 'log level {} is larger than max supported log level {}'.format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:                                        # trace_info : t_9125, t_16011, t_17554, t_17891, t_18032, ...
            return self._dummy_timer                                           # trace_info : t_17892, t_18033, t_18185, t_18290, t_19439, ...
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)                                       # trace_info : t_9126, t_16012, t_17555
        self._log_levels[name] = log_level                                     # trace_info : t_9134, t_16020, t_17563
        return self._timers[name]                                              # trace_info : t_9135, t_16021, t_17564

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
        if barrier:                                                            # trace_info : t_17432
            torch.distributed.barrier()                                        # trace_info : t_17433

        world_size = torch.distributed.get_world_size()                        # trace_info : t_17434
        rank = torch.distributed.get_rank()                                    # trace_info : t_17435

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros(                                       # trace_info : t_17436, t_17438
            (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()# trace_info : t_17437
        )
        for i, name in enumerate(names):                                       # trace_info : t_17439, t_17451, t_17463
            if name in self._timers:                                           # trace_info : t_17440, t_17452
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)# trace_info : t_17441, t_17453

        # See the note above for why we are not using gather.
        torch.distributed._all_gather_base(                                    # trace_info : t_17464, t_17466
            rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1)    # trace_info : t_17465
        )

        return rank_name_to_time                                               # trace_info : t_17467

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)# trace_info : t_17431
        name_to_min_max_time = {}                                              # trace_info : t_17468
        for i, name in enumerate(names):                                       # trace_info : t_17469, t_17476, t_17483
            rank_to_time = rank_name_to_time[:, i]                             # trace_info : t_17470, t_17477
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]                    # trace_info : t_17471, t_17478
            # If the timer exists:
            if rank_to_time.numel() > 0:                                       # trace_info : t_17472, t_17479
                name_to_min_max_time[name] = (                                 # trace_info : t_17475, t_17482
                    rank_to_time.min().item() / normalizer,                    # trace_info : t_17473, t_17480
                    rank_to_time.max().item() / normalizer,                    # trace_info : t_17474, t_17481
                )
        return name_to_min_max_time                                            # trace_info : t_17484

    def _get_global_min_max_time_string(self, names, reset, barrier, normalizer, max_only):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)# trace_info : t_17430
        if not name_to_min_max_time:                                           # trace_info : t_17485
            return None
        if max_only:                                                           # trace_info : t_17486
            output_string = 'max time across ranks (ms):'
        else:
            output_string = '(min, max) time across ranks (ms):'               # trace_info : t_17487
        for name in name_to_min_max_time:                                      # trace_info : t_17488, t_17494, t_17500
            min_time, max_time = name_to_min_max_time[name]                    # trace_info : t_17489, t_17495
            if max_only:                                                       # trace_info : t_17490, t_17496
                output_string += '\n    {}: {:.2f}'.format((name + ' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(          # trace_info : t_17491, t_17493, t_17497, t_17499
                    (name + ' ').ljust(48, '.'), min_time, max_time            # trace_info : t_17492, t_17498
                )
        return output_string                                                   # trace_info : t_17501

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

        if names == None:  # get all registered timers                         # trace_info : t_17422
            names = self._timers.keys()

        assert normalizer > 0.0                                                # trace_info : t_17423
        if self._log_option in ['max', 'minmax']:                              # trace_info : t_17424
            max_only = False                                                   # trace_info : t_17425
            if self._log_option == 'max':                                      # trace_info : t_17426
                max_only = True
            output_string = self._get_global_min_max_time_string(              # trace_info : t_17427, t_17429
                names, reset, barrier, normalizer / 1000.0, max_only           # trace_info : t_17428
            )
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception('unknown timing log option {}'.format(self._log_option))
        return output_string                                                   # trace_info : t_17502

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

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)# trace_info : t_17421
        # If no input rank is provided, log on last rank.
        if rank is None:                                                       # trace_info : t_17503
            rank = torch.distributed.get_world_size() - 1                      # trace_info : t_17504
        if rank == torch.distributed.get_rank() and output_string is not None: # trace_info : t_17505
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
