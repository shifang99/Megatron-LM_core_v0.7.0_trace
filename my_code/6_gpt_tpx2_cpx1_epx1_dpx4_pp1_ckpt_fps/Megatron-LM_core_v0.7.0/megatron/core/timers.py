# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

import time
from abc import ABC, abstractmethod
from typing import List

import torch


class TimerBase(ABC):
    def __init__(self, name):
        self.name = name                                                       # trace_info : t_4311, t_8793, t_15977, t_17628, t_24907

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
        return                                                                 # trace_info : t_17975, t_18025, t_18183, t_19658, t_19737, ...

    def stop(self, barrier=False):
        return                                                                 # trace_info : t_18304, t_19618, t_19715, t_19769, t_19799, ...

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
        super().__init__(name)                                                 # trace_info : t_8792, t_15976, t_17627, t_24906
        self._elapsed = 0.0                                                    # trace_info : t_8794, t_15978, t_17629, t_24908
        self._active_time = 0.0                                                # trace_info : t_8795, t_15979, t_17630, t_24909
        self._started = False                                                  # trace_info : t_8796, t_15980, t_17631, t_24910
        # Note that None will default to the global process group
        self._barrier_group = None                                             # trace_info : t_8797, t_15981, t_17632, t_24911
        self._start_time = time.time()                                         # trace_info : t_8798, t_15982, t_17633, t_24912

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
        assert not self._started, 'timer has already been started'             # trace_info : t_8801, t_15987, t_17636, t_24915, t_88840
        if barrier:                                                            # trace_info : t_8802, t_15988, t_17637, t_24916, t_88841
            torch.distributed.barrier(group=self._barrier_group)               # trace_info : t_8803, t_15989, t_17638, t_24917, t_88842
        torch.cuda.synchronize()                                               # trace_info : t_8804, t_15990, t_17639, t_24918, t_88843
        self._start_time = time.time()                                         # trace_info : t_8805, t_15991, t_17640, t_24919, t_88844
        self._started = True                                                   # trace_info : t_8806, t_15992, t_17641, t_24920, t_88845

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, 'timer is not started'                           # trace_info : t_15947, t_17468, t_24880, t_88765
        if barrier:                                                            # trace_info : t_15948, t_17469, t_24881, t_88766
            torch.distributed.barrier(group=self._barrier_group)               # trace_info : t_88767
        torch.cuda.synchronize()                                               # trace_info : t_15949, t_17470, t_24882, t_88768
        elapsed = time.time() - self._start_time                               # trace_info : t_15950, t_17471, t_24883, t_88769
        self._elapsed += elapsed                                               # trace_info : t_15951, t_17472, t_24884, t_88770
        self._active_time += elapsed                                           # trace_info : t_15952, t_17473, t_24885, t_88771
        self._started = False                                                  # trace_info : t_15953, t_17474, t_24886, t_88772

    def reset(self):
        """Reset timer.
        """
        # Don't reset _active_time
        self._elapsed = 0.0                                                    # trace_info : t_17518, t_17530, t_88799
        self._started = False                                                  # trace_info : t_17519, t_17531, t_88800

    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started                                               # trace_info : t_17513, t_17525, t_88794
        # If the timing in progress, end it first.
        if self._started:                                                      # trace_info : t_17514, t_17526, t_88795
            self.stop(barrier=barrier)
        # Get the elapsed time.
        _elapsed = self._elapsed                                               # trace_info : t_17515, t_17527, t_88796
        # Reset the elapsed time
        if reset:                                                              # trace_info : t_17516, t_17528, t_88797
            self.reset()                                                       # trace_info : t_17517, t_17529, t_88798
        # If timing was in progress, set it back.
        if _started:                                                           # trace_info : t_17520, t_17532, t_88801
            self.start(barrier=barrier)
        return _elapsed                                                        # trace_info : t_17521, t_17533, t_88802

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
        if name in self._timers:                                               # trace_info : t_8787, t_15944, t_15971, t_17465, t_17622, ...
            if log_level is not None:                                          # trace_info : t_15945, t_17466, t_24878, t_88763, t_88837
                assert log_level == self._log_levels[name], (                  # trace_info : t_88838
                    'input log level {} does not match already existing '
                    'log level {} for {} timer'.format(log_level, self._log_levels[name], name)
                )
            return self._timers[name]                                          # trace_info : t_15946, t_17467, t_24879, t_88764, t_88839
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:                                                  # trace_info : t_8788, t_15972, t_17623, t_17971, t_18021, ...
            log_level = self._max_log_level                                    # trace_info : t_18300, t_19614, t_19711, t_19765, t_19795, ...
        assert (
            log_level <= self._max_log_level                                   # trace_info : t_8789, t_15973, t_17624, t_17972, t_18022, ...
        ), 'log level {} is larger than max supported log level {}'.format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:                                        # trace_info : t_8790, t_15974, t_17625, t_17973, t_18023, ...
            return self._dummy_timer                                           # trace_info : t_17974, t_18024, t_18182, t_18303, t_19617, ...
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)                                       # trace_info : t_8791, t_15975, t_17626, t_24905
        self._log_levels[name] = log_level                                     # trace_info : t_8799, t_15983, t_17634, t_24913
        return self._timers[name]                                              # trace_info : t_8800, t_15984, t_17635, t_24914

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
        if barrier:                                                            # trace_info : t_17503, t_88785
            torch.distributed.barrier()                                        # trace_info : t_17504

        world_size = torch.distributed.get_world_size()                        # trace_info : t_17505, t_88786
        rank = torch.distributed.get_rank()                                    # trace_info : t_17506, t_88787

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros(                                       # trace_info : t_17507, t_17509, t_88788, t_88790
            (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()# trace_info : t_17508, t_88789
        )
        for i, name in enumerate(names):                                       # trace_info : t_17510, t_17522, t_17534, t_88791, t_88803
            if name in self._timers:                                           # trace_info : t_17511, t_17523, t_88792
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)# trace_info : t_17512, t_17524, t_88793

        # See the note above for why we are not using gather.
        torch.distributed._all_gather_base(                                    # trace_info : t_17535, t_17537, t_88804, t_88806
            rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1)    # trace_info : t_17536, t_88805
        )

        return rank_name_to_time                                               # trace_info : t_17538, t_88807

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)# trace_info : t_17502, t_88784
        name_to_min_max_time = {}                                              # trace_info : t_17539, t_88808
        for i, name in enumerate(names):                                       # trace_info : t_17540, t_17547, t_17554, t_88809, t_88816
            rank_to_time = rank_name_to_time[:, i]                             # trace_info : t_17541, t_17548, t_88810
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]                    # trace_info : t_17542, t_17549, t_88811
            # If the timer exists:
            if rank_to_time.numel() > 0:                                       # trace_info : t_17543, t_17550, t_88812
                name_to_min_max_time[name] = (                                 # trace_info : t_17546, t_17553, t_88815
                    rank_to_time.min().item() / normalizer,                    # trace_info : t_17544, t_17551, t_88813
                    rank_to_time.max().item() / normalizer,                    # trace_info : t_17545, t_17552, t_88814
                )
        return name_to_min_max_time                                            # trace_info : t_17555, t_88817

    def _get_global_min_max_time_string(self, names, reset, barrier, normalizer, max_only):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)# trace_info : t_17501, t_88783
        if not name_to_min_max_time:                                           # trace_info : t_17556, t_88818
            return None
        if max_only:                                                           # trace_info : t_17557, t_88819
            output_string = 'max time across ranks (ms):'
        else:
            output_string = '(min, max) time across ranks (ms):'               # trace_info : t_17558, t_88820
        for name in name_to_min_max_time:                                      # trace_info : t_17559, t_17565, t_17571, t_88821, t_88827
            min_time, max_time = name_to_min_max_time[name]                    # trace_info : t_17560, t_17566, t_88822
            if max_only:                                                       # trace_info : t_17561, t_17567, t_88823
                output_string += '\n    {}: {:.2f}'.format((name + ' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(          # trace_info : t_17562, t_17564, t_17568, t_17570, t_88824, ...
                    (name + ' ').ljust(48, '.'), min_time, max_time            # trace_info : t_17563, t_17569, t_88825
                )
        return output_string                                                   # trace_info : t_17572, t_88828

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

        if names == None:  # get all registered timers                         # trace_info : t_17493, t_88775
            names = self._timers.keys()

        assert normalizer > 0.0                                                # trace_info : t_17494, t_88776
        if self._log_option in ['max', 'minmax']:                              # trace_info : t_17495, t_88777
            max_only = False                                                   # trace_info : t_17496, t_88778
            if self._log_option == 'max':                                      # trace_info : t_17497, t_88779
                max_only = True
            output_string = self._get_global_min_max_time_string(              # trace_info : t_17498, t_17500, t_88780, t_88782
                names, reset, barrier, normalizer / 1000.0, max_only           # trace_info : t_17499, t_88781
            )
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception('unknown timing log option {}'.format(self._log_option))
        return output_string                                                   # trace_info : t_17573, t_88829

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

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)# trace_info : t_17492, t_88774
        # If no input rank is provided, log on last rank.
        if rank is None:                                                       # trace_info : t_17574, t_88830
            rank = torch.distributed.get_world_size() - 1                      # trace_info : t_17575, t_88831
        if rank == torch.distributed.get_rank() and output_string is not None: # trace_info : t_17576, t_88832
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
