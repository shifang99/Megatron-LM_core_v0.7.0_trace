# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys
import torch

from megatron.training import dist_signal_handler
from megatron.core import Timers
from megatron.training.tokenizer import build_tokenizer
from .microbatches import build_num_microbatches_calculator

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')                           # trace_info : t_5852, t_5858, t_5863, t_10376, t_10453, ...
    return _GLOBAL_ARGS                                                        # trace_info : t_5854, t_5860, t_5865, t_10378, t_10455, ...


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()                           # trace_info : t_19385, t_19405, t_19408, t_19678, t_22729, ...


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,               # trace_info : t_19400, t_19402, t_19411, t_19413, t_22956, ...
                                               consistency_check)              # trace_info : t_19401, t_19412, t_22957, t_22968, t_26567, ...


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')                 # trace_info : t_17896
    return _GLOBAL_TOKENIZER                                                   # trace_info : t_17898


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER                                          # trace_info : t_19335, t_22887, t_26497, t_30107, t_30170


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER                                                # trace_info : t_22889, t_26499, t_30109, t_30173


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER                                                  # trace_info : t_10622, t_19343, t_22891, t_26501, t_30111

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME                                             # trace_info : t_10372


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')                       # trace_info : t_10457, t_10618, t_10653, t_19326, t_19428, ...
    return _GLOBAL_TIMERS                                                      # trace_info : t_10459, t_10620, t_10655, t_19328, t_19430, ...


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()



def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None                                                    # trace_info : t_3100

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')                       # trace_info : t_3101
    set_args(args)                                                             # trace_info : t_3103

    _build_num_microbatches_calculator(args)                                   # trace_info : t_3105
    if build_tokenizer:                                                        # trace_info : t_3135
        _ = _build_tokenizer(args)                                             # trace_info : t_3136
    _set_tensorboard_writer(args)                                              # trace_info : t_5812
    _set_wandb_writer(args)                                                    # trace_info : t_5820
    _set_one_logger(args)                                                      # trace_info : t_5826
    _set_adlr_autoresume(args)                                                 # trace_info : t_5830
    _set_timers(args)                                                          # trace_info : t_5834

    if args.exit_signal_handler:                                               # trace_info : t_5848
        _set_signal_handler()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args                                                        # trace_info : t_3104


def _build_num_microbatches_calculator(args):

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,        # trace_info : t_3106, t_3108
                                   'num microbatches calculator')              # trace_info : t_3107

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(   # trace_info : t_3110, t_3112
        args)                                                                  # trace_info : t_3111


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')             # trace_info : t_3137
    _GLOBAL_TOKENIZER = build_tokenizer(args)                                  # trace_info : t_3139
    return _GLOBAL_TOKENIZER                                                   # trace_info : t_5811


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,                 # trace_info : t_5813, t_5815
                                   'tensorboard writer')                       # trace_info : t_5814

    if hasattr(args, 'tensorboard_dir') and \                                  # trace_info : t_5817, t_5819
       args.tensorboard_dir and args.rank == (args.world_size - 1):            # trace_info : t_5818
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,                       # trace_info : t_5821, t_5823
                                   'wandb writer')                             # trace_info : t_5822
    if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):# trace_info : t_5825
        if args.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, 'wandb')
        wandb_kwargs = {
            'dir': save_dir,
            'name': args.wandb_exp_name,
            'project': args.wandb_project,
            'config': vars(args)}
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')           # trace_info : t_5827

    if args.enable_one_logger and args.rank == (args.world_size - 1):          # trace_info : t_5829
        try:
            from one_logger.core import OneLogger
            config = {
               'project': args.one_logger_project,
               'entity': args.one_logger_entity,
               'name': args.one_logger_run_name
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except BaseException:
            print('WARNING: one_logger package is required to enable e2e metrics '
                  'tracking. Try pip install '
                  '--index-url=https://sc-hw-artf.nvidia.com/api/pypi/hwinf-ml-pypi/simple'
                  ' one_logger to install it')

def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume') # trace_info : t_5831

    if args.adlr_autoresume:                                                   # trace_info : t_5833
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')                   # trace_info : t_5835
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)     # trace_info : t_5837


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)              # trace_info : t_5853, t_5859, t_5864, t_10377, t_10454, ...


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)              # trace_info : t_3102, t_3109, t_3138, t_5816, t_5824, ...


