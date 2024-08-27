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
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')                           # trace_info : t_4316, t_4322, t_4327, t_8822, t_8896, ...
    return _GLOBAL_ARGS                                                        # trace_info : t_4318, t_4324, t_4329, t_8824, t_8898, ...


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()                           # trace_info : t_17377, t_17397, t_17400, t_17670, t_20295, ...


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,               # trace_info : t_17392, t_17394, t_17403, t_17405, t_20522, ...
                                               consistency_check)              # trace_info : t_17393, t_17404, t_20523, t_20534, t_23709, ...


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')                 # trace_info : t_15888
    return _GLOBAL_TOKENIZER                                                   # trace_info : t_15890


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER                                          # trace_info : t_17327, t_20453, t_23639, t_26825, t_26888


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER                                                # trace_info : t_20455, t_23641, t_26827, t_26891


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER                                                  # trace_info : t_9065, t_17335, t_20457, t_23643, t_26829

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME                                             # trace_info : t_8818


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')                       # trace_info : t_8900, t_9061, t_9096, t_17318, t_17420, ...
    return _GLOBAL_TIMERS                                                      # trace_info : t_8902, t_9063, t_9098, t_17320, t_17422, ...


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
    _set_tensorboard_writer(args)                                              # trace_info : t_4276
    _set_wandb_writer(args)                                                    # trace_info : t_4284
    _set_one_logger(args)                                                      # trace_info : t_4290
    _set_adlr_autoresume(args)                                                 # trace_info : t_4294
    _set_timers(args)                                                          # trace_info : t_4298

    if args.exit_signal_handler:                                               # trace_info : t_4312
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
    return _GLOBAL_TOKENIZER                                                   # trace_info : t_4275


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,                 # trace_info : t_4277, t_4279
                                   'tensorboard writer')                       # trace_info : t_4278

    if hasattr(args, 'tensorboard_dir') and \                                  # trace_info : t_4281, t_4283
       args.tensorboard_dir and args.rank == (args.world_size - 1):            # trace_info : t_4282
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
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,                       # trace_info : t_4285, t_4287
                                   'wandb writer')                             # trace_info : t_4286
    if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):# trace_info : t_4289
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
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')           # trace_info : t_4291

    if args.enable_one_logger and args.rank == (args.world_size - 1):          # trace_info : t_4293
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
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume') # trace_info : t_4295

    if args.adlr_autoresume:                                                   # trace_info : t_4297
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
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')                   # trace_info : t_4299
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)     # trace_info : t_4301


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)              # trace_info : t_4317, t_4323, t_4328, t_8823, t_8897, ...


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)              # trace_info : t_3102, t_3109, t_3138, t_4280, t_4288, ...


