# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib
from importlib.metadata import version

import torch
from pkg_resources import packaging
from torch import _C
from torch.cuda import _lazy_call
from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import safely_set_viewless_tensor_data

from .utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
_EXPERT_PARALLEL_RNG_TRACKER_NAME = 'expert-parallel-rng'
_DATA_PARALLEL_RNG_TRACKER_NAME = 'data-parallel-rng'


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):    # trace_info : t_10302, t_10321, t_10360, t_11372, t_11389, ...
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:                                                       # trace_info : t_10303, t_10322, t_10361, t_11373, t_11390, ...
            device = torch.device('cuda')                                      # trace_info : t_10304, t_10323, t_10362, t_11374, t_11391, ...
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():                                                              # trace_info : t_10305, t_10324, t_10363, t_11375, t_11392, ...
            idx = device.index                                                 # trace_info : t_10307, t_10326, t_10365, t_11377, t_11394, ...
            if idx is None:                                                    # trace_info : t_10308, t_10327, t_10366, t_11378, t_11395, ...
                idx = torch.cuda.current_device()                              # trace_info : t_10309, t_10328, t_10367, t_11379, t_11396, ...
            default_generator = torch.cuda.default_generators[idx]             # trace_info : t_10310, t_10329, t_10368, t_11380, t_11397, ...
            default_generator.set_state(new_state)                             # trace_info : t_10311, t_10330, t_10369, t_11381, t_11398, ...

    _lazy_call(cb)                                                             # trace_info : t_10306, t_10325, t_10364, t_11376, t_11393, ...


def get_expert_parallel_rng_tracker_name():
    global _EXPERT_PARALLEL_RNG_TRACKER_NAME
    return _EXPERT_PARALLEL_RNG_TRACKER_NAME


def get_data_parallel_rng_tracker_name():
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        self.reset()                                                           # trace_info : t_10283

    def is_initialized(self):
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False                                           # trace_info : t_10284, t_10289

        # Map from a string name to the cuda rng state.
        self.states_ = {}                                                      # trace_info : t_10285, t_10290

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()                                                    # trace_info : t_10286, t_10291

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        self._is_initialized = True                                            # trace_info : t_10294, t_10313, t_10352
        # Check seed is not already used.
        if seed in self.seeds_:                                                # trace_info : t_10295, t_10314, t_10353
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)                                                  # trace_info : t_10296, t_10315, t_10354
        # Check that state is not already defined.
        if name in self.states_:                                               # trace_info : t_10297, t_10316, t_10355
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()                            # trace_info : t_10298, t_10317, t_10356
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)                                           # trace_info : t_10299, t_10318, t_10357
        self.states_[name] = torch.cuda.get_rng_state()                        # trace_info : t_10300, t_10319, t_10358
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)                                    # trace_info : t_10301, t_10320, t_10359

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:                                           # trace_info : t_11369, t_11780, t_11922, t_12251, t_12409, ...
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()                       # trace_info : t_11370, t_11781, t_11923, t_12252, t_12410, ...
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])                                # trace_info : t_11371, t_11782, t_11924, t_12253, t_12411, ...
        # Do the stuff we wanted to do.
        try:                                                                   # trace_info : t_11382, t_11793, t_11935, t_12264, t_12422, ...
            yield                                                              # trace_info : t_11383, t_11794, t_11936, t_12265, t_12423, ...
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()                    # trace_info : t_11387, t_11798, t_11940, t_12269, t_12427, ...
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)                           # trace_info : t_11388, t_11799, t_11941, t_12270, t_12428, ...


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(use_te_rng_tracker: bool = False):
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:                                    # trace_info : t_10279, t_11366, t_11777, t_11919, t_12248, ...
        return                                                                 # trace_info : t_11367, t_11778, t_11920, t_12249, t_12407, ...
    if use_te_rng_tracker:                                                     # trace_info : t_10280
        try:
            import transformer_engine.pytorch as te

            _te_version = packaging.version.Version(version("transformer-engine"))
            if _te_version < packaging.version.Version("1.5.0"):
                raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")
        except:
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine, but not installed")
    if use_te_rng_tracker:                                                     # trace_info : t_10281
        _CUDA_RNG_STATE_TRACKER = te.distributed.CudaRNGStatesTracker()
    else:
        _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()                       # trace_info : t_10282
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True                                 # trace_info : t_10287


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    initialize_rng_tracker()                                                   # trace_info : t_11365, t_11776, t_11918, t_12247, t_12405, ...
    return _CUDA_RNG_STATE_TRACKER                                             # trace_info : t_11368, t_11779, t_11921, t_12250, t_12408, ...


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
    default state: This is for data parallelism and is the same among a set of model parallel GPUs but different across different model paralle groups. This is used for example for dropout in the non-tensor-model-parallel regions.
    tensor-model-parallel state: This state is different among a set of model parallel GPUs, but the same across data parallel groups. This is used for example for dropout in model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718                                                       # trace_info : t_10270
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()     # trace_info : t_10271
    # Data parallel gets the original seed.
    data_parallel_seed = seed                                                  # trace_info : t_10277

    initialize_rng_tracker()                                                   # trace_info : t_10278
    _CUDA_RNG_STATE_TRACKER.reset()                                            # trace_info : t_10288
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)                                 # trace_info : t_10292
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)# trace_info : t_10293

    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)# trace_info : t_10312

    expert_parallel_seed = (                                                   # trace_info : t_10350
        seed + 1024 + 100 * get_expert_model_parallel_rank() + get_tensor_model_parallel_rank()# trace_info : t_10331
    )
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)# trace_info : t_10351


class CheckpointFunction(torch.autograd.Function):
    """Checkpoint Function 

    This function is adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
            )

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)
