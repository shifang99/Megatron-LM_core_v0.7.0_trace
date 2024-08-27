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
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):    # trace_info : t_8384, t_8403, t_8442, t_9459, t_9476, ...
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:                                                       # trace_info : t_8385, t_8404, t_8443, t_9460, t_9477, ...
            device = torch.device('cuda')                                      # trace_info : t_8386, t_8405, t_8444, t_9461, t_9478, ...
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():                                                              # trace_info : t_8387, t_8406, t_8445, t_9462, t_9479, ...
            idx = device.index                                                 # trace_info : t_8389, t_8408, t_8447, t_9464, t_9481, ...
            if idx is None:                                                    # trace_info : t_8390, t_8409, t_8448, t_9465, t_9482, ...
                idx = torch.cuda.current_device()                              # trace_info : t_8391, t_8410, t_8449, t_9466, t_9483, ...
            default_generator = torch.cuda.default_generators[idx]             # trace_info : t_8392, t_8411, t_8450, t_9467, t_9484, ...
            default_generator.set_state(new_state)                             # trace_info : t_8393, t_8412, t_8451, t_9468, t_9485, ...

    _lazy_call(cb)                                                             # trace_info : t_8388, t_8407, t_8446, t_9463, t_9480, ...


def get_expert_parallel_rng_tracker_name():
    global _EXPERT_PARALLEL_RNG_TRACKER_NAME
    return _EXPERT_PARALLEL_RNG_TRACKER_NAME                                   # trace_info : t_10450, t_10611, t_11590, t_11751


def get_data_parallel_rng_tracker_name():
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME                                     # trace_info : t_10302, t_11442


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        self.reset()                                                           # trace_info : t_8365

    def is_initialized(self):
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False                                           # trace_info : t_8366, t_8371

        # Map from a string name to the cuda rng state.
        self.states_ = {}                                                      # trace_info : t_8367, t_8372

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()                                                    # trace_info : t_8368, t_8373

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
        self._is_initialized = True                                            # trace_info : t_8376, t_8395, t_8434
        # Check seed is not already used.
        if seed in self.seeds_:                                                # trace_info : t_8377, t_8396, t_8435
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)                                                  # trace_info : t_8378, t_8397, t_8436
        # Check that state is not already defined.
        if name in self.states_:                                               # trace_info : t_8379, t_8398, t_8437
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()                            # trace_info : t_8380, t_8399, t_8438
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)                                           # trace_info : t_8381, t_8400, t_8439
        self.states_[name] = torch.cuda.get_rng_state()                        # trace_info : t_8382, t_8401, t_8440
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)                                    # trace_info : t_8383, t_8402, t_8441

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:                                           # trace_info : t_9456, t_9865, t_10007, t_10303, t_10451, ...
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()                       # trace_info : t_9457, t_9866, t_10008, t_10304, t_10452, ...
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])                                # trace_info : t_9458, t_9867, t_10009, t_10305, t_10453, ...
        # Do the stuff we wanted to do.
        try:                                                                   # trace_info : t_9469, t_9878, t_10020, t_10316, t_10464, ...
            yield                                                              # trace_info : t_9470, t_9879, t_10021, t_10317, t_10465, ...
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()                    # trace_info : t_9474, t_9883, t_10025, t_10321, t_10469, ...
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)                           # trace_info : t_9475, t_9884, t_10026, t_10322, t_10470, ...


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(use_te_rng_tracker: bool = False):
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:                                    # trace_info : t_8361, t_9453, t_9862, t_10004, t_10299, ...
        return                                                                 # trace_info : t_9454, t_9863, t_10005, t_10300, t_10448, ...
    if use_te_rng_tracker:                                                     # trace_info : t_8362
        try:
            import transformer_engine.pytorch as te

            _te_version = packaging.version.Version(version("transformer-engine"))
            if _te_version < packaging.version.Version("1.5.0"):
                raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")
        except:
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine, but not installed")
    if use_te_rng_tracker:                                                     # trace_info : t_8363
        _CUDA_RNG_STATE_TRACKER = te.distributed.CudaRNGStatesTracker()
    else:
        _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()                       # trace_info : t_8364
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True                                 # trace_info : t_8369


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    initialize_rng_tracker()                                                   # trace_info : t_9452, t_9861, t_10003, t_10298, t_10446, ...
    return _CUDA_RNG_STATE_TRACKER                                             # trace_info : t_9455, t_9864, t_10006, t_10301, t_10449, ...


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
    offset = seed + 2718                                                       # trace_info : t_8352
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()     # trace_info : t_8353
    # Data parallel gets the original seed.
    data_parallel_seed = seed                                                  # trace_info : t_8359

    initialize_rng_tracker()                                                   # trace_info : t_8360
    _CUDA_RNG_STATE_TRACKER.reset()                                            # trace_info : t_8370
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)                                 # trace_info : t_8374
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)# trace_info : t_8375

    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)# trace_info : t_8394

    expert_parallel_seed = (                                                   # trace_info : t_8432
        seed + 1024 + 100 * get_expert_model_parallel_rank() + get_tensor_model_parallel_rank()# trace_info : t_8413
    )
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)# trace_info : t_8433


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
