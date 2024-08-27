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
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):    # trace_info : t_8748, t_8767, t_8806, t_9811, t_9828, ...
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:                                                       # trace_info : t_8749, t_8768, t_8807, t_9812, t_9829, ...
            device = torch.device('cuda')                                      # trace_info : t_8750, t_8769, t_8808, t_9813, t_9830, ...
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():                                                              # trace_info : t_8751, t_8770, t_8809, t_9814, t_9831, ...
            idx = device.index                                                 # trace_info : t_8753, t_8772, t_8811, t_9816, t_9833, ...
            if idx is None:                                                    # trace_info : t_8754, t_8773, t_8812, t_9817, t_9834, ...
                idx = torch.cuda.current_device()                              # trace_info : t_8755, t_8774, t_8813, t_9818, t_9835, ...
            default_generator = torch.cuda.default_generators[idx]             # trace_info : t_8756, t_8775, t_8814, t_9819, t_9836, ...
            default_generator.set_state(new_state)                             # trace_info : t_8757, t_8776, t_8815, t_9820, t_9837, ...

    _lazy_call(cb)                                                             # trace_info : t_8752, t_8771, t_8810, t_9815, t_9832, ...


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
        self.reset()                                                           # trace_info : t_8729

    def is_initialized(self):
        return self._is_initialized                                            # trace_info : t_10084, t_10166, t_10267, t_10520, t_10633, ...

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False                                           # trace_info : t_8730, t_8735

        # Map from a string name to the cuda rng state.
        self.states_ = {}                                                      # trace_info : t_8731, t_8736

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()                                                    # trace_info : t_8732, t_8737

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
        self._is_initialized = True                                            # trace_info : t_8740, t_8759, t_8798
        # Check seed is not already used.
        if seed in self.seeds_:                                                # trace_info : t_8741, t_8760, t_8799
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)                                                  # trace_info : t_8742, t_8761, t_8800
        # Check that state is not already defined.
        if name in self.states_:                                               # trace_info : t_8743, t_8762, t_8801
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()                            # trace_info : t_8744, t_8763, t_8802
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)                                           # trace_info : t_8745, t_8764, t_8803
        self.states_[name] = torch.cuda.get_rng_state()                        # trace_info : t_8746, t_8765, t_8804
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)                                    # trace_info : t_8747, t_8766, t_8805

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:                                           # trace_info : t_9808, t_10180, t_10283, t_10536, t_10647, ...
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()                       # trace_info : t_9809, t_10181, t_10284, t_10537, t_10648, ...
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])                                # trace_info : t_9810, t_10182, t_10285, t_10538, t_10649, ...
        # Do the stuff we wanted to do.
        try:                                                                   # trace_info : t_9821, t_10193, t_10296, t_10549, t_10660, ...
            yield                                                              # trace_info : t_9822, t_10194, t_10297, t_10550, t_10661, ...
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()                    # trace_info : t_9826, t_10196, t_10299, t_10552, t_10663, ...
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)                           # trace_info : t_9827, t_10197, t_10300, t_10553, t_10664, ...


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(use_te_rng_tracker: bool = False):
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:                                    # trace_info : t_8725, t_9805, t_10081, t_10094, t_10163, ...
        return                                                                 # trace_info : t_9806, t_10082, t_10095, t_10164, t_10178, ...
    if use_te_rng_tracker:                                                     # trace_info : t_8726
        try:
            import transformer_engine.pytorch as te

            _te_version = packaging.version.Version(version("transformer-engine"))
            if _te_version < packaging.version.Version("1.5.0"):
                raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")
        except:
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine, but not installed")
    if use_te_rng_tracker:                                                     # trace_info : t_8727
        _CUDA_RNG_STATE_TRACKER = te.distributed.CudaRNGStatesTracker()
    else:
        _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()                       # trace_info : t_8728
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True                                 # trace_info : t_8733


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    initialize_rng_tracker()                                                   # trace_info : t_9804, t_10080, t_10093, t_10162, t_10176, ...
    return _CUDA_RNG_STATE_TRACKER                                             # trace_info : t_9807, t_10083, t_10096, t_10165, t_10179, ...


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
    offset = seed + 2718                                                       # trace_info : t_8716
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()     # trace_info : t_8717
    # Data parallel gets the original seed.
    data_parallel_seed = seed                                                  # trace_info : t_8723

    initialize_rng_tracker()                                                   # trace_info : t_8724
    _CUDA_RNG_STATE_TRACKER.reset()                                            # trace_info : t_8734
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)                                 # trace_info : t_8738
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)# trace_info : t_8739

    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)# trace_info : t_8758

    expert_parallel_seed = (                                                   # trace_info : t_8796
        seed + 1024 + 100 * get_expert_model_parallel_rank() + get_tensor_model_parallel_rank()# trace_info : t_8777
    )
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)# trace_info : t_8797


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
