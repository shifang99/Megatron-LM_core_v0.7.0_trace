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
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):    # trace_info : t_5438, t_5457, t_5496, t_6505, t_6522, ...
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:                                                       # trace_info : t_5439, t_5458, t_5497, t_6506, t_6523, ...
            device = torch.device('cuda')                                      # trace_info : t_5440, t_5459, t_5498, t_6507, t_6524, ...
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():                                                              # trace_info : t_5441, t_5460, t_5499, t_6508, t_6525, ...
            idx = device.index                                                 # trace_info : t_5443, t_5462, t_5501, t_6510, t_6527, ...
            if idx is None:                                                    # trace_info : t_5444, t_5463, t_5502, t_6511, t_6528, ...
                idx = torch.cuda.current_device()                              # trace_info : t_5445, t_5464, t_5503, t_6512, t_6529, ...
            default_generator = torch.cuda.default_generators[idx]             # trace_info : t_5446, t_5465, t_5504, t_6513, t_6530, ...
            default_generator.set_state(new_state)                             # trace_info : t_5447, t_5466, t_5505, t_6514, t_6531, ...

    _lazy_call(cb)                                                             # trace_info : t_5442, t_5461, t_5500, t_6509, t_6526, ...


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
        self.reset()                                                           # trace_info : t_5419

    def is_initialized(self):
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False                                           # trace_info : t_5420, t_5425

        # Map from a string name to the cuda rng state.
        self.states_ = {}                                                      # trace_info : t_5421, t_5426

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()                                                    # trace_info : t_5422, t_5427

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
        self._is_initialized = True                                            # trace_info : t_5430, t_5449, t_5488
        # Check seed is not already used.
        if seed in self.seeds_:                                                # trace_info : t_5431, t_5450, t_5489
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)                                                  # trace_info : t_5432, t_5451, t_5490
        # Check that state is not already defined.
        if name in self.states_:                                               # trace_info : t_5433, t_5452, t_5491
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()                            # trace_info : t_5434, t_5453, t_5492
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)                                           # trace_info : t_5435, t_5454, t_5493
        self.states_[name] = torch.cuda.get_rng_state()                        # trace_info : t_5436, t_5455, t_5494
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)                                    # trace_info : t_5437, t_5456, t_5495

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:                                           # trace_info : t_6502, t_6913, t_7055, t_7384, t_7542, ...
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()                       # trace_info : t_6503, t_6914, t_7056, t_7385, t_7543, ...
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])                                # trace_info : t_6504, t_6915, t_7057, t_7386, t_7544, ...
        # Do the stuff we wanted to do.
        try:                                                                   # trace_info : t_6515, t_6926, t_7068, t_7397, t_7555, ...
            yield                                                              # trace_info : t_6516, t_6927, t_7069, t_7398, t_7556, ...
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()                    # trace_info : t_6520, t_6931, t_7073, t_7402, t_7560, ...
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)                           # trace_info : t_6521, t_6932, t_7074, t_7403, t_7561, ...


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(use_te_rng_tracker: bool = False):
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:                                    # trace_info : t_5415, t_6499, t_6910, t_7052, t_7381, ...
        return                                                                 # trace_info : t_6500, t_6911, t_7053, t_7382, t_7540, ...
    if use_te_rng_tracker:                                                     # trace_info : t_5416
        try:
            import transformer_engine.pytorch as te

            _te_version = packaging.version.Version(version("transformer-engine"))
            if _te_version < packaging.version.Version("1.5.0"):
                raise RuntimeError("use_te_rng_tracker requires TransformerEngine version >= 1.5")
        except:
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine, but not installed")
    if use_te_rng_tracker:                                                     # trace_info : t_5417
        _CUDA_RNG_STATE_TRACKER = te.distributed.CudaRNGStatesTracker()
    else:
        _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()                       # trace_info : t_5418
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True                                 # trace_info : t_5423


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    initialize_rng_tracker()                                                   # trace_info : t_6498, t_6909, t_7051, t_7380, t_7538, ...
    return _CUDA_RNG_STATE_TRACKER                                             # trace_info : t_6501, t_6912, t_7054, t_7383, t_7541, ...


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
    offset = seed + 2718                                                       # trace_info : t_5406
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()     # trace_info : t_5407
    # Data parallel gets the original seed.
    data_parallel_seed = seed                                                  # trace_info : t_5413

    initialize_rng_tracker()                                                   # trace_info : t_5414
    _CUDA_RNG_STATE_TRACKER.reset()                                            # trace_info : t_5424
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)                                 # trace_info : t_5428
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)# trace_info : t_5429

    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)# trace_info : t_5448

    expert_parallel_seed = (                                                   # trace_info : t_5486
        seed + 1024 + 100 * get_expert_model_parallel_rank() + get_tensor_model_parallel_rank()# trace_info : t_5467
    )
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)# trace_info : t_5487


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
