# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
import os
from enum import Enum
from logging import getLogger
from typing import Dict, List, Optional

import torch

from .. import parallel_state
from .distributed_data_parallel_config import DistributedDataParallelConfig

logger = getLogger(__name__)


class BufferType(Enum):
    PARAM = 1
    GRAD = 2


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Args:
        ddp_config: DistributedDataParallel config object.
        params: List of parameters whose gradients are collated in this bucket.
        param_data: View in larger ParamAndGradBuffer.param_data that this bucket is responsible for.
        grad_data: View in larger ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        gradient_scaling_factor: float,
    ):
        self.ddp_config = ddp_config                                           # trace_info : t_13751, t_14559

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params                                              # trace_info : t_13752, t_14560
        self.params = set(params)                                              # trace_info : t_13753, t_14561
        self.params_with_grad = set()                                          # trace_info : t_13754, t_14562
        self.param_data = param_data                                           # trace_info : t_13755, t_14563
        self.grad_data = grad_data                                             # trace_info : t_13756, t_14564
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset                                                   # trace_info : t_13757, t_14565
        self.numel_unpadded = numel_unpadded                                   # trace_info : t_13758, t_14566
        self.data_parallel_group = data_parallel_group                         # trace_info : t_13759, t_14567
        self.data_parallel_world_size = data_parallel_world_size               # trace_info : t_13760, t_14568
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)# trace_info : t_13761, t_14569
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_13762, t_14570

        self.reset()                                                           # trace_info : t_13763, t_14571

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()                                          # trace_info : t_13764, t_14572, t_17730, t_17740, t_22023, ...
        self.communication_handle = None                                       # trace_info : t_13765, t_14573, t_17731, t_17741, t_22024, ...
        self.communication_issued = False                                      # trace_info : t_13766, t_14574, t_17732, t_17742, t_22025, ...

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued# trace_info : t_20198, t_20220, t_24543, t_24565, t_28888, ...
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:                              # trace_info : t_20199, t_20221, t_24544, t_24566, t_28889, ...
            global_rank = torch.distributed.get_rank()                         # trace_info : t_20200, t_20222, t_24545, t_24567, t_28890, ...
            norm = self.grad_data.norm(p=2)                                    # trace_info : t_20201, t_20223, t_24546, t_24568, t_28891, ...
            assert not norm.isnan(), (                                         # trace_info : t_20202, t_20224, t_24547, t_24569, t_28892, ...
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        if self.gradient_scaling_factor != 1.0:                                # trace_info : t_20203, t_20225, t_24548, t_24570, t_28893, ...
            self.grad_data *= self.gradient_scaling_factor                     # trace_info : t_20204, t_20226, t_24549, t_24571, t_28894, ...
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_20205, t_20227, t_24550, t_24572, t_28895, ...
            local_data_view = shard_buffer(self.grad_data, self.data_parallel_world_size)[
                self.data_parallel_rank
            ]
            self.communication_handle = torch.distributed._reduce_scatter_base(
                local_data_view,
                self.grad_data,
                group=self.data_parallel_group,
                async_op=self.ddp_config.overlap_grad_reduce,
            )
        else:
            self.communication_handle = torch.distributed.all_reduce(          # trace_info : t_20206, t_20210, t_20228, t_20232, t_24551, ...
                self.grad_data,                                                # trace_info : t_20207, t_20229, t_24552, t_24574, t_28897, ...
                group=self.data_parallel_group,                                # trace_info : t_20208, t_20230, t_24553, t_24575, t_28898, ...
                async_op=self.ddp_config.overlap_grad_reduce,                  # trace_info : t_20209, t_20231, t_24554, t_24576, t_28899, ...
            )
        self.communication_issued = True                                       # trace_info : t_20211, t_20233, t_24556, t_24578, t_28901, ...

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:                            # trace_info : t_20196, t_20218, t_24541, t_24563, t_28886, ...
            self.start_grad_sync()                                             # trace_info : t_20197, t_20219, t_24542, t_24564, t_28887, ...
            return                                                             # trace_info : t_20212, t_20234, t_24557, t_24579, t_28902, ...
        assert self.communication_handle is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()


class ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
    ):
        self.ddp_config = ddp_config                                           # trace_info : t_12826, t_14166

        # Check that params are unique.
        unique_params = set()                                                  # trace_info : t_12827, t_14167
        for param in params:                                                   # trace_info : t_12828, t_12831, t_12834, t_12837, t_12840, ...
            assert param not in unique_params                                  # trace_info : t_12829, t_12832, t_12835, t_12838, t_12841, ...
            unique_params.add(param)                                           # trace_info : t_12830, t_12833, t_12836, t_12839, t_12842, ...
        del unique_params                                                      # trace_info : t_12895, t_14193

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype                                         # trace_info : t_12896, t_14194
        self.grad_dtype = grad_dtype                                           # trace_info : t_12897, t_14195
        self.data_parallel_group = data_parallel_group                         # trace_info : t_12898, t_14196
        self.data_parallel_world_size = torch.distributed.get_world_size(      # trace_info : t_12899, t_12901, t_14197, t_14199
            group=self.data_parallel_group                                     # trace_info : t_12900, t_14198
        )
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_12902, t_14200
        self.is_last_microbatch = True                                         # trace_info : t_12903, t_14201

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []                                                      # trace_info : t_12904, t_14202
        self.param_to_bucket = {}  # Param -> bucket mapping.                  # trace_info : t_12905, t_14203
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).# trace_info : t_12906, t_14204

        def _pad(number_to_be_padded: int, divisor: int) -> int:               # trace_info : t_12907, t_14205
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:                            # trace_info : t_12908, t_14206
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:                      # trace_info : t_13339, t_13717, t_14371, t_14525
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index                                                  # trace_info : t_13340, t_13718, t_14372, t_14526

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0                                                   # trace_info : t_12909, t_14207
        bucket_data_start_index = data_start_index                             # trace_info : t_12910, t_14208
        bucket_params = set()                                                  # trace_info : t_12911, t_14209
        self.bucket_indices = []                                               # trace_info : t_12912, t_14210
        per_bucket_numel_unpadded = []                                         # trace_info : t_12913, t_14211
        bucket_id = 0                                                          # trace_info : t_12914, t_14212

        def _create_new_bucket(data_end_index: int) -> int:                    # trace_info : t_12915, t_14213
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)# trace_info : t_13337, t_14369
            data_end_index = _pad_if_needed(data_end_index)                    # trace_info : t_13338, t_14370
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))# trace_info : t_13341, t_14373
            bucket_data_start_index = data_end_index                           # trace_info : t_13342, t_14374
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()                                              # trace_info : t_13343, t_14375
            bucket_id += 1                                                     # trace_info : t_13344, t_14376
            # Return the potentially padded data_end_index.
            return data_end_index                                              # trace_info : t_13345, t_14377

        for param in params[::-1]:                                             # trace_info : t_12916, t_12935, t_12954, t_12973, t_12992, ...
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:                                        # trace_info : t_12917, t_12936, t_12955, t_12974, t_12993, ...
                continue
            this_numel = param.data.nelement()                                 # trace_info : t_12918, t_12937, t_12956, t_12975, t_12994, ...
            data_end_index = data_start_index + this_numel                     # trace_info : t_12919, t_12938, t_12957, t_12976, t_12995, ...

            def _does_param_require_new_bucket(param):                         # trace_info : t_12920, t_12939, t_12958, t_12977, t_12996, ...
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (                                                       # trace_info : t_12923, t_12932, t_12942, t_12951, t_12961, ...
                    getattr(param, "shared_embedding", False)                  # trace_info : t_12922, t_12931, t_12941, t_12950, t_12960, ...
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:# trace_info : t_12921, t_12940, t_12959, t_12978, t_12997, ...
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (                                    # trace_info : t_12927, t_12946, t_12965, t_12984, t_13003, ...
                data_start_index,                                              # trace_info : t_12924, t_12943, t_12962, t_12981, t_13000, ...
                data_end_index,                                                # trace_info : t_12925, t_12944, t_12963, t_12982, t_13001, ...
                bucket_id,                                                     # trace_info : t_12926, t_12945, t_12964, t_12983, t_13002, ...
            )
            bucket_params.add(param)                                           # trace_info : t_12928, t_12947, t_12966, t_12985, t_13004, ...

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None                                        # trace_info : t_12929, t_12948, t_12967, t_12986, t_13005, ...
                and (data_end_index - bucket_data_start_index) >= bucket_size  # trace_info : t_12933, t_12952, t_12971, t_12990, t_13009, ...
            ) or _does_param_require_new_bucket(param):                        # trace_info : t_12930, t_12949, t_12968, t_12987, t_13006, ...
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index                                  # trace_info : t_12934, t_12953, t_12972, t_12991, t_13010, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13335, t_14367
            data_end_index = _create_new_bucket(data_end_index)                # trace_info : t_13336, t_14368

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index                                            # trace_info : t_13346, t_14378
        self.numel_unpadded = sum(per_bucket_numel_unpadded)                   # trace_info : t_13347, t_14379
        assert self.numel_unpadded <= self.numel                               # trace_info : t_13348, t_14380
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13349, t_14381
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded                           # trace_info : t_13350, t_14382

        self.param_data = None                                                 # trace_info : t_13351, t_14383
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13352, t_14384
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(                                          # trace_info : t_13353, t_13358, t_14385, t_14390
            self.numel,                                                        # trace_info : t_13354, t_14386
            dtype=self.grad_dtype,                                             # trace_info : t_13355, t_14387
            device=torch.cuda.current_device(),                                # trace_info : t_13356, t_14388
            requires_grad=False,                                               # trace_info : t_13357, t_14389
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = set()                                                  # trace_info : t_13359, t_14391
        bucket_data_start_index = 0                                            # trace_info : t_13360, t_14392
        cur_bucket_id = 0                                                      # trace_info : t_13361, t_14393
        for param in params[::-1]:                                             # trace_info : t_13362, t_13378, t_13394, t_13410, t_13426, ...
            if not param.requires_grad:                                        # trace_info : t_13363, t_13379, t_13395, t_13411, t_13427, ...
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]# trace_info : t_13364, t_13380, t_13396, t_13412, t_13428, ...

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:                                    # trace_info : t_13365, t_13381, t_13397, t_13413, t_13429, ...
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

            param.main_grad = self._get(                                       # trace_info : t_13366, t_13368, t_13382, t_13384, t_13398, ...
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD# trace_info : t_13367, t_13383, t_13399, t_13415, t_13431, ...
            )
            if bucket_id != cur_bucket_id:                                     # trace_info : t_13376, t_13392, t_13408, t_13424, t_13440, ...
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_data_start_index,
                    end_index=bucket_data_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = set()
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.add(param)                                           # trace_info : t_13377, t_13393, t_13409, t_13425, t_13441, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13715, t_14523
            bucket_data_end_index = _pad_if_needed(data_end_index)             # trace_info : t_13716, t_14524
            self._set_bucket(                                                  # trace_info : t_13719, t_13725, t_14527, t_14533
                bucket_params=bucket_params,                                   # trace_info : t_13720, t_14528
                start_index=bucket_data_start_index,                           # trace_info : t_13721, t_14529
                end_index=bucket_data_end_index,                               # trace_info : t_13722, t_14530
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],       # trace_info : t_13723, t_14531
                bucket_id=cur_bucket_id,                                       # trace_info : t_13724, t_14532
            )

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0# trace_info : t_13835, t_14601
            and parallel_state.get_tensor_model_parallel_rank() == 0           # trace_info : t_13843, t_14609
        ):
            logger.info(                                                       # trace_info : t_13849, t_13851, t_14615, t_14617
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'# trace_info : t_13850, t_14616
            )
            for index, bucket in enumerate(self.buckets):                      # trace_info : t_13856, t_14041, t_14622, t_14695
                numel = 0                                                      # trace_info : t_13857, t_14623
                for param in bucket.params:                                    # trace_info : t_13858, t_13860, t_13862, t_13864, t_13866, ...
                    numel += param.data.nelement()                             # trace_info : t_13859, t_13861, t_13863, t_13865, t_13867, ...
                logger.info(f'Params for bucket {index+1} ({numel} elements):')# trace_info : t_13903, t_14641
                for param in bucket.params:                                    # trace_info : t_13908, t_13914, t_13920, t_13926, t_13932, ...
                    logger.info(f'    {param_to_name[param]}')                 # trace_info : t_13909, t_13915, t_13921, t_13927, t_13933, ...

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()                                # trace_info : t_13369, t_13385, t_13401, t_13417, t_13433, ...
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'# trace_info : t_13370, t_13386, t_13402, t_13418, t_13434, ...
        if buffer_type == BufferType.PARAM:                                    # trace_info : t_13371, t_13387, t_13403, t_13419, t_13435, ...
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:                                   # trace_info : t_13372, t_13388, t_13404, t_13420, t_13436, ...
            buffer_tensor = self.grad_data[start_index:end_index]              # trace_info : t_13373, t_13389, t_13405, t_13421, t_13437, ...
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)                              # trace_info : t_13374, t_13390, t_13406, t_13422, t_13438, ...
        return buffer_tensor                                                   # trace_info : t_13375, t_13391, t_13407, t_13423, t_13439, ...

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13726, t_14534
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]      # trace_info : t_13727, t_14535

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None                                             # trace_info : t_13728, t_14536
        if self.param_data is not None:                                        # trace_info : t_13729, t_14537
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(                                        # trace_info : t_13730, t_13732, t_14538, t_14540
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD# trace_info : t_13731, t_14539
        )
        bucket = Bucket(                                                       # trace_info : t_13740, t_13750, t_14548, t_14558
            ddp_config=self.ddp_config,                                        # trace_info : t_13741, t_14549
            params=bucket_params,                                              # trace_info : t_13742, t_14550
            param_data=bucketed_param_data,                                    # trace_info : t_13743, t_14551
            grad_data=bucketed_grad_data,                                      # trace_info : t_13744, t_14552
            offset=start_index,                                                # trace_info : t_13745, t_14553
            numel_unpadded=numel_unpadded,                                     # trace_info : t_13746, t_14554
            data_parallel_group=self.data_parallel_group,                      # trace_info : t_13747, t_14555
            data_parallel_world_size=self.data_parallel_world_size,            # trace_info : t_13748, t_14556
            gradient_scaling_factor=self.gradient_scaling_factor,              # trace_info : t_13749, t_14557
        )
        self.buckets.append(bucket)                                            # trace_info : t_13767, t_14575
        for bucket_param in bucket_params:                                     # trace_info : t_13768, t_13771, t_13774, t_13777, t_13780, ...
            assert bucket_param not in self.param_to_bucket                    # trace_info : t_13769, t_13772, t_13775, t_13778, t_13781, ...
            self.param_to_bucket[bucket_param] = bucket                        # trace_info : t_13770, t_13773, t_13776, t_13779, t_13782, ...

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()                                                 # trace_info : t_17727, t_17737, t_22020, t_22030, t_26365, ...
        for bucket in self.buckets:                                            # trace_info : t_17728, t_17733, t_17738, t_17743, t_22021, ...
            bucket.reset()                                                     # trace_info : t_17729, t_17739, t_22022, t_22032, t_26367, ...
        self.is_last_microbatch = True                                         # trace_info : t_17734, t_17744, t_22027, t_22037, t_26372, ...

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:                                            # trace_info : t_20194, t_20213, t_20216, t_20235, t_24539, ...
            bucket.finish_grad_sync()                                          # trace_info : t_20195, t_20217, t_24540, t_24562, t_28885, ...

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)
