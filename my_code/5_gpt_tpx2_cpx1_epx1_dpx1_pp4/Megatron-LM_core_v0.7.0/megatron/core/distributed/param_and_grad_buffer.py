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
        self.ddp_config = ddp_config                                           # trace_info : t_13948

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params                                              # trace_info : t_13949
        self.params = set(params)                                              # trace_info : t_13950
        self.params_with_grad = set()                                          # trace_info : t_13951
        self.param_data = param_data                                           # trace_info : t_13952
        self.grad_data = grad_data                                             # trace_info : t_13953
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset                                                   # trace_info : t_13954
        self.numel_unpadded = numel_unpadded                                   # trace_info : t_13955
        self.data_parallel_group = data_parallel_group                         # trace_info : t_13956
        self.data_parallel_world_size = data_parallel_world_size               # trace_info : t_13957
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)# trace_info : t_13958
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_13959

        self.reset()                                                           # trace_info : t_13960

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()                                          # trace_info : t_13961, t_17716, t_21394, t_25122
        self.communication_handle = None                                       # trace_info : t_13962, t_17717, t_21395, t_25123
        self.communication_issued = False                                      # trace_info : t_13963, t_17718, t_21396, t_25124

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued# trace_info : t_19830, t_23558, t_27286
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:                              # trace_info : t_19831, t_23559, t_27287
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        if self.gradient_scaling_factor != 1.0:                                # trace_info : t_19832, t_23560, t_27288
            self.grad_data *= self.gradient_scaling_factor
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_19833, t_23561, t_27289
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
            self.communication_handle = torch.distributed.all_reduce(          # trace_info : t_19834, t_19838, t_23562, t_23566, t_27290, ...
                self.grad_data,                                                # trace_info : t_19835, t_23563, t_27291
                group=self.data_parallel_group,                                # trace_info : t_19836, t_23564, t_27292
                async_op=self.ddp_config.overlap_grad_reduce,                  # trace_info : t_19837, t_23565, t_27293
            )
        self.communication_issued = True                                       # trace_info : t_19839, t_23567, t_27295

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:                            # trace_info : t_19828, t_23556, t_27284
            self.start_grad_sync()                                             # trace_info : t_19829, t_23557, t_27285
            return                                                             # trace_info : t_19840, t_23568, t_27296
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
        self.ddp_config = ddp_config                                           # trace_info : t_12869

        # Check that params are unique.
        unique_params = set()                                                  # trace_info : t_12870
        for param in params:                                                   # trace_info : t_12871, t_12874, t_12877, t_12880, t_12883, ...
            assert param not in unique_params                                  # trace_info : t_12872, t_12875, t_12878, t_12881, t_12884, ...
            unique_params.add(param)                                           # trace_info : t_12873, t_12876, t_12879, t_12882, t_12885, ...
        del unique_params                                                      # trace_info : t_12950

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype                                         # trace_info : t_12951
        self.grad_dtype = grad_dtype                                           # trace_info : t_12952
        self.data_parallel_group = data_parallel_group                         # trace_info : t_12953
        self.data_parallel_world_size = torch.distributed.get_world_size(      # trace_info : t_12954, t_12956
            group=self.data_parallel_group                                     # trace_info : t_12955
        )
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_12957
        self.is_last_microbatch = True                                         # trace_info : t_12958

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []                                                      # trace_info : t_12959
        self.param_to_bucket = {}  # Param -> bucket mapping.                  # trace_info : t_12960
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).# trace_info : t_12961

        def _pad(number_to_be_padded: int, divisor: int) -> int:               # trace_info : t_12962
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:                            # trace_info : t_12963
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:                      # trace_info : t_13472, t_13914
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index                                                  # trace_info : t_13473, t_13915

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0                                                   # trace_info : t_12964
        bucket_data_start_index = data_start_index                             # trace_info : t_12965
        bucket_params = set()                                                  # trace_info : t_12966
        self.bucket_indices = []                                               # trace_info : t_12967
        per_bucket_numel_unpadded = []                                         # trace_info : t_12968
        bucket_id = 0                                                          # trace_info : t_12969

        def _create_new_bucket(data_end_index: int) -> int:                    # trace_info : t_12970
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)# trace_info : t_13470
            data_end_index = _pad_if_needed(data_end_index)                    # trace_info : t_13471
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))# trace_info : t_13474
            bucket_data_start_index = data_end_index                           # trace_info : t_13475
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()                                              # trace_info : t_13476
            bucket_id += 1                                                     # trace_info : t_13477
            # Return the potentially padded data_end_index.
            return data_end_index                                              # trace_info : t_13478

        for param in params[::-1]:                                             # trace_info : t_12971, t_12990, t_13009, t_13028, t_13047, ...
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:                                        # trace_info : t_12972, t_12991, t_13010, t_13029, t_13048, ...
                continue
            this_numel = param.data.nelement()                                 # trace_info : t_12973, t_12992, t_13011, t_13030, t_13049, ...
            data_end_index = data_start_index + this_numel                     # trace_info : t_12974, t_12993, t_13012, t_13031, t_13050, ...

            def _does_param_require_new_bucket(param):                         # trace_info : t_12975, t_12994, t_13013, t_13032, t_13051, ...
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (                                                       # trace_info : t_12978, t_12987, t_12997, t_13006, t_13016, ...
                    getattr(param, "shared_embedding", False)                  # trace_info : t_12977, t_12986, t_12996, t_13005, t_13015, ...
                    and self.ddp_config.use_distributed_optimizer              # trace_info : t_13453, t_13463
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:# trace_info : t_12976, t_12995, t_13014, t_13033, t_13052, ...
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (                                    # trace_info : t_12982, t_13001, t_13020, t_13039, t_13058, ...
                data_start_index,                                              # trace_info : t_12979, t_12998, t_13017, t_13036, t_13055, ...
                data_end_index,                                                # trace_info : t_12980, t_12999, t_13018, t_13037, t_13056, ...
                bucket_id,                                                     # trace_info : t_12981, t_13000, t_13019, t_13038, t_13057, ...
            )
            bucket_params.add(param)                                           # trace_info : t_12983, t_13002, t_13021, t_13040, t_13059, ...

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None                                        # trace_info : t_12984, t_13003, t_13022, t_13041, t_13060, ...
                and (data_end_index - bucket_data_start_index) >= bucket_size  # trace_info : t_12988, t_13007, t_13026, t_13045, t_13064, ...
            ) or _does_param_require_new_bucket(param):                        # trace_info : t_12985, t_13004, t_13023, t_13042, t_13061, ...
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index                                  # trace_info : t_12989, t_13008, t_13027, t_13046, t_13065, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13468
            data_end_index = _create_new_bucket(data_end_index)                # trace_info : t_13469

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index                                            # trace_info : t_13479
        self.numel_unpadded = sum(per_bucket_numel_unpadded)                   # trace_info : t_13480
        assert self.numel_unpadded <= self.numel                               # trace_info : t_13481
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13482
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded                           # trace_info : t_13483

        self.param_data = None                                                 # trace_info : t_13484
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13485
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(                                          # trace_info : t_13486, t_13491
            self.numel,                                                        # trace_info : t_13487
            dtype=self.grad_dtype,                                             # trace_info : t_13488
            device=torch.cuda.current_device(),                                # trace_info : t_13489
            requires_grad=False,                                               # trace_info : t_13490
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = set()                                                  # trace_info : t_13492
        bucket_data_start_index = 0                                            # trace_info : t_13493
        cur_bucket_id = 0                                                      # trace_info : t_13494
        for param in params[::-1]:                                             # trace_info : t_13495, t_13511, t_13527, t_13543, t_13559, ...
            if not param.requires_grad:                                        # trace_info : t_13496, t_13512, t_13528, t_13544, t_13560, ...
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]# trace_info : t_13497, t_13513, t_13529, t_13545, t_13561, ...

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:                                    # trace_info : t_13498, t_13514, t_13530, t_13546, t_13562, ...
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

            param.main_grad = self._get(                                       # trace_info : t_13499, t_13501, t_13515, t_13517, t_13531, ...
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD# trace_info : t_13500, t_13516, t_13532, t_13548, t_13564, ...
            )
            if bucket_id != cur_bucket_id:                                     # trace_info : t_13509, t_13525, t_13541, t_13557, t_13573, ...
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
            bucket_params.add(param)                                           # trace_info : t_13510, t_13526, t_13542, t_13558, t_13574, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13912
            bucket_data_end_index = _pad_if_needed(data_end_index)             # trace_info : t_13913
            self._set_bucket(                                                  # trace_info : t_13916, t_13922
                bucket_params=bucket_params,                                   # trace_info : t_13917
                start_index=bucket_data_start_index,                           # trace_info : t_13918
                end_index=bucket_data_end_index,                               # trace_info : t_13919
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],       # trace_info : t_13920
                bucket_id=cur_bucket_id,                                       # trace_info : t_13921
            )

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0# trace_info : t_14044
            and parallel_state.get_tensor_model_parallel_rank() == 0           # trace_info : t_14052
        ):
            logger.info(                                                       # trace_info : t_14058, t_14060
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'# trace_info : t_14059
            )
            for index, bucket in enumerate(self.buckets):                      # trace_info : t_14065, t_14282
                numel = 0                                                      # trace_info : t_14066
                for param in bucket.params:                                    # trace_info : t_14067, t_14069, t_14071, t_14073, t_14075, ...
                    numel += param.data.nelement()                             # trace_info : t_14068, t_14070, t_14072, t_14074, t_14076, ...
                logger.info(f'Params for bucket {index+1} ({numel} elements):')# trace_info : t_14120
                for param in bucket.params:                                    # trace_info : t_14125, t_14131, t_14137, t_14143, t_14149, ...
                    logger.info(f'    {param_to_name[param]}')                 # trace_info : t_14126, t_14132, t_14138, t_14144, t_14150, ...

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()                                # trace_info : t_13502, t_13518, t_13534, t_13550, t_13566, ...
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'# trace_info : t_13503, t_13519, t_13535, t_13551, t_13567, ...
        if buffer_type == BufferType.PARAM:                                    # trace_info : t_13504, t_13520, t_13536, t_13552, t_13568, ...
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:                                   # trace_info : t_13505, t_13521, t_13537, t_13553, t_13569, ...
            buffer_tensor = self.grad_data[start_index:end_index]              # trace_info : t_13506, t_13522, t_13538, t_13554, t_13570, ...
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)                              # trace_info : t_13507, t_13523, t_13539, t_13555, t_13571, ...
        return buffer_tensor                                                   # trace_info : t_13508, t_13524, t_13540, t_13556, t_13572, ...

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
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13923
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]      # trace_info : t_13924

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None                                             # trace_info : t_13925
        if self.param_data is not None:                                        # trace_info : t_13926
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(                                        # trace_info : t_13927, t_13929
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD# trace_info : t_13928
        )
        bucket = Bucket(                                                       # trace_info : t_13937, t_13947
            ddp_config=self.ddp_config,                                        # trace_info : t_13938
            params=bucket_params,                                              # trace_info : t_13939
            param_data=bucketed_param_data,                                    # trace_info : t_13940
            grad_data=bucketed_grad_data,                                      # trace_info : t_13941
            offset=start_index,                                                # trace_info : t_13942
            numel_unpadded=numel_unpadded,                                     # trace_info : t_13943
            data_parallel_group=self.data_parallel_group,                      # trace_info : t_13944
            data_parallel_world_size=self.data_parallel_world_size,            # trace_info : t_13945
            gradient_scaling_factor=self.gradient_scaling_factor,              # trace_info : t_13946
        )
        self.buckets.append(bucket)                                            # trace_info : t_13964
        for bucket_param in bucket_params:                                     # trace_info : t_13965, t_13968, t_13971, t_13974, t_13977, ...
            assert bucket_param not in self.param_to_bucket                    # trace_info : t_13966, t_13969, t_13972, t_13975, t_13978, ...
            self.param_to_bucket[bucket_param] = bucket                        # trace_info : t_13967, t_13970, t_13973, t_13976, t_13979, ...

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()                                                 # trace_info : t_17713, t_21391, t_25119
        for bucket in self.buckets:                                            # trace_info : t_17714, t_17719, t_21392, t_21397, t_25120, ...
            bucket.reset()                                                     # trace_info : t_17715, t_21393, t_25121
        self.is_last_microbatch = True                                         # trace_info : t_17720, t_21398, t_25126

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
        for bucket in self.buckets:                                            # trace_info : t_19826, t_19841, t_23554, t_23569, t_27282, ...
            bucket.finish_grad_sync()                                          # trace_info : t_19827, t_23555, t_27283

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
