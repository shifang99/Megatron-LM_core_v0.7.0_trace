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
        self.ddp_config = ddp_config                                           # trace_info : t_13619

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params                                              # trace_info : t_13620
        self.params = set(params)                                              # trace_info : t_13621
        self.params_with_grad = set()                                          # trace_info : t_13622
        self.param_data = param_data                                           # trace_info : t_13623
        self.grad_data = grad_data                                             # trace_info : t_13624
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset                                                   # trace_info : t_13625
        self.numel_unpadded = numel_unpadded                                   # trace_info : t_13626
        self.data_parallel_group = data_parallel_group                         # trace_info : t_13627
        self.data_parallel_world_size = data_parallel_world_size               # trace_info : t_13628
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)# trace_info : t_13629
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_13630

        self.reset()                                                           # trace_info : t_13631

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()                                          # trace_info : t_13632, t_17515, t_20645, t_23831
        self.communication_handle = None                                       # trace_info : t_13633, t_17516, t_20646, t_23832
        self.communication_issued = False                                      # trace_info : t_13634, t_17517, t_20647, t_23833

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued# trace_info : t_19016, t_22202, t_25388
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:                              # trace_info : t_19017, t_22203, t_25389
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        if self.gradient_scaling_factor != 1.0:                                # trace_info : t_19018, t_22204, t_25390
            self.grad_data *= self.gradient_scaling_factor                     # trace_info : t_19019, t_22205, t_25391
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_19020, t_22206, t_25392
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
            self.communication_handle = torch.distributed.all_reduce(          # trace_info : t_19021, t_19025, t_22207, t_22211, t_25393, ...
                self.grad_data,                                                # trace_info : t_19022, t_22208, t_25394
                group=self.data_parallel_group,                                # trace_info : t_19023, t_22209, t_25395
                async_op=self.ddp_config.overlap_grad_reduce,                  # trace_info : t_19024, t_22210, t_25396
            )
        self.communication_issued = True                                       # trace_info : t_19026, t_22212, t_25398

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:                            # trace_info : t_19014, t_22200, t_25386
            self.start_grad_sync()                                             # trace_info : t_19015, t_22201, t_25387
            return                                                             # trace_info : t_19027, t_22213, t_25399
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
        self.ddp_config = ddp_config                                           # trace_info : t_12466

        # Check that params are unique.
        unique_params = set()                                                  # trace_info : t_12467
        for param in params:                                                   # trace_info : t_12468, t_12471, t_12474, t_12477, t_12480, ...
            assert param not in unique_params                                  # trace_info : t_12469, t_12472, t_12475, t_12478, t_12481, ...
            unique_params.add(param)                                           # trace_info : t_12470, t_12473, t_12476, t_12479, t_12482, ...
        del unique_params                                                      # trace_info : t_12553

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype                                         # trace_info : t_12554
        self.grad_dtype = grad_dtype                                           # trace_info : t_12555
        self.data_parallel_group = data_parallel_group                         # trace_info : t_12556
        self.data_parallel_world_size = torch.distributed.get_world_size(      # trace_info : t_12557, t_12559
            group=self.data_parallel_group                                     # trace_info : t_12558
        )
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_12560
        self.is_last_microbatch = True                                         # trace_info : t_12561

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []                                                      # trace_info : t_12562
        self.param_to_bucket = {}  # Param -> bucket mapping.                  # trace_info : t_12563
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).# trace_info : t_12564

        def _pad(number_to_be_padded: int, divisor: int) -> int:               # trace_info : t_12565
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:                            # trace_info : t_12566
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:                      # trace_info : t_13111, t_13585
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index                                                  # trace_info : t_13112, t_13586

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0                                                   # trace_info : t_12567
        bucket_data_start_index = data_start_index                             # trace_info : t_12568
        bucket_params = set()                                                  # trace_info : t_12569
        self.bucket_indices = []                                               # trace_info : t_12570
        per_bucket_numel_unpadded = []                                         # trace_info : t_12571
        bucket_id = 0                                                          # trace_info : t_12572

        def _create_new_bucket(data_end_index: int) -> int:                    # trace_info : t_12573
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)# trace_info : t_13109
            data_end_index = _pad_if_needed(data_end_index)                    # trace_info : t_13110
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))# trace_info : t_13113
            bucket_data_start_index = data_end_index                           # trace_info : t_13114
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()                                              # trace_info : t_13115
            bucket_id += 1                                                     # trace_info : t_13116
            # Return the potentially padded data_end_index.
            return data_end_index                                              # trace_info : t_13117

        for param in params[::-1]:                                             # trace_info : t_12574, t_12593, t_12612, t_12631, t_12650, ...
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:                                        # trace_info : t_12575, t_12594, t_12613, t_12632, t_12651, ...
                continue
            this_numel = param.data.nelement()                                 # trace_info : t_12576, t_12595, t_12614, t_12633, t_12652, ...
            data_end_index = data_start_index + this_numel                     # trace_info : t_12577, t_12596, t_12615, t_12634, t_12653, ...

            def _does_param_require_new_bucket(param):                         # trace_info : t_12578, t_12597, t_12616, t_12635, t_12654, ...
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (                                                       # trace_info : t_12581, t_12590, t_12600, t_12609, t_12619, ...
                    getattr(param, "shared_embedding", False)                  # trace_info : t_12580, t_12589, t_12599, t_12608, t_12618, ...
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:# trace_info : t_12579, t_12598, t_12617, t_12636, t_12655, ...
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (                                    # trace_info : t_12585, t_12604, t_12623, t_12642, t_12661, ...
                data_start_index,                                              # trace_info : t_12582, t_12601, t_12620, t_12639, t_12658, ...
                data_end_index,                                                # trace_info : t_12583, t_12602, t_12621, t_12640, t_12659, ...
                bucket_id,                                                     # trace_info : t_12584, t_12603, t_12622, t_12641, t_12660, ...
            )
            bucket_params.add(param)                                           # trace_info : t_12586, t_12605, t_12624, t_12643, t_12662, ...

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None                                        # trace_info : t_12587, t_12606, t_12625, t_12644, t_12663, ...
                and (data_end_index - bucket_data_start_index) >= bucket_size  # trace_info : t_12591, t_12610, t_12629, t_12648, t_12667, ...
            ) or _does_param_require_new_bucket(param):                        # trace_info : t_12588, t_12607, t_12626, t_12645, t_12664, ...
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index                                  # trace_info : t_12592, t_12611, t_12630, t_12649, t_12668, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13107
            data_end_index = _create_new_bucket(data_end_index)                # trace_info : t_13108

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index                                            # trace_info : t_13118
        self.numel_unpadded = sum(per_bucket_numel_unpadded)                   # trace_info : t_13119
        assert self.numel_unpadded <= self.numel                               # trace_info : t_13120
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13121
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded                           # trace_info : t_13122

        self.param_data = None                                                 # trace_info : t_13123
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13124
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(                                          # trace_info : t_13125, t_13130
            self.numel,                                                        # trace_info : t_13126
            dtype=self.grad_dtype,                                             # trace_info : t_13127
            device=torch.cuda.current_device(),                                # trace_info : t_13128
            requires_grad=False,                                               # trace_info : t_13129
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = set()                                                  # trace_info : t_13131
        bucket_data_start_index = 0                                            # trace_info : t_13132
        cur_bucket_id = 0                                                      # trace_info : t_13133
        for param in params[::-1]:                                             # trace_info : t_13134, t_13150, t_13166, t_13182, t_13198, ...
            if not param.requires_grad:                                        # trace_info : t_13135, t_13151, t_13167, t_13183, t_13199, ...
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]# trace_info : t_13136, t_13152, t_13168, t_13184, t_13200, ...

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:                                    # trace_info : t_13137, t_13153, t_13169, t_13185, t_13201, ...
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

            param.main_grad = self._get(                                       # trace_info : t_13138, t_13140, t_13154, t_13156, t_13170, ...
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD# trace_info : t_13139, t_13155, t_13171, t_13187, t_13203, ...
            )
            if bucket_id != cur_bucket_id:                                     # trace_info : t_13148, t_13164, t_13180, t_13196, t_13212, ...
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
            bucket_params.add(param)                                           # trace_info : t_13149, t_13165, t_13181, t_13197, t_13213, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13583
            bucket_data_end_index = _pad_if_needed(data_end_index)             # trace_info : t_13584
            self._set_bucket(                                                  # trace_info : t_13587, t_13593
                bucket_params=bucket_params,                                   # trace_info : t_13588
                start_index=bucket_data_start_index,                           # trace_info : t_13589
                end_index=bucket_data_end_index,                               # trace_info : t_13590
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],       # trace_info : t_13591
                bucket_id=cur_bucket_id,                                       # trace_info : t_13592
            )

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0# trace_info : t_13721
            and parallel_state.get_tensor_model_parallel_rank() == 0           # trace_info : t_13729
        ):
            logger.info(                                                       # trace_info : t_13735, t_13737
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'# trace_info : t_13736
            )
            for index, bucket in enumerate(self.buckets):                      # trace_info : t_13742, t_13975
                numel = 0                                                      # trace_info : t_13743
                for param in bucket.params:                                    # trace_info : t_13744, t_13746, t_13748, t_13750, t_13752, ...
                    numel += param.data.nelement()                             # trace_info : t_13745, t_13747, t_13749, t_13751, t_13753, ...
                logger.info(f'Params for bucket {index+1} ({numel} elements):')# trace_info : t_13801
                for param in bucket.params:                                    # trace_info : t_13806, t_13812, t_13818, t_13824, t_13830, ...
                    logger.info(f'    {param_to_name[param]}')                 # trace_info : t_13807, t_13813, t_13819, t_13825, t_13831, ...

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()                                # trace_info : t_13141, t_13157, t_13173, t_13189, t_13205, ...
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'# trace_info : t_13142, t_13158, t_13174, t_13190, t_13206, ...
        if buffer_type == BufferType.PARAM:                                    # trace_info : t_13143, t_13159, t_13175, t_13191, t_13207, ...
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:                                   # trace_info : t_13144, t_13160, t_13176, t_13192, t_13208, ...
            buffer_tensor = self.grad_data[start_index:end_index]              # trace_info : t_13145, t_13161, t_13177, t_13193, t_13209, ...
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)                              # trace_info : t_13146, t_13162, t_13178, t_13194, t_13210, ...
        return buffer_tensor                                                   # trace_info : t_13147, t_13163, t_13179, t_13195, t_13211, ...

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
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13594
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]      # trace_info : t_13595

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None                                             # trace_info : t_13596
        if self.param_data is not None:                                        # trace_info : t_13597
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(                                        # trace_info : t_13598, t_13600
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD# trace_info : t_13599
        )
        bucket = Bucket(                                                       # trace_info : t_13608, t_13618
            ddp_config=self.ddp_config,                                        # trace_info : t_13609
            params=bucket_params,                                              # trace_info : t_13610
            param_data=bucketed_param_data,                                    # trace_info : t_13611
            grad_data=bucketed_grad_data,                                      # trace_info : t_13612
            offset=start_index,                                                # trace_info : t_13613
            numel_unpadded=numel_unpadded,                                     # trace_info : t_13614
            data_parallel_group=self.data_parallel_group,                      # trace_info : t_13615
            data_parallel_world_size=self.data_parallel_world_size,            # trace_info : t_13616
            gradient_scaling_factor=self.gradient_scaling_factor,              # trace_info : t_13617
        )
        self.buckets.append(bucket)                                            # trace_info : t_13635
        for bucket_param in bucket_params:                                     # trace_info : t_13636, t_13639, t_13642, t_13645, t_13648, ...
            assert bucket_param not in self.param_to_bucket                    # trace_info : t_13637, t_13640, t_13643, t_13646, t_13649, ...
            self.param_to_bucket[bucket_param] = bucket                        # trace_info : t_13638, t_13641, t_13644, t_13647, t_13650, ...

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()                                                 # trace_info : t_17512, t_20642, t_23828
        for bucket in self.buckets:                                            # trace_info : t_17513, t_17518, t_20643, t_20648, t_23829, ...
            bucket.reset()                                                     # trace_info : t_17514, t_20644, t_23830
        self.is_last_microbatch = True                                         # trace_info : t_17519, t_20649, t_23835

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
        for bucket in self.buckets:                                            # trace_info : t_19012, t_19028, t_22198, t_22214, t_25384, ...
            bucket.finish_grad_sync()                                          # trace_info : t_19013, t_22199, t_25385

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
