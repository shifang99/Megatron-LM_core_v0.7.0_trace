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
        self.ddp_config = ddp_config                                           # trace_info : t_10760

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params                                              # trace_info : t_10761
        self.params = set(params)                                              # trace_info : t_10762
        self.params_with_grad = set()                                          # trace_info : t_10763
        self.param_data = param_data                                           # trace_info : t_10764
        self.grad_data = grad_data                                             # trace_info : t_10765
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset                                                   # trace_info : t_10766
        self.numel_unpadded = numel_unpadded                                   # trace_info : t_10767
        self.data_parallel_group = data_parallel_group                         # trace_info : t_10768
        self.data_parallel_world_size = data_parallel_world_size               # trace_info : t_10769
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)# trace_info : t_10770
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_10771

        self.reset()                                                           # trace_info : t_10772

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()                                          # trace_info : t_10773, t_14657, t_18242, t_21881
        self.communication_handle = None                                       # trace_info : t_10774, t_14658, t_18243, t_21882
        self.communication_issued = False                                      # trace_info : t_10775, t_14659, t_18244, t_21883

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued# trace_info : t_16618, t_20257, t_23896
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:                              # trace_info : t_16619, t_20258, t_23897
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        if self.gradient_scaling_factor != 1.0:                                # trace_info : t_16620, t_20259, t_23898
            self.grad_data *= self.gradient_scaling_factor
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_16621, t_20260, t_23899
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
            self.communication_handle = torch.distributed.all_reduce(          # trace_info : t_16622, t_16626, t_20261, t_20265, t_23900, ...
                self.grad_data,                                                # trace_info : t_16623, t_20262, t_23901
                group=self.data_parallel_group,                                # trace_info : t_16624, t_20263, t_23902
                async_op=self.ddp_config.overlap_grad_reduce,                  # trace_info : t_16625, t_20264, t_23903
            )
        self.communication_issued = True                                       # trace_info : t_16627, t_20266, t_23905

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:                            # trace_info : t_16616, t_20255, t_23894
            self.start_grad_sync()                                             # trace_info : t_16617, t_20256, t_23895
            return                                                             # trace_info : t_16628, t_20267, t_23906
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
        self.ddp_config = ddp_config                                           # trace_info : t_9607

        # Check that params are unique.
        unique_params = set()                                                  # trace_info : t_9608
        for param in params:                                                   # trace_info : t_9609, t_9612, t_9615, t_9618, t_9621, ...
            assert param not in unique_params                                  # trace_info : t_9610, t_9613, t_9616, t_9619, t_9622, ...
            unique_params.add(param)                                           # trace_info : t_9611, t_9614, t_9617, t_9620, t_9623, ...
        del unique_params                                                      # trace_info : t_9694

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype                                         # trace_info : t_9695
        self.grad_dtype = grad_dtype                                           # trace_info : t_9696
        self.data_parallel_group = data_parallel_group                         # trace_info : t_9697
        self.data_parallel_world_size = torch.distributed.get_world_size(      # trace_info : t_9698, t_9700
            group=self.data_parallel_group                                     # trace_info : t_9699
        )
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_9701
        self.is_last_microbatch = True                                         # trace_info : t_9702

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []                                                      # trace_info : t_9703
        self.param_to_bucket = {}  # Param -> bucket mapping.                  # trace_info : t_9704
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).# trace_info : t_9705

        def _pad(number_to_be_padded: int, divisor: int) -> int:               # trace_info : t_9706
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:                            # trace_info : t_9707
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:                      # trace_info : t_10252, t_10726
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index                                                  # trace_info : t_10253, t_10727

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0                                                   # trace_info : t_9708
        bucket_data_start_index = data_start_index                             # trace_info : t_9709
        bucket_params = set()                                                  # trace_info : t_9710
        self.bucket_indices = []                                               # trace_info : t_9711
        per_bucket_numel_unpadded = []                                         # trace_info : t_9712
        bucket_id = 0                                                          # trace_info : t_9713

        def _create_new_bucket(data_end_index: int) -> int:                    # trace_info : t_9714
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)# trace_info : t_10250
            data_end_index = _pad_if_needed(data_end_index)                    # trace_info : t_10251
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))# trace_info : t_10254
            bucket_data_start_index = data_end_index                           # trace_info : t_10255
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()                                              # trace_info : t_10256
            bucket_id += 1                                                     # trace_info : t_10257
            # Return the potentially padded data_end_index.
            return data_end_index                                              # trace_info : t_10258

        for param in params[::-1]:                                             # trace_info : t_9715, t_9734, t_9753, t_9772, t_9791, ...
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:                                        # trace_info : t_9716, t_9735, t_9754, t_9773, t_9792, ...
                continue
            this_numel = param.data.nelement()                                 # trace_info : t_9717, t_9736, t_9755, t_9774, t_9793, ...
            data_end_index = data_start_index + this_numel                     # trace_info : t_9718, t_9737, t_9756, t_9775, t_9794, ...

            def _does_param_require_new_bucket(param):                         # trace_info : t_9719, t_9738, t_9757, t_9776, t_9795, ...
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (                                                       # trace_info : t_9722, t_9731, t_9741, t_9750, t_9760, ...
                    getattr(param, "shared_embedding", False)                  # trace_info : t_9721, t_9730, t_9740, t_9749, t_9759, ...
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:# trace_info : t_9720, t_9739, t_9758, t_9777, t_9796, ...
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (                                    # trace_info : t_9726, t_9745, t_9764, t_9783, t_9802, ...
                data_start_index,                                              # trace_info : t_9723, t_9742, t_9761, t_9780, t_9799, ...
                data_end_index,                                                # trace_info : t_9724, t_9743, t_9762, t_9781, t_9800, ...
                bucket_id,                                                     # trace_info : t_9725, t_9744, t_9763, t_9782, t_9801, ...
            )
            bucket_params.add(param)                                           # trace_info : t_9727, t_9746, t_9765, t_9784, t_9803, ...

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None                                        # trace_info : t_9728, t_9747, t_9766, t_9785, t_9804, ...
                and (data_end_index - bucket_data_start_index) >= bucket_size  # trace_info : t_9732, t_9751, t_9770, t_9789, t_9808, ...
            ) or _does_param_require_new_bucket(param):                        # trace_info : t_9729, t_9748, t_9767, t_9786, t_9805, ...
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index                                  # trace_info : t_9733, t_9752, t_9771, t_9790, t_9809, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_10248
            data_end_index = _create_new_bucket(data_end_index)                # trace_info : t_10249

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index                                            # trace_info : t_10259
        self.numel_unpadded = sum(per_bucket_numel_unpadded)                   # trace_info : t_10260
        assert self.numel_unpadded <= self.numel                               # trace_info : t_10261
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_10262
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded                           # trace_info : t_10263

        self.param_data = None                                                 # trace_info : t_10264
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_10265
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(                                          # trace_info : t_10266, t_10271
            self.numel,                                                        # trace_info : t_10267
            dtype=self.grad_dtype,                                             # trace_info : t_10268
            device=torch.cuda.current_device(),                                # trace_info : t_10269
            requires_grad=False,                                               # trace_info : t_10270
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = set()                                                  # trace_info : t_10272
        bucket_data_start_index = 0                                            # trace_info : t_10273
        cur_bucket_id = 0                                                      # trace_info : t_10274
        for param in params[::-1]:                                             # trace_info : t_10275, t_10291, t_10307, t_10323, t_10339, ...
            if not param.requires_grad:                                        # trace_info : t_10276, t_10292, t_10308, t_10324, t_10340, ...
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]# trace_info : t_10277, t_10293, t_10309, t_10325, t_10341, ...

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:                                    # trace_info : t_10278, t_10294, t_10310, t_10326, t_10342, ...
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

            param.main_grad = self._get(                                       # trace_info : t_10279, t_10281, t_10295, t_10297, t_10311, ...
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD# trace_info : t_10280, t_10296, t_10312, t_10328, t_10344, ...
            )
            if bucket_id != cur_bucket_id:                                     # trace_info : t_10289, t_10305, t_10321, t_10337, t_10353, ...
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
            bucket_params.add(param)                                           # trace_info : t_10290, t_10306, t_10322, t_10338, t_10354, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_10724
            bucket_data_end_index = _pad_if_needed(data_end_index)             # trace_info : t_10725
            self._set_bucket(                                                  # trace_info : t_10728, t_10734
                bucket_params=bucket_params,                                   # trace_info : t_10729
                start_index=bucket_data_start_index,                           # trace_info : t_10730
                end_index=bucket_data_end_index,                               # trace_info : t_10731
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],       # trace_info : t_10732
                bucket_id=cur_bucket_id,                                       # trace_info : t_10733
            )

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0# trace_info : t_10862
            and parallel_state.get_tensor_model_parallel_rank() == 0           # trace_info : t_10870
        ):
            logger.info(                                                       # trace_info : t_10876, t_10878
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'# trace_info : t_10877
            )
            for index, bucket in enumerate(self.buckets):                      # trace_info : t_10883, t_11116
                numel = 0                                                      # trace_info : t_10884
                for param in bucket.params:                                    # trace_info : t_10885, t_10887, t_10889, t_10891, t_10893, ...
                    numel += param.data.nelement()                             # trace_info : t_10886, t_10888, t_10890, t_10892, t_10894, ...
                logger.info(f'Params for bucket {index+1} ({numel} elements):')# trace_info : t_10942
                for param in bucket.params:                                    # trace_info : t_10947, t_10953, t_10959, t_10965, t_10971, ...
                    logger.info(f'    {param_to_name[param]}')                 # trace_info : t_10948, t_10954, t_10960, t_10966, t_10972, ...

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()                                # trace_info : t_10282, t_10298, t_10314, t_10330, t_10346, ...
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'# trace_info : t_10283, t_10299, t_10315, t_10331, t_10347, ...
        if buffer_type == BufferType.PARAM:                                    # trace_info : t_10284, t_10300, t_10316, t_10332, t_10348, ...
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:                                   # trace_info : t_10285, t_10301, t_10317, t_10333, t_10349, ...
            buffer_tensor = self.grad_data[start_index:end_index]              # trace_info : t_10286, t_10302, t_10318, t_10334, t_10350, ...
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)                              # trace_info : t_10287, t_10303, t_10319, t_10335, t_10351, ...
        return buffer_tensor                                                   # trace_info : t_10288, t_10304, t_10320, t_10336, t_10352, ...

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
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_10735
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]      # trace_info : t_10736

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None                                             # trace_info : t_10737
        if self.param_data is not None:                                        # trace_info : t_10738
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(                                        # trace_info : t_10739, t_10741
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD# trace_info : t_10740
        )
        bucket = Bucket(                                                       # trace_info : t_10749, t_10759
            ddp_config=self.ddp_config,                                        # trace_info : t_10750
            params=bucket_params,                                              # trace_info : t_10751
            param_data=bucketed_param_data,                                    # trace_info : t_10752
            grad_data=bucketed_grad_data,                                      # trace_info : t_10753
            offset=start_index,                                                # trace_info : t_10754
            numel_unpadded=numel_unpadded,                                     # trace_info : t_10755
            data_parallel_group=self.data_parallel_group,                      # trace_info : t_10756
            data_parallel_world_size=self.data_parallel_world_size,            # trace_info : t_10757
            gradient_scaling_factor=self.gradient_scaling_factor,              # trace_info : t_10758
        )
        self.buckets.append(bucket)                                            # trace_info : t_10776
        for bucket_param in bucket_params:                                     # trace_info : t_10777, t_10780, t_10783, t_10786, t_10789, ...
            assert bucket_param not in self.param_to_bucket                    # trace_info : t_10778, t_10781, t_10784, t_10787, t_10790, ...
            self.param_to_bucket[bucket_param] = bucket                        # trace_info : t_10779, t_10782, t_10785, t_10788, t_10791, ...

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()                                                 # trace_info : t_14654, t_18239, t_21878
        for bucket in self.buckets:                                            # trace_info : t_14655, t_14660, t_18240, t_18245, t_21879, ...
            bucket.reset()                                                     # trace_info : t_14656, t_18241, t_21880
        self.is_last_microbatch = True                                         # trace_info : t_14661, t_18246, t_21885

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
        for bucket in self.buckets:                                            # trace_info : t_16614, t_16629, t_20253, t_20268, t_23892, ...
            bucket.finish_grad_sync()                                          # trace_info : t_16615, t_20254, t_23893

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
