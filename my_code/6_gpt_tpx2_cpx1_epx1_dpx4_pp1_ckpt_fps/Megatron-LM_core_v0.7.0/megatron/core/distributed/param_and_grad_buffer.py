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
        self.ddp_config = ddp_config                                           # trace_info : t_13789

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params                                              # trace_info : t_13790
        self.params = set(params)                                              # trace_info : t_13791
        self.params_with_grad = set()                                          # trace_info : t_13792
        self.param_data = param_data                                           # trace_info : t_13793
        self.grad_data = grad_data                                             # trace_info : t_13794
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset                                                   # trace_info : t_13795
        self.numel_unpadded = numel_unpadded                                   # trace_info : t_13796
        self.data_parallel_group = data_parallel_group                         # trace_info : t_13797
        self.data_parallel_world_size = data_parallel_world_size               # trace_info : t_13798
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)# trace_info : t_13799
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_13800

        self.reset()                                                           # trace_info : t_13801

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()                                          # trace_info : t_13802, t_17793, t_21376, t_88983
        self.communication_handle = None                                       # trace_info : t_13803, t_17794, t_21377, t_88984
        self.communication_issued = False                                      # trace_info : t_13804, t_17795, t_21378, t_88985

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued# trace_info : t_19746, t_23383, t_90990
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:                              # trace_info : t_19747, t_23384, t_90991
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        if self.gradient_scaling_factor != 1.0:                                # trace_info : t_19748, t_23385, t_90992
            self.grad_data *= self.gradient_scaling_factor                     # trace_info : t_19749, t_23386, t_90993
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_19750, t_23387, t_90994
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
            self.communication_handle = torch.distributed.all_reduce(          # trace_info : t_19751, t_19755, t_23388, t_23392, t_90995, ...
                self.grad_data,                                                # trace_info : t_19752, t_23389, t_90996
                group=self.data_parallel_group,                                # trace_info : t_19753, t_23390, t_90997
                async_op=self.ddp_config.overlap_grad_reduce,                  # trace_info : t_19754, t_23391, t_90998
            )
        self.communication_issued = True                                       # trace_info : t_19756, t_23393, t_91000

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:                            # trace_info : t_19744, t_23381, t_90988
            self.start_grad_sync()                                             # trace_info : t_19745, t_23382, t_90989
            return                                                             # trace_info : t_19757, t_23394, t_91001
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
        self.ddp_config = ddp_config                                           # trace_info : t_12636

        # Check that params are unique.
        unique_params = set()                                                  # trace_info : t_12637
        for param in params:                                                   # trace_info : t_12638, t_12641, t_12644, t_12647, t_12650, ...
            assert param not in unique_params                                  # trace_info : t_12639, t_12642, t_12645, t_12648, t_12651, ...
            unique_params.add(param)                                           # trace_info : t_12640, t_12643, t_12646, t_12649, t_12652, ...
        del unique_params                                                      # trace_info : t_12723

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype                                         # trace_info : t_12724
        self.grad_dtype = grad_dtype                                           # trace_info : t_12725
        self.data_parallel_group = data_parallel_group                         # trace_info : t_12726
        self.data_parallel_world_size = torch.distributed.get_world_size(      # trace_info : t_12727, t_12729
            group=self.data_parallel_group                                     # trace_info : t_12728
        )
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_12730
        self.is_last_microbatch = True                                         # trace_info : t_12731

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []                                                      # trace_info : t_12732
        self.param_to_bucket = {}  # Param -> bucket mapping.                  # trace_info : t_12733
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).# trace_info : t_12734

        def _pad(number_to_be_padded: int, divisor: int) -> int:               # trace_info : t_12735
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:                            # trace_info : t_12736
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:                      # trace_info : t_13281, t_13755
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index                                                  # trace_info : t_13282, t_13756

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0                                                   # trace_info : t_12737
        bucket_data_start_index = data_start_index                             # trace_info : t_12738
        bucket_params = set()                                                  # trace_info : t_12739
        self.bucket_indices = []                                               # trace_info : t_12740
        per_bucket_numel_unpadded = []                                         # trace_info : t_12741
        bucket_id = 0                                                          # trace_info : t_12742

        def _create_new_bucket(data_end_index: int) -> int:                    # trace_info : t_12743
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)# trace_info : t_13279
            data_end_index = _pad_if_needed(data_end_index)                    # trace_info : t_13280
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))# trace_info : t_13283
            bucket_data_start_index = data_end_index                           # trace_info : t_13284
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()                                              # trace_info : t_13285
            bucket_id += 1                                                     # trace_info : t_13286
            # Return the potentially padded data_end_index.
            return data_end_index                                              # trace_info : t_13287

        for param in params[::-1]:                                             # trace_info : t_12744, t_12763, t_12782, t_12801, t_12820, ...
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:                                        # trace_info : t_12745, t_12764, t_12783, t_12802, t_12821, ...
                continue
            this_numel = param.data.nelement()                                 # trace_info : t_12746, t_12765, t_12784, t_12803, t_12822, ...
            data_end_index = data_start_index + this_numel                     # trace_info : t_12747, t_12766, t_12785, t_12804, t_12823, ...

            def _does_param_require_new_bucket(param):                         # trace_info : t_12748, t_12767, t_12786, t_12805, t_12824, ...
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (                                                       # trace_info : t_12751, t_12760, t_12770, t_12779, t_12789, ...
                    getattr(param, "shared_embedding", False)                  # trace_info : t_12750, t_12759, t_12769, t_12778, t_12788, ...
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:# trace_info : t_12749, t_12768, t_12787, t_12806, t_12825, ...
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (                                    # trace_info : t_12755, t_12774, t_12793, t_12812, t_12831, ...
                data_start_index,                                              # trace_info : t_12752, t_12771, t_12790, t_12809, t_12828, ...
                data_end_index,                                                # trace_info : t_12753, t_12772, t_12791, t_12810, t_12829, ...
                bucket_id,                                                     # trace_info : t_12754, t_12773, t_12792, t_12811, t_12830, ...
            )
            bucket_params.add(param)                                           # trace_info : t_12756, t_12775, t_12794, t_12813, t_12832, ...

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None                                        # trace_info : t_12757, t_12776, t_12795, t_12814, t_12833, ...
                and (data_end_index - bucket_data_start_index) >= bucket_size  # trace_info : t_12761, t_12780, t_12799, t_12818, t_12837, ...
            ) or _does_param_require_new_bucket(param):                        # trace_info : t_12758, t_12777, t_12796, t_12815, t_12834, ...
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index                                  # trace_info : t_12762, t_12781, t_12800, t_12819, t_12838, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13277
            data_end_index = _create_new_bucket(data_end_index)                # trace_info : t_13278

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index                                            # trace_info : t_13288
        self.numel_unpadded = sum(per_bucket_numel_unpadded)                   # trace_info : t_13289
        assert self.numel_unpadded <= self.numel                               # trace_info : t_13290
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13291
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded                           # trace_info : t_13292

        self.param_data = None                                                 # trace_info : t_13293
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13294
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(                                          # trace_info : t_13295, t_13300
            self.numel,                                                        # trace_info : t_13296
            dtype=self.grad_dtype,                                             # trace_info : t_13297
            device=torch.cuda.current_device(),                                # trace_info : t_13298
            requires_grad=False,                                               # trace_info : t_13299
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = set()                                                  # trace_info : t_13301
        bucket_data_start_index = 0                                            # trace_info : t_13302
        cur_bucket_id = 0                                                      # trace_info : t_13303
        for param in params[::-1]:                                             # trace_info : t_13304, t_13320, t_13336, t_13352, t_13368, ...
            if not param.requires_grad:                                        # trace_info : t_13305, t_13321, t_13337, t_13353, t_13369, ...
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]# trace_info : t_13306, t_13322, t_13338, t_13354, t_13370, ...

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:                                    # trace_info : t_13307, t_13323, t_13339, t_13355, t_13371, ...
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

            param.main_grad = self._get(                                       # trace_info : t_13308, t_13310, t_13324, t_13326, t_13340, ...
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD# trace_info : t_13309, t_13325, t_13341, t_13357, t_13373, ...
            )
            if bucket_id != cur_bucket_id:                                     # trace_info : t_13318, t_13334, t_13350, t_13366, t_13382, ...
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
            bucket_params.add(param)                                           # trace_info : t_13319, t_13335, t_13351, t_13367, t_13383, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_13753
            bucket_data_end_index = _pad_if_needed(data_end_index)             # trace_info : t_13754
            self._set_bucket(                                                  # trace_info : t_13757, t_13763
                bucket_params=bucket_params,                                   # trace_info : t_13758
                start_index=bucket_data_start_index,                           # trace_info : t_13759
                end_index=bucket_data_end_index,                               # trace_info : t_13760
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],       # trace_info : t_13761
                bucket_id=cur_bucket_id,                                       # trace_info : t_13762
            )

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0# trace_info : t_13891
            and parallel_state.get_tensor_model_parallel_rank() == 0           # trace_info : t_13899
        ):
            logger.info(                                                       # trace_info : t_13905, t_13907
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'# trace_info : t_13906
            )
            for index, bucket in enumerate(self.buckets):                      # trace_info : t_13912, t_14145
                numel = 0                                                      # trace_info : t_13913
                for param in bucket.params:                                    # trace_info : t_13914, t_13916, t_13918, t_13920, t_13922, ...
                    numel += param.data.nelement()                             # trace_info : t_13915, t_13917, t_13919, t_13921, t_13923, ...
                logger.info(f'Params for bucket {index+1} ({numel} elements):')# trace_info : t_13971
                for param in bucket.params:                                    # trace_info : t_13976, t_13982, t_13988, t_13994, t_14000, ...
                    logger.info(f'    {param_to_name[param]}')                 # trace_info : t_13977, t_13983, t_13989, t_13995, t_14001, ...

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()                                # trace_info : t_13311, t_13327, t_13343, t_13359, t_13375, ...
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'# trace_info : t_13312, t_13328, t_13344, t_13360, t_13376, ...
        if buffer_type == BufferType.PARAM:                                    # trace_info : t_13313, t_13329, t_13345, t_13361, t_13377, ...
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:                                   # trace_info : t_13314, t_13330, t_13346, t_13362, t_13378, ...
            buffer_tensor = self.grad_data[start_index:end_index]              # trace_info : t_13315, t_13331, t_13347, t_13363, t_13379, ...
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)                              # trace_info : t_13316, t_13332, t_13348, t_13364, t_13380, ...
        return buffer_tensor                                                   # trace_info : t_13317, t_13333, t_13349, t_13365, t_13381, ...

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
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_13764
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]      # trace_info : t_13765

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None                                             # trace_info : t_13766
        if self.param_data is not None:                                        # trace_info : t_13767
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(                                        # trace_info : t_13768, t_13770
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD# trace_info : t_13769
        )
        bucket = Bucket(                                                       # trace_info : t_13778, t_13788
            ddp_config=self.ddp_config,                                        # trace_info : t_13779
            params=bucket_params,                                              # trace_info : t_13780
            param_data=bucketed_param_data,                                    # trace_info : t_13781
            grad_data=bucketed_grad_data,                                      # trace_info : t_13782
            offset=start_index,                                                # trace_info : t_13783
            numel_unpadded=numel_unpadded,                                     # trace_info : t_13784
            data_parallel_group=self.data_parallel_group,                      # trace_info : t_13785
            data_parallel_world_size=self.data_parallel_world_size,            # trace_info : t_13786
            gradient_scaling_factor=self.gradient_scaling_factor,              # trace_info : t_13787
        )
        self.buckets.append(bucket)                                            # trace_info : t_13805
        for bucket_param in bucket_params:                                     # trace_info : t_13806, t_13809, t_13812, t_13815, t_13818, ...
            assert bucket_param not in self.param_to_bucket                    # trace_info : t_13807, t_13810, t_13813, t_13816, t_13819, ...
            self.param_to_bucket[bucket_param] = bucket                        # trace_info : t_13808, t_13811, t_13814, t_13817, t_13820, ...

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()                                                 # trace_info : t_17790, t_21373, t_88980
        for bucket in self.buckets:                                            # trace_info : t_17791, t_17796, t_21374, t_21379, t_88981, ...
            bucket.reset()                                                     # trace_info : t_17792, t_21375, t_88982
        self.is_last_microbatch = True                                         # trace_info : t_17797, t_21380, t_88987

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
        for bucket in self.buckets:                                            # trace_info : t_19742, t_19758, t_23379, t_23395, t_90986, ...
            bucket.finish_grad_sync()                                          # trace_info : t_19743, t_23380, t_90987

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
