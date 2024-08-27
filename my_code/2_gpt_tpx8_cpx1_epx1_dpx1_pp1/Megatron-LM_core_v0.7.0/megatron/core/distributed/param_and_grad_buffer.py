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
        self.ddp_config = ddp_config                                           # trace_info : t_15627

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params                                              # trace_info : t_15628
        self.params = set(params)                                              # trace_info : t_15629
        self.params_with_grad = set()                                          # trace_info : t_15630
        self.param_data = param_data                                           # trace_info : t_15631
        self.grad_data = grad_data                                             # trace_info : t_15632
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset                                                   # trace_info : t_15633
        self.numel_unpadded = numel_unpadded                                   # trace_info : t_15634
        self.data_parallel_group = data_parallel_group                         # trace_info : t_15635
        self.data_parallel_world_size = data_parallel_world_size               # trace_info : t_15636
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)# trace_info : t_15637
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_15638

        self.reset()                                                           # trace_info : t_15639

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()                                          # trace_info : t_15640, t_19523, t_23079, t_26689
        self.communication_handle = None                                       # trace_info : t_15641, t_19524, t_23080, t_26690
        self.communication_issued = False                                      # trace_info : t_15642, t_19525, t_23081, t_26691

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued# trace_info : t_21451, t_25061, t_28671
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:                              # trace_info : t_21452, t_25062, t_28672
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        if self.gradient_scaling_factor != 1.0:                                # trace_info : t_21453, t_25063, t_28673
            self.grad_data *= self.gradient_scaling_factor
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_21454, t_25064, t_28674
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
            self.communication_handle = torch.distributed.all_reduce(          # trace_info : t_21455, t_21459, t_25065, t_25069, t_28675, ...
                self.grad_data,                                                # trace_info : t_21456, t_25066, t_28676
                group=self.data_parallel_group,                                # trace_info : t_21457, t_25067, t_28677
                async_op=self.ddp_config.overlap_grad_reduce,                  # trace_info : t_21458, t_25068, t_28678
            )
        self.communication_issued = True                                       # trace_info : t_21460, t_25070, t_28680

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:                            # trace_info : t_21449, t_25059, t_28669
            self.start_grad_sync()                                             # trace_info : t_21450, t_25060, t_28670
            return                                                             # trace_info : t_21461, t_25071, t_28681
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
        self.ddp_config = ddp_config                                           # trace_info : t_14474

        # Check that params are unique.
        unique_params = set()                                                  # trace_info : t_14475
        for param in params:                                                   # trace_info : t_14476, t_14479, t_14482, t_14485, t_14488, ...
            assert param not in unique_params                                  # trace_info : t_14477, t_14480, t_14483, t_14486, t_14489, ...
            unique_params.add(param)                                           # trace_info : t_14478, t_14481, t_14484, t_14487, t_14490, ...
        del unique_params                                                      # trace_info : t_14561

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype                                         # trace_info : t_14562
        self.grad_dtype = grad_dtype                                           # trace_info : t_14563
        self.data_parallel_group = data_parallel_group                         # trace_info : t_14564
        self.data_parallel_world_size = torch.distributed.get_world_size(      # trace_info : t_14565, t_14567
            group=self.data_parallel_group                                     # trace_info : t_14566
        )
        self.gradient_scaling_factor = gradient_scaling_factor                 # trace_info : t_14568
        self.is_last_microbatch = True                                         # trace_info : t_14569

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []                                                      # trace_info : t_14570
        self.param_to_bucket = {}  # Param -> bucket mapping.                  # trace_info : t_14571
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).# trace_info : t_14572

        def _pad(number_to_be_padded: int, divisor: int) -> int:               # trace_info : t_14573
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:                            # trace_info : t_14574
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:                      # trace_info : t_15119, t_15593
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index                                                  # trace_info : t_15120, t_15594

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0                                                   # trace_info : t_14575
        bucket_data_start_index = data_start_index                             # trace_info : t_14576
        bucket_params = set()                                                  # trace_info : t_14577
        self.bucket_indices = []                                               # trace_info : t_14578
        per_bucket_numel_unpadded = []                                         # trace_info : t_14579
        bucket_id = 0                                                          # trace_info : t_14580

        def _create_new_bucket(data_end_index: int) -> int:                    # trace_info : t_14581
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)# trace_info : t_15117
            data_end_index = _pad_if_needed(data_end_index)                    # trace_info : t_15118
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))# trace_info : t_15121
            bucket_data_start_index = data_end_index                           # trace_info : t_15122
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()                                              # trace_info : t_15123
            bucket_id += 1                                                     # trace_info : t_15124
            # Return the potentially padded data_end_index.
            return data_end_index                                              # trace_info : t_15125

        for param in params[::-1]:                                             # trace_info : t_14582, t_14601, t_14620, t_14639, t_14658, ...
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:                                        # trace_info : t_14583, t_14602, t_14621, t_14640, t_14659, ...
                continue
            this_numel = param.data.nelement()                                 # trace_info : t_14584, t_14603, t_14622, t_14641, t_14660, ...
            data_end_index = data_start_index + this_numel                     # trace_info : t_14585, t_14604, t_14623, t_14642, t_14661, ...

            def _does_param_require_new_bucket(param):                         # trace_info : t_14586, t_14605, t_14624, t_14643, t_14662, ...
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (                                                       # trace_info : t_14589, t_14598, t_14608, t_14617, t_14627, ...
                    getattr(param, "shared_embedding", False)                  # trace_info : t_14588, t_14597, t_14607, t_14616, t_14626, ...
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:# trace_info : t_14587, t_14606, t_14625, t_14644, t_14663, ...
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (                                    # trace_info : t_14593, t_14612, t_14631, t_14650, t_14669, ...
                data_start_index,                                              # trace_info : t_14590, t_14609, t_14628, t_14647, t_14666, ...
                data_end_index,                                                # trace_info : t_14591, t_14610, t_14629, t_14648, t_14667, ...
                bucket_id,                                                     # trace_info : t_14592, t_14611, t_14630, t_14649, t_14668, ...
            )
            bucket_params.add(param)                                           # trace_info : t_14594, t_14613, t_14632, t_14651, t_14670, ...

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None                                        # trace_info : t_14595, t_14614, t_14633, t_14652, t_14671, ...
                and (data_end_index - bucket_data_start_index) >= bucket_size  # trace_info : t_14599, t_14618, t_14637, t_14656, t_14675, ...
            ) or _does_param_require_new_bucket(param):                        # trace_info : t_14596, t_14615, t_14634, t_14653, t_14672, ...
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index                                  # trace_info : t_14600, t_14619, t_14638, t_14657, t_14676, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_15115
            data_end_index = _create_new_bucket(data_end_index)                # trace_info : t_15116

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index                                            # trace_info : t_15126
        self.numel_unpadded = sum(per_bucket_numel_unpadded)                   # trace_info : t_15127
        assert self.numel_unpadded <= self.numel                               # trace_info : t_15128
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_15129
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded                           # trace_info : t_15130

        self.param_data = None                                                 # trace_info : t_15131
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_15132
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(                                          # trace_info : t_15133, t_15138
            self.numel,                                                        # trace_info : t_15134
            dtype=self.grad_dtype,                                             # trace_info : t_15135
            device=torch.cuda.current_device(),                                # trace_info : t_15136
            requires_grad=False,                                               # trace_info : t_15137
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = set()                                                  # trace_info : t_15139
        bucket_data_start_index = 0                                            # trace_info : t_15140
        cur_bucket_id = 0                                                      # trace_info : t_15141
        for param in params[::-1]:                                             # trace_info : t_15142, t_15158, t_15174, t_15190, t_15206, ...
            if not param.requires_grad:                                        # trace_info : t_15143, t_15159, t_15175, t_15191, t_15207, ...
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]# trace_info : t_15144, t_15160, t_15176, t_15192, t_15208, ...

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:                                    # trace_info : t_15145, t_15161, t_15177, t_15193, t_15209, ...
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

            param.main_grad = self._get(                                       # trace_info : t_15146, t_15148, t_15162, t_15164, t_15178, ...
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD# trace_info : t_15147, t_15163, t_15179, t_15195, t_15211, ...
            )
            if bucket_id != cur_bucket_id:                                     # trace_info : t_15156, t_15172, t_15188, t_15204, t_15220, ...
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
            bucket_params.add(param)                                           # trace_info : t_15157, t_15173, t_15189, t_15205, t_15221, ...

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:                                             # trace_info : t_15591
            bucket_data_end_index = _pad_if_needed(data_end_index)             # trace_info : t_15592
            self._set_bucket(                                                  # trace_info : t_15595, t_15601
                bucket_params=bucket_params,                                   # trace_info : t_15596
                start_index=bucket_data_start_index,                           # trace_info : t_15597
                end_index=bucket_data_end_index,                               # trace_info : t_15598
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],       # trace_info : t_15599
                bucket_id=cur_bucket_id,                                       # trace_info : t_15600
            )

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0# trace_info : t_15729
            and parallel_state.get_tensor_model_parallel_rank() == 0           # trace_info : t_15737
        ):
            logger.info(                                                       # trace_info : t_15743, t_15745
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'# trace_info : t_15744
            )
            for index, bucket in enumerate(self.buckets):                      # trace_info : t_15750, t_15983
                numel = 0                                                      # trace_info : t_15751
                for param in bucket.params:                                    # trace_info : t_15752, t_15754, t_15756, t_15758, t_15760, ...
                    numel += param.data.nelement()                             # trace_info : t_15753, t_15755, t_15757, t_15759, t_15761, ...
                logger.info(f'Params for bucket {index+1} ({numel} elements):')# trace_info : t_15809
                for param in bucket.params:                                    # trace_info : t_15814, t_15820, t_15826, t_15832, t_15838, ...
                    logger.info(f'    {param_to_name[param]}')                 # trace_info : t_15815, t_15821, t_15827, t_15833, t_15839, ...

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()                                # trace_info : t_15149, t_15165, t_15181, t_15197, t_15213, ...
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'# trace_info : t_15150, t_15166, t_15182, t_15198, t_15214, ...
        if buffer_type == BufferType.PARAM:                                    # trace_info : t_15151, t_15167, t_15183, t_15199, t_15215, ...
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:                                   # trace_info : t_15152, t_15168, t_15184, t_15200, t_15216, ...
            buffer_tensor = self.grad_data[start_index:end_index]              # trace_info : t_15153, t_15169, t_15185, t_15201, t_15217, ...
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)                              # trace_info : t_15154, t_15170, t_15186, t_15202, t_15218, ...
        return buffer_tensor                                                   # trace_info : t_15155, t_15171, t_15187, t_15203, t_15219, ...

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
        if self.ddp_config.use_distributed_optimizer:                          # trace_info : t_15602
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]      # trace_info : t_15603

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None                                             # trace_info : t_15604
        if self.param_data is not None:                                        # trace_info : t_15605
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(                                        # trace_info : t_15606, t_15608
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD# trace_info : t_15607
        )
        bucket = Bucket(                                                       # trace_info : t_15616, t_15626
            ddp_config=self.ddp_config,                                        # trace_info : t_15617
            params=bucket_params,                                              # trace_info : t_15618
            param_data=bucketed_param_data,                                    # trace_info : t_15619
            grad_data=bucketed_grad_data,                                      # trace_info : t_15620
            offset=start_index,                                                # trace_info : t_15621
            numel_unpadded=numel_unpadded,                                     # trace_info : t_15622
            data_parallel_group=self.data_parallel_group,                      # trace_info : t_15623
            data_parallel_world_size=self.data_parallel_world_size,            # trace_info : t_15624
            gradient_scaling_factor=self.gradient_scaling_factor,              # trace_info : t_15625
        )
        self.buckets.append(bucket)                                            # trace_info : t_15643
        for bucket_param in bucket_params:                                     # trace_info : t_15644, t_15647, t_15650, t_15653, t_15656, ...
            assert bucket_param not in self.param_to_bucket                    # trace_info : t_15645, t_15648, t_15651, t_15654, t_15657, ...
            self.param_to_bucket[bucket_param] = bucket                        # trace_info : t_15646, t_15649, t_15652, t_15655, t_15658, ...

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()                                                 # trace_info : t_19520, t_23076, t_26686
        for bucket in self.buckets:                                            # trace_info : t_19521, t_19526, t_23077, t_23082, t_26687, ...
            bucket.reset()                                                     # trace_info : t_19522, t_23078, t_26688
        self.is_last_microbatch = True                                         # trace_info : t_19527, t_23083, t_26693

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
        for bucket in self.buckets:                                            # trace_info : t_21447, t_21462, t_25057, t_25072, t_28667, ...
            bucket.finish_grad_sync()                                          # trace_info : t_21448, t_25058, t_28668

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
