# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.           # trace_info : t_88673, t_156040

""" Strategies using PyTorch distributed.checkpoint as an underlying format. """# trace_info : t_31428
import dataclasses                                                             # trace_info : t_31429
import io                                                                      # trace_info : t_31430
import itertools                                                               # trace_info : t_31431
from collections import defaultdict                                            # trace_info : t_31432
from logging import getLogger                                                  # trace_info : t_31433
from pathlib import Path                                                       # trace_info : t_31434
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast     # trace_info : t_31435

import torch                                                                   # trace_info : t_31436
from torch.distributed import checkpoint                                       # trace_info : t_31437
from torch.distributed._shard.metadata import ShardMetadata                    # trace_info : t_31438
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensorMetadata, TensorProperties# trace_info : t_31439
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor# trace_info : t_31440
from torch.distributed.checkpoint import (                                     # trace_info : t_31441
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    LoadPlan,
    SavePlan,
    TensorStorageMetadata,
    WriteItem,
)
from torch.distributed.checkpoint._nested_dict import FLATTEN_MAPPING, unflatten_state_dict# trace_info : t_31442
from torch.distributed.checkpoint._traverse import OBJ_PATH, traverse_state_dict# trace_info : t_31443
from torch.distributed.checkpoint.default_planner import create_default_local_save_plan# trace_info : t_31444
from torch.distributed.checkpoint.planner_helpers import _create_write_items   # trace_info : t_31445

from ..core import CheckpointingException                                      # trace_info : t_31446
from ..dict_utils import nested_values                                         # trace_info : t_31447
from ..mapping import (                                                        # trace_info : t_31448
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    StateDict,
    is_main_replica,
)
from .async_utils import AsyncRequest                                          # trace_info : t_31449
from .base import (                                                            # trace_info : t_31450
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
    SaveShardedStrategy,
    StrategyAction,
    default_strategies,
)
from .filesystem_async import FileSystemWriterAsync                            # trace_info : t_31451
from .state_dict_saver import save_state_dict_async_finalize, save_state_dict_async_plan# trace_info : t_31500

_import_trigger = None                                                         # trace_info : t_31539

logger = getLogger(__name__)                                                   # trace_info : t_31540


def flatten_state_dict(                                                        # trace_info : t_31541, t_31543, t_31545
    state_dict: ShardedStateDict,                                              # trace_info : t_31542
) -> Tuple[ShardedStateDict, Dict[str, OBJ_PATH]]:                             # trace_info : t_31544
    """ Flattens state dict into a single level dict.

    It's a copy of torch.distributed.checkpoint._nested_dict.flatten_state_dict
    which also accepts ShardedBase tensors as terminal objects

    Args:
        state_dict (ShardedStateDict): state dict to be flattened

    Returns (tuple): flattened state dict and a mapping allowing to recreate the original one

    """
    flattened = {}                                                             # trace_info : t_86130, t_153497
    mappings = {}                                                              # trace_info : t_86131, t_153498

    def flat_copy(path: OBJ_PATH, value: Any) -> None:                         # trace_info : t_86132, t_153499
        new_fqn = ".".join(map(str, path))                                     # trace_info : t_86135, t_86139, t_86143, t_86147, t_86151, ...
        if new_fqn in flattened:                                               # trace_info : t_86136, t_86140, t_86144, t_86148, t_86152, ...
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value                                             # trace_info : t_86137, t_86141, t_86145, t_86149, t_86153, ...
        mappings[new_fqn] = path                                               # trace_info : t_86138, t_86142, t_86146, t_86150, t_86154, ...

    traverse_state_dict(state_dict, flat_copy, lambda x: isinstance(x, (torch.Tensor, ShardedBase)))# trace_info : t_86133, t_86134, t_86279, t_86280, t_86281, ...
    return flattened, mappings                                                 # trace_info : t_86653, t_154020


def sharded_tensor_to_torch_sharded_tensor(                                    # trace_info : t_31547, t_31549, t_31551, t_31553
    sh_tens: List[ShardedTensor], rank: Optional[int] = None                   # trace_info : t_31546, t_31548, t_31550
) -> TorchShardedTensor:                                                       # trace_info : t_31552
    """Convert MCore ShardedTensor to PyT ShardedTensor. PyT requires information about all chunks.

    NOTE: this function assumes regular (grid) sharding of the MCore ShardedTensor.
    The only local irregularities could be introduced with a `flattened_range` attribute.

    NOTE: `flattened_range` is currently supported only for 1D tensors.

    This function follows the logic of torch.distributed.fsdp._shard_utils._create_chunk_sharded_tensor.
    Additionally, it saves `prepend_axis_num` (specific to MCore) as an attribute
    for further restoration in `_unwrap_pyt_sharded_tensor`.

    Args:
        sh_tens (List[ShardedTensor]): list of sharded tensors to convert
        rank (int, optional): current process rank passed to PyT ShardedTensor.
            If None, assumes rank in the default pg.

    Returns (TorchShardedTensor): PyT ShardedTensor containing all passed shards.

    """
    if rank is None:                                                           # trace_info : t_87513, t_87648, t_87716, t_87801, t_87886, ...
        rank = torch.distributed.get_rank()

    some_sh_ten = sh_tens[0]                                                   # trace_info : t_87514, t_87649, t_87717, t_87802, t_87887, ...
    has_flattened_range = some_sh_ten.flattened_range is not None              # trace_info : t_87515, t_87650, t_87718, t_87803, t_87888, ...

    prepend_axis_num = sh_tens[0].prepend_axis_num                             # trace_info : t_87516, t_87651, t_87719, t_87804, t_87889, ...
    # Determine local shards
    if has_flattened_range:                                                    # trace_info : t_87517, t_87652, t_87720, t_87805, t_87890, ...
        if prepend_axis_num:
            raise NotImplementedError(
                '`prepend_axis_num` attribute of ShardedTensor not supported'
                'together with `flattened_range` for PyT Distributed format'
            )
        for sh_ten in sh_tens:
            assert sh_ten.flattened_range is not None
            assert len(sh_ten.global_offset) == 1, sh_ten

        local_shards = [
            Shard.from_tensor_and_offsets(
                sh_ten.data, [sh_ten.global_offset[0] + sh_ten.flattened_range.start], rank
            )
            for sh_ten in sh_tens
        ]
        offsets_shape = some_sh_ten.local_shape  # used to determine local offsets
    else:
        # Apply extra axes `prepend_axis_num` with a view
        for sh_ten in sh_tens:                                                 # trace_info : t_87518, t_87522, t_87653, t_87656, t_87721, ...
            assert sh_ten.flattened_range is None, sh_ten.flattened_range      # trace_info : t_87519, t_87654, t_87722, t_87807, t_87892, ...
            if prepend_axis_num:                                               # trace_info : t_87520, t_87655, t_87723, t_87808, t_87893, ...
                sh_ten.data = sh_ten.data.view((1,) * prepend_axis_num + sh_ten.local_shape)# trace_info : t_87521, t_87724, t_87809, t_87894, t_87979, ...

        local_shards = [                                                       # trace_info : t_87523, t_87525, t_87657, t_87659, t_87726, ...
            Shard.from_tensor_and_offsets(sh_ten.data, list(sh_ten.global_offset), rank)
            for sh_ten in sh_tens                                              # trace_info : t_87524, t_87658, t_87727, t_87812, t_87897, ...
        ]
        offsets_shape = some_sh_ten.data.shape  # includes prepended axes      # trace_info : t_87526, t_87660, t_87729, t_87814, t_87899, ...

    local_global_offsets = {}                                                  # trace_info : t_87527, t_87661, t_87730, t_87815, t_87900, ...
    for sh_ten in sh_tens:                                                     # trace_info : t_87528, t_87530, t_87662, t_87664, t_87731, ...
        local_global_offsets.setdefault(sh_ten.global_offset, []).append(sh_ten)# trace_info : t_87529, t_87663, t_87732, t_87817, t_87902, ...

    # Create a ShardedTensor without invoking communication. Determine global shards
    shard_metadata = []                                                        # trace_info : t_87531, t_87665, t_87734, t_87819, t_87904, ...
    # NOTE: here we assume a regular grid of shards
    for fragment_offsets in itertools.product(*map(range, some_sh_ten.axis_fragmentations)):# trace_info : t_87532, t_87544, t_87551, t_87558, t_87565, ...
        offset = tuple(map(lambda x: x[0] * x[1], zip(fragment_offsets, offsets_shape)))# trace_info : t_87533, t_87534, t_87535, t_87536, t_87545, ...
        if offset in local_global_offsets:                                     # trace_info : t_87537, t_87549, t_87556, t_87563, t_87670, ...
            # local shard
            placement = f"rank:{rank}/cuda"                                    # trace_info : t_87538, t_87671, t_87741, t_87826, t_87925, ...
            for sh_ten in local_global_offsets[offset]:                        # trace_info : t_87539, t_87543, t_87672, t_87676, t_87742, ...
                if has_flattened_range:                                        # trace_info : t_87540, t_87673, t_87743, t_87828, t_87927, ...
                    offset = (sh_ten.global_offset[0] + sh_ten.flattened_range.start,)
                size = sh_ten.data.shape                                       # trace_info : t_87541, t_87674, t_87744, t_87829, t_87928, ...
                shard_metadata.append(ShardMetadata(offset, size, placement))  # trace_info : t_87542, t_87675, t_87745, t_87830, t_87929, ...

        else:
            # for shards from other ranks we provide simplistic data - this information will be discarded
            # during TorchShardedTensor._init_from_local_shards_and_global_metadata call
            shard_metadata.append(ShardMetadata(offset, offsets_shape, "cuda"))# trace_info : t_87550, t_87557, t_87564, t_87682, t_87753, ...

    tensor = some_sh_ten.data                                                  # trace_info : t_87566, t_87684, t_87769, t_87854, t_87939, ...
    sharded_tensor_metadata = ShardedTensorMetadata(                           # trace_info : t_87567, t_87577, t_87685, t_87695, t_87770, ...
        shards_metadata=shard_metadata,                                        # trace_info : t_87568, t_87686, t_87771, t_87856, t_87941, ...
        size=torch.Size(some_sh_ten.global_shape),                             # trace_info : t_87569, t_87687, t_87772, t_87857, t_87942, ...
        tensor_properties=TensorProperties(                                    # trace_info : t_87570, t_87576, t_87688, t_87694, t_87773, ...
            dtype=tensor.dtype,                                                # trace_info : t_87571, t_87689, t_87774, t_87859, t_87944, ...
            layout=tensor.layout,                                              # trace_info : t_87572, t_87690, t_87775, t_87860, t_87945, ...
            requires_grad=tensor.requires_grad,                                # trace_info : t_87573, t_87691, t_87776, t_87861, t_87946, ...
            memory_format=torch.contiguous_format,                             # trace_info : t_87574, t_87692, t_87777, t_87862, t_87947, ...
            pin_memory=tensor.is_pinned(),                                     # trace_info : t_87575, t_87693, t_87778, t_87863, t_87948, ...
        ),
    )
    pyt_sh_ten = TorchShardedTensor._init_from_local_shards_and_global_metadata(# trace_info : t_87578, t_87580, t_87696, t_87698, t_87781, ...
        local_shards, sharded_tensor_metadata=sharded_tensor_metadata, process_group=None# trace_info : t_87579, t_87697, t_87782, t_87867, t_87952, ...
    )
    pyt_sh_ten.prepend_axis_num = prepend_axis_num                             # trace_info : t_87581, t_87699, t_87784, t_87869, t_87954, ...
    return pyt_sh_ten                                                          # trace_info : t_87582, t_87700, t_87785, t_87870, t_87955, ...


def mcore_to_pyt_state_dict(                                                   # trace_info : t_31556, t_31558, t_31560, t_31562, t_31564
    state_dict: Dict[str, List[ShardedBase]],                                  # trace_info : t_31557
    is_loading: bool = False,                                                  # trace_info : t_31554, t_31559
    init_device: torch.device = torch.device("cpu"),                           # trace_info : t_31555, t_31561
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:                         # trace_info : t_31563
    """Turn state dict with ShardedTensors and ShardedObjects to state dict compatible with PyT Dist format.

    Operates in-place and returns the original state dict.

    Args:
        state_dict (Dict[str, List[ShardedBase]]): flattened state dict, where values
            are lists of either ShardedTensor or ShardedObjects.
        is_loading (bool, optional): flag indicating if loading or saving. Defaults to False.
        init_device (torch.device, optional): device to initialize potentially missing tensors
            during loading. Defaults to 'cpu'.

    Returns (Dict[str, Union[TorchShardedTensor, io.BytesIO]]): original dictionary with values
        converted either into PyT ShardedTensors or io.BytesIO.

    """
    rank = torch.distributed.get_rank()                                        # trace_info : t_87466, t_154833
    pyt_state_dict = {}                                                        # trace_info : t_87467, t_154834

    def _mcore_to_torch_sharded_tensor(sh_tens: List[ShardedTensor]) -> TorchShardedTensor:# trace_info : t_87468, t_154835
        """Build a PyT ShardedTensor from given shards.

        During loading:
        - if data is None, initialize it with an empty tensor (will be used to copy the data into)
        - if `allow_shape_mismatch` is True, the data is initialized with zeros
            prior to loading (not all parts of the tensor will be read from the checkpoint)
        """
        assert all(isinstance(sh_ten, ShardedTensor) for sh_ten in sh_tens), sh_tens# trace_info : t_87504, t_87505, t_87506, t_87639, t_87640, ...
        for sh_ten in sh_tens:                                                 # trace_info : t_87507, t_87511, t_87642, t_87646, t_87710, ...
            if sh_ten.data is None:                                            # trace_info : t_87508, t_87643, t_87711, t_87796, t_87881, ...
                if is_loading:
                    sh_ten.init_data(
                        init_device,
                        init_fn=torch.zeros if sh_ten.allow_shape_mismatch else torch.empty,
                    )
                else:
                    raise CheckpointingException(f'`data` attr is None for {sh_ten}')
            else:
                sh_ten.data = sh_ten.data.detach()                             # trace_info : t_87509, t_87644, t_87712, t_87797, t_87882, ...
                if sh_ten.allow_shape_mismatch and is_loading:                 # trace_info : t_87510, t_87645, t_87713, t_87798, t_87883, ...
                    sh_ten.data.zero_()

        torch_sh_ten = sharded_tensor_to_torch_sharded_tensor(sh_tens, rank)   # trace_info : t_87512, t_87647, t_87715, t_87800, t_87885, ...
        torch_sh_ten.key = sh_tens[0].key                                      # trace_info : t_87583, t_87701, t_87786, t_87871, t_87956, ...
        return torch_sh_ten                                                    # trace_info : t_87584, t_87702, t_87787, t_87872, t_87957, ...

    def _mcore_to_torch_sharded_object(sh_objs: List[ShardedObject]) -> io.BytesIO:# trace_info : t_87469, t_154836
        """Build io.BytesIO from given sharded objects data."""
        assert all(isinstance(sh_obj, ShardedObject) for sh_obj in sh_objs), sh_objs# trace_info : t_87474, t_87475, t_87476, t_87484, t_87485, ...
        serialized_data = io.BytesIO()                                         # trace_info : t_87477, t_87487, t_87497, t_87592, t_87602, ...
        torch.save([sh_obj.data for sh_obj in sh_objs], serialized_data)       # trace_info : t_87478, t_87488, t_87498, t_87593, t_87603, ...
        return serialized_data                                                 # trace_info : t_87479, t_87489, t_87499, t_87594, t_87604, ...

    for k, v in state_dict.items():                                            # trace_info : t_87470, t_87480, t_87490, t_87500, t_87585, ...
        if isinstance(v[0], ShardedTensor):                                    # trace_info : t_87471, t_87481, t_87491, t_87501, t_87586, ...
            v = cast(List[ShardedTensor], v)                                   # trace_info : t_87502, t_87637, t_87705, t_87790, t_87875, ...
            pyt_state_dict[k] = _mcore_to_torch_sharded_tensor(v)              # trace_info : t_87503, t_87638, t_87706, t_87791, t_87876, ...
        else:
            v = cast(List[ShardedObject], v)                                   # trace_info : t_87472, t_87482, t_87492, t_87587, t_87597, ...
            pyt_state_dict[k] = _mcore_to_torch_sharded_object(v)              # trace_info : t_87473, t_87483, t_87493, t_87588, t_87598, ...

    return pyt_state_dict                                                      # trace_info : t_88286, t_155653


def _unwrap_pyt_sharded_tensor(sh_ten: TorchShardedTensor) -> List[torch.Tensor]:# trace_info : t_31565
    """ Unwrap tensor from PyT ShardedTensor instance.

    If `prepend_axis_num` was non-zero (which is specific to MCore ShardedTensor)
    then the tensor has additional singleton dimensions which should be squeezed.
    """
    prepend_axis_num = getattr(sh_ten, 'prepend_axis_num', 0)
    if prepend_axis_num == 0:
        return [sh.tensor for sh in sh_ten.local_shards()]
    ret_tensors = []
    for sh in sh_ten.local_shards():
        ten = sh.tensor
        for _ in range(prepend_axis_num):
            ten = ten.squeeze(0)
        ret_tensors.append(ten)
    return ret_tensors


def _replace_state_dict_keys_with_sharded_keys(                                # trace_info : t_31567, t_31569, t_31571, t_31573
    sharded_state_dict: ShardedStateDict, keep_only_main_replica: bool = False # trace_info : t_31566, t_31568, t_31570
) -> Tuple[Dict[str, List[ShardedBase]], FLATTEN_MAPPING, Dict[str, List[str]]]:# trace_info : t_31572
    """Group ShardedBase objects by keys and return mappings required for recreating the original dict. """
    flat_sd, flat_mapping = flatten_state_dict(sharded_state_dict)             # trace_info : t_86129, t_153496
    rename_mapping = defaultdict(list)                                         # trace_info : t_86654, t_154021
    new_flat_sd = defaultdict(list)                                            # trace_info : t_86655, t_154022
    for k, sh_base in flat_sd.items():                                         # trace_info : t_86656, t_86662, t_86668, t_86674, t_86680, ...
        assert isinstance(sh_base, ShardedBase), type(sh_base)                 # trace_info : t_86657, t_86663, t_86669, t_86675, t_86681, ...
        key = sh_base.unique_key if isinstance(sh_base, ShardedObject) else sh_base.key# trace_info : t_86658, t_86664, t_86670, t_86676, t_86682, ...
        if is_main_replica(sh_base.replica_id) or not keep_only_main_replica:  # trace_info : t_86659, t_86665, t_86671, t_86677, t_86683, ...
            rename_mapping[key].append(k)                                      # trace_info : t_86703, t_86728, t_86765, t_86773, t_86792, ...
            new_flat_sd[key].append(sh_base)                                   # trace_info : t_86704, t_86729, t_86766, t_86774, t_86793, ...
    return new_flat_sd, flat_mapping, rename_mapping                           # trace_info : t_87460, t_154827


def _replace_sharded_keys_with_state_dict_keys(                                # trace_info : t_31574, t_31576, t_31578, t_31580
    state_dict: Dict[str, List[Union[torch.Tensor, io.BytesIO]]],              # trace_info : t_31575
    flat_mapping: FLATTEN_MAPPING,                                             # trace_info : t_31577
    rename_mapping: Dict[str, List[str]],                                      # trace_info : t_31579
):
    """ Inverse of _replace_state_dict_keys_with_sharded_keys. """
    recovered_sd = {}
    for k, tensors in state_dict.items():
        assert len(tensors) == len(rename_mapping[k])
        for ten, recovered_k in zip(tensors, rename_mapping[k]):
            recovered_sd[recovered_k] = ten

    return unflatten_state_dict(recovered_sd, flat_mapping)


def _restore_dict_types(x: Union[dict, list, Any], keys_template: Union[dict, list, Any]):# trace_info : t_31581
    """ Recursively update `x` keys, based on `keys_template`. """
    if isinstance(keys_template, dict):
        assert isinstance(x, dict), type(x)
        for k, v in keys_template.items():
            if not isinstance(k, str):
                assert str(k) in x, (k, x.keys)
                x[k] = x.pop(str(k))
            _restore_dict_types(x[k], v)
    elif isinstance(keys_template, list):
        assert isinstance(x, list), type(x)
        for x_val, templ_val in zip(x, keys_template):
            _restore_dict_types(x_val, templ_val)


class MCoreSavePlanner(DefaultSavePlanner):                                    # trace_info : t_31582, t_31583
    """Differs with the default planner by saving BytesIO objects on all ranks.# trace_info : t_31584

    In the integration of MCore with PyT Distributed format, BytesIO objects
    come from ShardedObjects, which should be treated as separate objects on each rank
    (not common on all ranks).

    Also, the objects are already packed in io.BytesIO, so no need to redo it
    in transform_object.
    """

    def create_local_plan(self) -> SavePlan:                                   # trace_info : t_31585
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)# trace_info : t_88311, t_155678
        self._add_non_coordinator_iobytes_request(plan)                        # trace_info : t_88312, t_155679
        if self.flatten_state_dict:                                            # trace_info : t_88315, t_155682
            plan = dataclasses.replace(plan, planner_data=self.mappings)       # trace_info : t_88316, t_155683
        self.plan = plan                                                       # trace_info : t_88317, t_155684

        return self.plan                                                       # trace_info : t_88318, t_155685

    def _add_non_coordinator_iobytes_request(self, plan):                      # trace_info : t_31586
        if self.is_coordinator:                                                # trace_info : t_88313, t_155680
            return                                                             # trace_info : t_88314, t_155681
        for fqn, obj in self.state_dict.items():
            if isinstance(obj, io.BytesIO):
                plan.items.extend(_create_write_items(fqn, obj))

    def transform_object(self, write_item: WriteItem, object: Any):            # trace_info : t_31587
        return object                                                          # trace_info : t_88613, t_88614, t_88615, t_88616, t_88617, ...


class MCoreLoadPlanner(DefaultLoadPlanner):                                    # trace_info : t_31588, t_31589
    """Adds global shape validation to the default planner.                    # trace_info : t_31590

    If global shape validation can be ignored (shouldn't!), the default
    load planner can be used.
    """

    def __init__(                                                              # trace_info : t_31592, t_31594, t_31596
        self, *args, shapes_validation_sharded_tensors: Iterable[ShardedTensor] = (), **kwargs# trace_info : t_31591, t_31593
    ) -> None:                                                                 # trace_info : t_31595
        super().__init__(*args, **kwargs)
        self.shapes_validation_sharded_tensors = shapes_validation_sharded_tensors

    def _validate_global_shapes(self, metadata, sharded_tensors):              # trace_info : t_31597
        for sh_ten in sharded_tensors:
            loaded_shape = metadata.state_dict_metadata[sh_ten.key].size
            if loaded_shape != sh_ten.global_shape:
                _msg = (
                    f'Global shape mismatch for loaded ({loaded_shape})'
                    f' and expected ({sh_ten.global_shape}) tensor'
                    f' for key {sh_ten.key}'
                )
                raise CheckpointingException(_msg)

    def create_local_plan(self) -> LoadPlan:                                   # trace_info : t_31598
        self._validate_global_shapes(self.metadata, self.shapes_validation_sharded_tensors)
        return super().create_local_plan()


class TorchDistSaveShardedStrategy(AsyncSaveShardedStrategy):                  # trace_info : t_31599, t_31600
    """Async save strategy for the PyT Distributed format.                     # trace_info : t_31601

    The idea is to translate MCore ShardedTensors into PyT ShardedTensors
    and use the async-adjusted torch.distributed.checkpoint saving mechanism
    provided by the FileSystemWriterAsync writer.
    """

    def __init__(                                                              # trace_info : t_31603, t_31605, t_31607, t_31609, t_31611
        self, backend: str, version: int, keep_only_main_replica: bool = True, thread_count: int = 2# trace_info : t_31602, t_31604, t_31606, t_31608, t_31610
    ):
        """Adds parameters specific to PyT Distributed format
        Args:
            backend (str): format backend string
            version (int): format version
            keep_only_main_replica (bool, optional): PyT Distributed has a mechanism
                for deduplication, but replica_id aware deduplication is more coherent.
                Default is True (recommended to keep it).
            thread_count (int, optional): threads to use during saving.
                Affects the number of files in the checkpoint (saving ranks * num_threads).
        """
        super().__init__(backend, version)                                     # trace_info : t_31634
        self.keep_only_main_replica = keep_only_main_replica                   # trace_info : t_31637
        self.thread_count = thread_count                                       # trace_info : t_31638

    def async_save(                                                            # trace_info : t_31612, t_31614, t_31616, t_31618
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path       # trace_info : t_31613, t_31615
    ) -> AsyncRequest:                                                         # trace_info : t_31617
        """ Translates MCore ShardedTensors to PyT ShardedTensors and saves in PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Path): checkpoint directory

        Returns: None
        """
        # Translate the state dict
        (                                                                      # trace_info : t_87461, t_154828
            sharded_state_dict,                                                # trace_info : t_87462, t_154829
            flat_mapping,                                                      # trace_info : t_87463, t_154830
            rename_mapping,                                                    # trace_info : t_87464, t_154831
        ) = _replace_state_dict_keys_with_sharded_keys(                        # trace_info : t_86126, t_86128, t_153493, t_153495
            sharded_state_dict, self.keep_only_main_replica                    # trace_info : t_86127, t_153494
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)    # trace_info : t_87465, t_154832
        # Use PyT saving mechanism
        writer = FileSystemWriterAsync(checkpoint_dir, thread_count=self.thread_count)# trace_info : t_88287, t_155654

        save_state_dict_ret = save_state_dict_async_plan(                      # trace_info : t_88292, t_88297, t_155659, t_155664
            pyt_state_dict,                                                    # trace_info : t_88293, t_155660
            writer,                                                            # trace_info : t_88294, t_155661
            None,                                                              # trace_info : t_88295, t_155662
            planner=MCoreSavePlanner(dedup_replicated_tensors=not self.keep_only_main_replica),# trace_info : t_88296, t_155663
        )
        return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)# trace_info : t_88666, t_156033

    def _get_save_and_finalize_callbacks(self, writer, save_state_dict_ret) -> AsyncRequest:# trace_info : t_31619
        save_fn_args = writer.get_save_function_and_args()                     # trace_info : t_88667, t_156034
        save_fn, save_args = save_fn_args                                      # trace_info : t_88670, t_156037

        def finalize_fn():                                                     # trace_info : t_88671, t_156038
            save_state_dict_async_finalize(*save_state_dict_ret)               # trace_info : t_88699, t_156066
            torch.distributed.barrier()                                        # trace_info : t_88719, t_156086

        return AsyncRequest(save_fn, save_args, [finalize_fn])                 # trace_info : t_88672, t_156039

    def can_handle_sharded_objects(self):                                      # trace_info : t_31620
        return True


class TorchDistLoadShardedStrategy(LoadShardedStrategy):                       # trace_info : t_31621, t_31622
    """Basic load strategy for the PyT Distributed format. """                 # trace_info : t_31623

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:# trace_info : t_31624
        """Translates MCore ShardedTensors to PyT ShardedTensors and loads from PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]

        orig_sharded_state_dict = sharded_state_dict
        # MCore state dict to PyT Distributed compatible
        (
            sharded_state_dict,
            flat_mapping,
            rename_mapping,
        ) = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, True)
        # Load PyT Distributed format
        checkpoint.load_state_dict(
            pyt_state_dict,
            FileSystemReader(checkpoint_dir),
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors
            ),
        )
        pyt_state_dict = cast(
            Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict
        )
        # Unwrap ShardedTensors and return to original state dict
        mcore_state_dict = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            mcore_state_dict, flat_mapping, rename_mapping
        )
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        return mcore_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path):                     # trace_info : t_31625
        """Uses tensors metadata stored in the metadata file."""
        fs_reader = FileSystemReader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        return {
            k: ShardedTensor.from_rank_offsets(
                k, torch.empty(tp.size, **tp.properties.__dict__, device='meta')
            ).without_data()
            for k, tp in metadata.state_dict_metadata.items()
            if isinstance(tp, TensorStorageMetadata)
        }

    def can_handle_sharded_objects(self):                                      # trace_info : t_31626
        return True

    def check_backend_compatibility(self, loaded_version):                     # trace_info : t_31627
        pass  # TODO

    def check_version_compatibility(self, loaded_version):                     # trace_info : t_31628
        pass  # TODO


default_strategies[StrategyAction.LOAD_SHARDED.value][                         # trace_info : t_31630, t_31632
    ('torch_dist', 1)                                                          # trace_info : t_31631
] = TorchDistLoadShardedStrategy()                                             # trace_info : t_31629
default_strategies[StrategyAction.SAVE_SHARDED.value][                         # trace_info : t_31639, t_31641
    ('torch_dist', 1)                                                          # trace_info : t_31640
] = TorchDistSaveShardedStrategy('torch_dist', 1)                              # trace_info : t_31633
