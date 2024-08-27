# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy
import torch

logger = logging.getLogger(__name__)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process.
    """
    import os                                                                  # trace_info : t_8549
    import subprocess                                                          # trace_info : t_8550

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]       # trace_info : t_8551
    if subprocess.run(command).returncode != 0:                                # trace_info : t_8552
        import sys

        log_single_rank(logger, logging.ERROR, "Failed to compile the C++ dataset helper functions")
        sys.exit(1)


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    """If torch distributed is initialized, log only on rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    if torch.distributed.is_initialized():                                     # trace_info : t_16153, t_16188, t_16264, t_16290, t_16303, ...
        if torch.distributed.get_rank() == rank:                               # trace_info : t_16154, t_16189, t_16265, t_16291, t_16304, ...
            logger.log(*args, **kwargs)                                        # trace_info : t_16155, t_16190, t_16266, t_16292, t_16305, ...
    else:
        logger.log(*args, **kwargs)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """
    w = numpy.array(weights, dtype=numpy.float64)                              # trace_info : t_16119
    w_sum = numpy.sum(w)                                                       # trace_info : t_16120
    w = (w / w_sum).tolist()                                                   # trace_info : t_16121
    return w                                                                   # trace_info : t_16122


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    """Get the megatron.core.datasets.blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend from the blend list
    
    Args:
        blend (Optional[List[str]]): The blend list, which can be either (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    if blend is None:                                                          # trace_info : t_16065, t_16072, t_16075, t_16078
        return None                                                            # trace_info : t_16073, t_16076, t_16079

    if len(blend) % 2 == 1:                                                    # trace_info : t_16066
        weight_per_dataset = None                                              # trace_info : t_16067
        raw_prefix_per_dataset = blend                                         # trace_info : t_16068
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]     # trace_info : t_16069

    return prefix_per_dataset, weight_per_dataset                              # trace_info : t_16070
