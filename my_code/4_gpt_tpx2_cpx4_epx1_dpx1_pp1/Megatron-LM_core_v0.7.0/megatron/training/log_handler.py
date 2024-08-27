# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import sys
from logging import LogRecord, StreamHandler

BLACKLISTED_MODULES = ["torch.distributed"]


class CustomHandler(StreamHandler):
    """
    Custom handler to filter out logging from code outside of
    Megatron Core, and dump to stdout.
    """

    def __init__(self):
        super().__init__(stream=sys.stdout)

    def filter(self, record: LogRecord) -> bool:
        # Prevent log entries that come from the blacklisted modules
        # through (e.g., PyTorch Distributed).
        for blacklisted_module in BLACKLISTED_MODULES:                         # trace_info : t_12063, t_12065, t_13738, t_13740, t_13802, ...
            if record.name.startswith(blacklisted_module):                     # trace_info : t_12064, t_13739, t_13803, t_13809, t_13815, ...
                return False
        return True                                                            # trace_info : t_12066, t_13741, t_13805, t_13811, t_13817, ...
