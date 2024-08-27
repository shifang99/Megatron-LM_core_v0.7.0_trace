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
        for blacklisted_module in BLACKLISTED_MODULES:                         # trace_info : t_12453, t_12455, t_13852, t_13854, t_13904, ...
            if record.name.startswith(blacklisted_module):                     # trace_info : t_12454, t_13853, t_13905, t_13911, t_13917, ...
                return False
        return True                                                            # trace_info : t_12456, t_13855, t_13907, t_13913, t_13919, ...
