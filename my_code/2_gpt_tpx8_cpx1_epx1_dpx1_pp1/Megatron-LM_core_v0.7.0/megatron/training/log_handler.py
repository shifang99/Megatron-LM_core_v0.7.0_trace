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
        for blacklisted_module in BLACKLISTED_MODULES:                         # trace_info : t_14071, t_14073, t_15746, t_15748, t_15810, ...
            if record.name.startswith(blacklisted_module):                     # trace_info : t_14072, t_15747, t_15811, t_15817, t_15823, ...
                return False
        return True                                                            # trace_info : t_14074, t_15749, t_15813, t_15819, t_15825, ...
