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
        for blacklisted_module in BLACKLISTED_MODULES:                         # trace_info : t_12233, t_12235, t_13908, t_13910, t_13972, ...
            if record.name.startswith(blacklisted_module):                     # trace_info : t_12234, t_13909, t_13973, t_13979, t_13985, ...
                return False
        return True                                                            # trace_info : t_12236, t_13911, t_13975, t_13981, t_13987, ...
