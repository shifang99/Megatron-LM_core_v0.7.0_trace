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
        for blacklisted_module in BLACKLISTED_MODULES:                         # trace_info : t_9204, t_9206, t_10879, t_10881, t_10943, ...
            if record.name.startswith(blacklisted_module):                     # trace_info : t_9205, t_10880, t_10944, t_10950, t_10956, ...
                return False
        return True                                                            # trace_info : t_9207, t_10882, t_10946, t_10952, t_10958, ...
