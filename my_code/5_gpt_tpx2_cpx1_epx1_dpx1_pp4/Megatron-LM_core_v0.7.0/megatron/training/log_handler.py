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
        for blacklisted_module in BLACKLISTED_MODULES:                         # trace_info : t_12492, t_12494, t_14061, t_14063, t_14121, ...
            if record.name.startswith(blacklisted_module):                     # trace_info : t_12493, t_14062, t_14122, t_14128, t_14134, ...
                return False
        return True                                                            # trace_info : t_12495, t_14064, t_14124, t_14130, t_14136, ...
