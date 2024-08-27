import datetime
import linecache
import os
import torch
import csv

import sys
from functools import wraps

## Global variables
module_name = None
func_name = None
lineno = None
filename = None
relative_filename = None
log_line = False
log_this_line = False
last_memory_allocated = 0
trace_in_which_dir = ""
trace_filename = ""
trace_line_num = 0

def trace_with_params(trace_params):
    def decorator(func):

        global trace_in_which_dir, trace_filename

        trace_in_which_dir = trace_params.get("trace_in_which_dir", None)
        if trace_in_which_dir == None:
            trace_in_which_dir = os.path.dirname(os.path.abspath(__file__))

        trace_filename = trace_params.get("trace_filename", None)
        assert trace_filename is not None, (
            f'should set trace_filename in trace_params before using trace_with_params'
        )
        with open(trace_filename, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["trace_line_num", "relative_filename", "lineno", "func_name", "memory_allocated(MiB)", "memory_allocated_changed(MiB)", "max_memory_allocated(MiB)", "line"])        
        exclude_funcs = trace_params.get("exclude_funcs", None)                
           
        def trace(frame, event, arg):

            if event == 'line':
                try:                            
                    global func_name, filename, relative_filename, module_name, lineno, trace_line_num
                    global log_line, trace_filename, trace_in_which_dir
                    global last_memory_allocated

                    # about _previous_ line (!)
                    if log_line:
                        trace_line_num += 1
                        # where_str = module_name+':'+func_name+':'+str(lineno)
                        # where_str = filename+':'+func_name+':'+str(lineno)
                        line = linecache.getline(filename, lineno)
                        memory_allocated = torch.cuda.memory_allocated()
                        max_memory_allocated = torch.cuda.max_memory_allocated()
                        memory_allocated_changed = memory_allocated - last_memory_allocated

                        with open(trace_filename, 'a+', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([                        
                                f"{trace_line_num:<6d}", 
                                f"{relative_filename:<70}",
                                f"{lineno:<6}",
                                f"{func_name:<40}",
                                f"{(memory_allocated_changed)/1024**2:<7.1f}",                    
                                f"{(memory_allocated)/1024**2:<7.1f}",
                                f"{(max_memory_allocated)/1024**2:<7.1f}",
                                f"{line.rstrip()}"
                                ])                    
                            
                        last_memory_allocated = memory_allocated

                    # save details about line _to be_ executed
                    module_name = frame.f_globals["__name__"]
                    func_name = frame.f_code.co_name
                    lineno = frame.f_lineno
                    filename = frame.f_globals["__file__"]
                    if (filename.endswith(".pyc") or
                            filename.endswith(".pyo")):
                        filename = filename[:-1]          
                    filename = os.path.abspath(filename)

                    log_line = True
                    # only profile codes within the parenet folder, otherwise there are too many function calls into other pytorch scripts
                    # need to modify the key words below to suit your case.
                    # if trace_in_which_dir not in os.path.dirname(os.path.abspath(filename)):  
                    if not os.path.dirname(filename).startswith(trace_in_which_dir):
                        log_line = False  # skip current line evaluation
                    else:
                        relative_filename = "./" + filename[len(trace_in_which_dir):]

                    # if "transformer_engine/pytorch/attention.py" in filename:
                    #     log_line = True  # 临时跟踪这类文件
                    #     relative_filename = filename
                    
                    if exclude_funcs is not None:
                        if any(exclude_func in func_name for exclude_func in  exclude_funcs ):
                            log_line = False  # skip current line evaluation                    

                    return trace

                except (KeyError, AttributeError):
                    pass

            return trace

        @wraps(func)
        def inner(*args, **kwargs):
            sys.settrace(trace)
            result = func(*args, **kwargs)
            sys.settrace(None)
            return result
        
        return inner
    return decorator
