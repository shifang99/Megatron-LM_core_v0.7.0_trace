import os
import csv

def add_trace_info(filename, line_number, trace_info_prefix, trace_index, target_code_length=79):
    lines = []
    
    # 读取文件的所有内容
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # 替换指定行
    if 1 <= line_number <= len(lines):       
        old_line = lines[line_number-1].strip("\n")
        old_line = old_line.ljust(target_code_length, ' ')
        new_line = old_line                
        if trace_info_prefix not in old_line:
            new_line = f"{old_line}{trace_info_prefix}: t_{str(trace_index)}"  
        else:
            trace_info = old_line[old_line.rfind(trace_info_prefix):]
            if len(trace_info) < 50:                
                new_line = f"{old_line}, t_{str(trace_index)}"  
            else:
                if "..." not in trace_info:
                    new_line = f"{old_line}, ..."
            
        lines[line_number-1] = new_line+"\n"
    else:
        print(f"Error: Line number {line_number} is out of range.")

    # 写回文件
    with open(filename, 'w') as file:
        file.writelines(lines)
        
if __name__ == "__main__":  

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    # 获取当前文件所在目录的上一级目录
    parent_dir = os.path.dirname(current_dir)
    print(f"parent_dir:{parent_dir}")

    code_trace_dirs = []
    code_dir = "./my_code/1_gpt_tpx1_cpx1_epx1_dpx1_pp1/"
    trace_dir = "./my_test/1_gpt_tpx1_cpx1_epx1_dpx1_pp1/"   
    code_trace_dirs.append([code_dir, trace_dir] )
    
    code_dir = "./my_code/2_gpt_tpx8_cpx1_epx1_dpx1_pp1/"
    trace_dir = "./my_test/2_gpt_tpx8_cpx1_epx1_dpx1_pp1/"    
    code_trace_dirs.append([code_dir, trace_dir] )
    
    code_dir = "./my_code/3_gpt_tpx2_cpx1_epx4_dpx4_pp1_fp32/"
    trace_dir = "./my_test/3_gpt_tpx2_cpx1_epx4_dpx4_pp1_fp32/"    
    code_trace_dirs.append([code_dir, trace_dir] )
    
    code_dir = "./my_code/4_gpt_tpx2_cpx4_epx1_dpx1_pp1/"
    trace_dir = "./my_test/4_gpt_tpx2_cpx4_epx1_dpx1_pp1/"    
    code_trace_dirs.append([code_dir, trace_dir] )

    code_dir = "./my_code/5_gpt_tpx2_cpx1_epx1_dpx1_pp4/"
    trace_dir = "./my_test/5_gpt_tpx2_cpx1_epx1_dpx1_pp4/"    
    code_trace_dirs.append([code_dir, trace_dir] )

    code_dir = "./my_code/6_gpt_tpx2_cpx1_epx1_dpx4_pp1_ckpt_fps/"
    trace_dir = "./my_test/6_gpt_tpx2_cpx1_epx1_dpx4_pp1_ckpt_fps/"    
    code_trace_dirs.append([code_dir, trace_dir] )

    for code_dir, trace_dir in code_trace_dirs:
        print(f"add trace info ...")
        print(f"code_dir={code_dir}")
        print(f"trace_dir={trace_dir}")
        trace_filename = os.path.join(trace_dir, "my_trace_rank0.csv")
        trace_info_prefix = "# trace_info "    
        with open(trace_filename,encoding='utf-8') as file_obj:
            reader_obj = csv.reader(file_obj)        
            # 跳过第一行表头
            next(reader_obj)        
            for row in reader_obj:                     
                code_filename = os.path.abspath(os.path.join(code_dir,row[1].strip(" ")))
                trace_index = int(row[0])
                line_number = int(row[2])
                # print(f"trace_index:{trace_index}, filename:{code_filename}, lineno:{line_number}, code:{row[7]}")
                if not os.path.exists(code_filename):
                    print(f"warning: can not find {code_filename}.")
                    continue
                add_trace_info(code_filename, line_number, trace_info_prefix, trace_index)       


