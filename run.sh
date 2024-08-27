#!/bin/bash


# 获取脚本的绝对路径
SCRIPT_PATH=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# 修改默认的 python pyc 文件的存放地址
export PYTHONPYCACHEPREFIX=${SCRIPT_PATH}/../.cache/cpython/

TRACEFUNCS_PATH=${SCRIPT_PATH}/trace_funcs/
export PYTHONPATH=$TRACEFUNCS_PATH:$PYTHONPATH

# 需要下载megatron-lm 并checkout到 core_v0.7.0
MEGATRON_PATH=${SCRIPT_PATH}/../Megatron-LM_core_v0.7.0/
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

# 需要下载数据集
OPENWEBTXT_DATA_PATH=${SCRIPT_PATH}/../dataset/openwebtxt_275M/


cp ${SCRIPT_PATH}/pretrain_gpt/pretrain_gpt_trace_cuda_memory.py ${MEGATRON_PATH}/pretrain_gpt_trace_cuda_memory.py
PY_TRACE_MEM=${MEGATRON_PATH}/pretrain_gpt_trace_cuda_memory.py
LOG_TRACE_MEM=pretrain_gpt_trace_cuda_memory.log

cp ${SCRIPT_PATH}/pretrain_gpt/pretrain_gpt_trace_torch_dispatch.py ${MEGATRON_PATH}/pretrain_gpt_trace_torch_dispatch.py
PY_TRACE_DISPATCH=${MEGATRON_PATH}/pretrain_gpt_trace_torch_dispatch.py
LOG_TRACE_DISPATCH=pretrain_gpt_trace_torch_dispatch.log


#!/bin/bash
# 记录当前目录
before_for_dir=$(pwd)
# 设置要遍历的目录
base_dir=$(pwd)/my_test

# # 遍历目录中的所有子目录
# for sub_dir in "$base_dir"/*/; do
#     # 确保是目录
#     if [ -d "$sub_dir" ]; then
#         echo "Processing directory: $sub_dir"
        
#         # 进入子目录
#         cd "$sub_dir" || { echo "Failed to enter directory: $sub_dir"; exit 1; }
        
#         # 在子目录中执行操作
#         echo "Current directory is $(pwd)"
#         rm *.csv
#         rm *.log
#         bash train_gpt.sh  ${OPENWEBTXT_DATA_PATH}  ${PY_TRACE_MEM}  ${LOG_TRACE_MEM}
#         # bash train_gpt.sh  ${OPENWEBTXT_DATA_PATH}  ${PY_TRACE_DISPATCH}  ${LOG_TRACE_DISPATCH}     
        
#         # 返回到上级目录
#         cd "$base_dir" || { echo "Failed to return to directory: $base_dir"; exit 1; }
#     fi
# done

# # for 循环结束之后返回到之前记录的 before_for_dir
# cd "$before_for_dir" || { echo "Failed to return to directory: $before_for_dir"; exit 1; }
# echo "Current directory is $(pwd)"
# python trace_funcs/add_trace_info.py

cd my_test

cd 2_gpt_tpx8_cpx1_epx1_dpx1_pp1/
# 在子目录中执行操作
echo "Current directory is $(pwd)"
rm *.csv
rm *.log
bash train_gpt.sh  ${OPENWEBTXT_DATA_PATH}  ${PY_TRACE_MEM}  ${LOG_TRACE_MEM}
# bash train_gpt.sh  ${OPENWEBTXT_DATA_PATH}  ${PY_TRACE_DISPATCH}  ${LOG_TRACE_DISPATCH}     
cd ..

