#!/bin/bash
# Runs the "175B" parameter model

OPENWEBTXT_DATA_PATH=$1
PRETRAIN_PY_FILE=$2
LOG_NAME=$3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6510
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=./ #$1 #<Specify path>
TENSORBOARD_LOGS_PATH=./ #$2 #<Specify path>

VOCAB_FILE=${OPENWEBTXT_DATA_PATH}/gpt2-vocab.json #$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=${OPENWEBTXT_DATA_PATH}/gpt2-merges.txt #$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=${OPENWEBTXT_DATA_PATH}/openwebtxt/train #$5 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 2 
    --hidden-size 2048 
    --num-attention-heads 16 
    --seq-length 1024 
    --max-position-embeddings 1024 
)

    #--transformer-impl local
TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1
    --train-iters 3
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --use-mcore-models
    --initial-loss-scale 1024    
    --transformer-impl transformer_engine
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 1
    --context-parallel-size 4
    --use-flash-attn
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1000
    --save-interval 1000
    --eval-interval 1000
    --eval-iters   -1
)

torchrun ${DISTRIBUTED_ARGS[@]} ${PRETRAIN_PY_FILE} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    > ${LOG_NAME}  2>&1
