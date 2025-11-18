#!/bin/bash
set -x

# ========= Environment Setup ========= #
WORLD_SIZE=7
ENTRY_FILE=src/open_r1/grpo.py

export MODEL_BASE_PATH="/PATH/TO/MODELS" # setup your base path to models and checkpoints here
export MODEL_NAME="Qwen2.5-VL-7B-Instruct"
JOB_NAME="RL-Phase1-Adaptive-v2-${MODEL_NAME}-SFT-vocot-mc-orsta7b-distilled-1epochs_lr5e-6-AdaGRPO-binary-mix-max2048"

# ========= Model Path Settings ========= #
export MODEL_LOAD_PATH=${MODEL_BASE_PATH}/checkpoints/sft/vocot_orsrta_distilled/${MODEL_NAME}_adaptive_vocot-orsta32b-distilled-1epochs_lr5e-6 # path to the SFT checkpoint
export MODEL_SAVE_PATH=${MODEL_BASE_PATH}/checkpoints/rl/adaptive_r1/${JOB_NAME}

# ========= Data Path Settings ========= #
export DATA_CONFIG="config/datasets/rl/binary_mixture.yaml"

torchrun --nproc_per_node ${WORLD_SIZE} ${ENTRY_FILE} \
    --deepspeed config/deepspeed/zero3.json \
    --output_dir ${MODEL_SAVE_PATH} \
    --model_name_or_path ${MODEL_LOAD_PATH} \
    --dataset_name null \
    --data_config ${DATA_CONFIG} \
    --max_prompt_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --model_type adaptive_v2 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2073600 \
    --num_train_epochs 1 \
    --run_name $JOB_NAME \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8 \
    --max_completion_length 2048 \
    --use_cache_for_generation true \
    --rl_tasks "count,math" \
    --group_prefix "<text>,<grounding>" \
    --reward_funcs "adaptive,format" \
    --advantage_strategy "adaGRPO" \
    --vllm