#!/bin/bash
set -x

# ========= Environment Setup ========= #
WORLD_SIZE=8
GRD_ACC_STEPS=4 # keep the batch size = 32
ENTRY_FILE=src/open_r1/grpo.py

export MODEL_BASE_PATH="/PATH/TO/MODELS" # setup your base path to models and checkpoints here
export MODEL_NAME="Qwen2.5-VL-3B-Instruct" # change to Qwen2.5-VL-7B-Instruct for 7B-based experiments
JOB_NAME="RL-Phase2-Adaptive-v2-${MODEL_NAME}-SFT-vocot-mc-orsta7b-distilled-1epochs_lr5e-6_from-phase1_AdaGRPO-diverse-mix-max3072"

# ========= Model Path Settings ========= #
# export MODEL_LOAD_PATH=${MODEL_BASE_PATH}/checkpoints/sft/vocot_orsrta_distilled/${MODEL_NAME}_adaptive_vocot-orsta32b-distilled-1epochs_lr5e-6
export MODEL_LOAD_PATH=${MODEL_BASE_PATH}/checkpoints/rl/adaptive_r1/RL-Phase1-Adaptive-v2-${MODEL_NAME}-SFT-vocot-mc-orsta7b-distilled-1epochs_lr5e-6-AdaGRPO-binary-mix-max2048 # PATH to the binary-mix-trained model
export MODEL_SAVE_PATH=${MODEL_BASE_PATH}/checkpoints/rl/adaptive_r1/${JOB_NAME}

# ========= Data Path Settings ========= #
export DATA_CONFIG="config/datasets/rl/diverse_mixture.yaml"

torchrun --nproc_per_node ${WORLD_SIZE} ${ENTRY_FILE} \
    --deepspeed config/deepspeed/zero3.json \
    --output_dir ${MODEL_SAVE_PATH} \
    --model_name_or_path ${MODEL_LOAD_PATH} \
    --dataset_name null \
    --data_config ${DATA_CONFIG} \
    --max_prompt_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRD_ACC_STEPS} \
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
    --max_completion_length 3072 \
    --use_cache_for_generation true \
    --rl_tasks "TT_Math,TT_Chart,TT_Counting,TT_Science,TT_Detection,TT_Others,TT_Grounding,TT_Document,TT_OCR,TT_Puzzle" \
    --group_prefix "<text>,<grounding>" \
    --advantage_strategy "adaGRPO" \
    --reward_funcs "adaptive,format"