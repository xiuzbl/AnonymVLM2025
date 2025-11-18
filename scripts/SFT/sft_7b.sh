#!/bin/bash
set -x

# ========= Environment Setup ========= #
WORLD_SIZE=8
ENTRY_FILE="sft.py"

export MODEL_BASE_PATH="TODO" # setup your base path to models and checkpoints here
JOB_RUN_NAME="Adaptive-v2-Qwen25VL-7B-SFT-vocot-orsta32b-distilled-1epochs_lr5e-6"
job_name=${JOB_RUN_NAME}

export MODEL_NAME="Qwen2.5-VL-7B-Instruct"

# ========= Model Path Settings ========= #
export MODEL_LOAD_PATH=${MODEL_BASE_PATH}/models/${MODEL_NAME}
export MODEL_SAVE_PATH=${MODEL_BASE_PATH}/checkpoints/sft/vocot_orsrta_distilled/${MODEL_NAME}_adaptive_vocot-orsta32b-distilled-1epochs_lr5e-6

# ========= Data Path Settings ========= #
export DATA_CONFIG="config/datasets/sft/sft_grd-vocot_txt-orsta-32b-distilled.yaml"


# ========= Run Training ========= #
args="
    --model_name_or_path ${MODEL_LOAD_PATH} \
    --data_config_path ${DATA_CONFIG} \
    --output_dir ${MODEL_SAVE_PATH} \
    --report_to tensorboard \
    --gradient_checkpointing \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --per_device_eval_batch_size 4 \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 5 \
    --attn_implementation flash_attention_2 \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --deepspeed config/deepspeed/zero3.json \
    --run_name ${JOB_RUN_NAME} \
    --freeze_visual_encoder \
    --max_wh_limit 2280 \
    --max_seq_length 4096 \
    --bf16
"

torchrun --nproc_per_node ${WORLD_SIZE} ${ENTRY_FILE} ${args}