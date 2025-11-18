# Parameter setup
export CUDA_VISIBLE_DEVICES=$1
IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPUS="${#gpu_array[@]}"
export NUM_TRAIN_GPUS=$(($NUM_GPUS))
export MASTER_PORT=8810

export MODEL_PATH=$2 # "/mnt/public/Fudan/share/checkpoints/explicit-vocot/Qwen2.5-VL-3B-Instruct_debug/"
export STORE_NAME=$3 # "Qwen25-VL-3B-vocot-debug"
export MODE=$4 # "vocot"
export PROMPT_MODE=$5
export BASE_MODEL=$6
export BASE_MODEL_DIR=$7
export ADD_ARGS=$8
export MAX_NEW_TOKENS=$9
export EVAL_TASKS=${10}
export ADD_ARGS_EXP=${11}

echo "Tasks to evaluate: ${EVAL_TASKS}"

if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"wemath"* ]]; then
    # MathVision
    echo "running task wemath"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/WeMath.yaml \
    --output_dir output/math/${STORE_NAME}/math/ \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache --no_barrier  --cot_method ${PROMPT_MODE}  ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally provide the correct answer using the correct option letter, e.g., A, B, C, D, at the end."

    python3 eval/merge_benchmark.py --config_arg output/math/${STORE_NAME}/math/WeMath.args.bin
fi

if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"mathvista"* ]]; then
    # MathVista
    echo "running task mathvista"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/MathVista.yaml \
    --output_dir output/math/${STORE_NAME}/math/ \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache --no_barrier  --cot_method ${PROMPT_MODE}  ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally output the answer according to the question's requirements."

    python3 eval/merge_benchmark.py --config_arg output/math/${STORE_NAME}/math/MathVista.args.bin
fi


if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"mathvision"* ]]; then
    # MathVision
    echo "running task mathvision"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/MathVision.yaml \
    --output_dir output/math/${STORE_NAME}/math/ \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache --no_barrier  --cot_method ${PROMPT_MODE}  ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally output the answer according to the question's requirements."

    python3 eval/merge_benchmark.py --config_arg output/math/${STORE_NAME}/math/MathVision.args.bin
fi

if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"mathverse"* ]]; then
    # MathVision
    echo "running task mathverse"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/MathVerse.yaml \
    --output_dir output/math/${STORE_NAME}/math/ \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache --no_barrier  --cot_method ${PROMPT_MODE}  ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally output the answer according to the question's requirements."

    python3 eval/merge_benchmark.py --config_arg output/math/${STORE_NAME}/math/MathVerse.args.bin
fi

if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"pope"* ]]; then
    # POPE
    echo "running task pope"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/POPE_all.yaml \
    --output_dir output/hallucination/${STORE_NAME}/binary_instruct/complete_prompt_dynamic \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache --no_barrier  --cot_method ${PROMPT_MODE}  ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally answer the question with yes or no."

    python3 eval/merge_benchmark.py --config_arg output/hallucination/${STORE_NAME}/binary_instruct/complete_prompt_dynamic/POPE_all.args.bin
fi

if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"spatialscore"* ]]; then
    # SpatialScore
    echo "running task spatialscore"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/SpatialScore_hard.yaml \
    --output_dir output/spatial/${STORE_NAME}/complete_prompt_dynamic/ \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache --no_barrier  --cot_method ${PROMPT_MODE}  ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally output the answer according to the question's requirements."

    python3 eval/merge_benchmark.py --config_arg output/spatial/${STORE_NAME}/complete_prompt_dynamic/SpatialScore_hard.args.bin
fi

if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"vstar"* ]]; then
    # VStar
    echo "running task vstar"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/VStar.yaml \
    --output_dir output/vstar/${STORE_NAME}/option_instruct/complete_prompt_dynamic \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache  --no_barrier  --cot_method ${PROMPT_MODE} ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally provide the correct answer using the correct option letter, e.g., A, B, C, D, at the end."

    python3 eval/merge_benchmark.py --config_arg output/vstar/${STORE_NAME}/option_instruct/complete_prompt_dynamic/VStar.args.bin
fi

if [[ "$EVAL_TASKS" == "all" || "$EVAL_TASKS" == *"mmstar"* ]]; then
    # MMStar
    echo "running task mmstar"
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    evaluate_benchmark_qwen.py \
    --model_path ${MODEL_PATH} --base_model ${BASE_MODEL} --base_model_dir ${BASE_MODEL_DIR} \
    --eval_data config/datasets/eval/conversation_format_tb/MMStar.yaml \
    --output_dir output/mmstar/${STORE_NAME}/option_instruct/complete_prompt_dynamic \
    --temperature 0 --precision bf16 --model_type ${MODE} --use_cache --no_barrier  --cot_method ${PROMPT_MODE}  ${ADD_ARGS}  ${ADD_ARGS_EXP} \
    --max_new_tokens ${MAX_NEW_TOKENS} --additional_prompt "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally provide the correct answer using the correct option letter, e.g., A, B, C, D, at the end."

    python3 eval/merge_benchmark.py --config_arg output/mmstar/${STORE_NAME}/option_instruct/complete_prompt_dynamic/MMStar.args.bin
fi
