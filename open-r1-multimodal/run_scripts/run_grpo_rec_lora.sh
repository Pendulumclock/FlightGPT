export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3

RUN_NAME="FlightGPT"
CURRENT_DIR=$(pwd)
export LOG_PATH="$CURRENT_DIR/experiment/debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    $CURRENT_DIR/open-r1-multimodal/src/open_r1/grpo_jsonl_citynav.py \
    --deepspeed $CURRENT_DIR/open-r1-multimodal/local_scripts/zero2.json \
    --output_dir $CURRENT_DIR/experiment/$RUN_NAME \
    --model_name_or_path $CURRENT_DIR/model_weight/Qwen2.5-VL-7B-Instruct \
    --dataset_name $CURRENT_DIR/data/training_data/citynav_train_data.json \
    --image_folders $CURRENT_DIR/data/training_data/images \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true


