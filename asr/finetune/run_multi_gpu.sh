#!/bin/bash

# 多GPU训练启动脚本
# 使用accelerate框架进行分布式训练

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $NUM_GPUS 个GPU"

# 设置默认参数
MODEL_NAME="microsoft/Phi-4-multimodal-instruct"
DATA_PATH="data/catslu/train.csv"
KEYWORDS_DIR="data/catslu"
OUTPUT_DIR="asr/model/"
BATCH_SIZE=4
BATCH_SIZE_PER_GPU=1
EPOCHS=1
LEARNING_RATE=4.0e-5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME="$2"
            shift 2
            ;;
        --catslu_data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --keywords_dir)
            KEYWORDS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --batch_size_per_gpu)
            BATCH_SIZE_PER_GPU="$2"
            shift 2
            ;;
        --num_train_epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "训练参数:"
echo "模型: $MODEL_NAME"
echo "数据路径: $DATA_PATH"
echo "关键词目录: $KEYWORDS_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "批处理大小: $BATCH_SIZE"
echo "每GPU批处理大小: $BATCH_SIZE_PER_GPU"
echo "训练轮数: $EPOCHS"
echo "学习率: $LEARNING_RATE"

if [ $NUM_GPUS -gt 1 ]; then
    echo "使用 $NUM_GPUS 个GPU进行分布式训练"
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --use_deepspeed \
        finetune_speech_asr_keywords2.py \
        --model_name_or_path "$MODEL_NAME" \
        --catslu_data_path "$DATA_PATH" \
        --keywords_dir "$KEYWORDS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
        --num_train_epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --use_flash_attention
else
    echo "使用单GPU训练"
    python finetune_speech_asr_keywords2.py \
        --model_name_or_path "$MODEL_NAME" \
        --catslu_data_path "$DATA_PATH" \
        --keywords_dir "$KEYWORDS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
        --num_train_epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --use_flash_attention
fi 