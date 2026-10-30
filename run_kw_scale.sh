#!/bin/bash
mkdir -p logs
SIZES="orig 1000 1500 2000 2500 3000"

# 第 1 步：训练（已有模型的会跳过）
for N in $SIZES; do
  OUT=asr/model/Kw${N}Keywords
  if ls $OUT/*.safetensors >/dev/null 2>&1; then echo "[跳过] $OUT 已训练完成"; continue; fi
  if [ "$N" = orig ]; then KD=data/catslu; else KD=data/keyword_scale/kw$N; fi
  echo "[训练] $OUT  $(date)"
  python asr/finetune/finetune_catslu_keywords.py \
      --data_path data/catslu/train.csv --keywords_dir $KD \
      --output_dir $OUT --gpu_ids 0 2>&1 | tee logs/train_kw${N}.log
  if ! ls $OUT/*.safetensors >/dev/null 2>&1; then echo "[错误] $OUT 训练失败，停止"; exit 1; fi
done

# 第 3 步：每个模型在 6 种规模上测试
for N in $SIZES; do
  echo "[测试] Kw${N}Keywords  $(date)"
  python asr/phi4_keywords_scale.py --input data/catslu/test.csv --domain_col source \
      --model_path asr/model/Kw${N}Keywords --gpus 0 --attn flash_attention_2 \
      2>&1 | tee logs/test_kw${N}.log
done