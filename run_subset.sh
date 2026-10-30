#!/bin/bash
cd /home/v2129375/asr_biasing
mkdir -p logs
for N in 100 200; do
  OUT=asr/model/Sub${N}Keywords
  if ! ls $OUT/*.safetensors >/dev/null 2>&1; then
    echo "[训练] $OUT $(date)"
    python asr/finetune/finetune_catslu_keywords_subset.py \
        --data_path data/catslu/train.csv --keywords_dir data/catslu \
        --num_keywords $N --output_dir $OUT --gpu_ids 0 2>&1 | tee logs/train_sub${N}.log
    ls $OUT/*.safetensors >/dev/null 2>&1 || { echo "[错误] $OUT 训练失败"; exit 1; }
  fi
  echo "[测试] $OUT $(date)"
  python asr/phi4_keywords_scale.py --input data/catslu/test.csv --domain_col source \
      --model_path $OUT --conditions orig --gpus 0 --attn flash_attention_2 \
      2>&1 | tee logs/test_sub${N}.log
done
echo "全部完成 $(date)"