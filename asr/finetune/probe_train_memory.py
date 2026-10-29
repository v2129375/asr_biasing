#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 5：微調前的顯存實測（找出單張 GPU 可安全微調的最大 keyword list 規模）

在正式訓練前，用「最壞情況」的單一樣本實際跑一次 forward + backward + optimizer step，
量測顯存峰值；設定與 asr/finetune/finetune_catslu_keywords.py 相同：
  bf16、gradient checkpointing（use_reentrant=False）、speech LoRA、AdamW、batch_size_per_gpu=1。

最壞情況：
  - 領域：video（每個關鍵詞平均字數最多，提示詞最長）
  - 音訊：訓練集中最長的一句

比較兩種 loss 計算方式：
  full         原程式的做法：對整條序列每個位置都算 20 萬維詞表的 logits（再轉 float32 算 loss）
  answer_only  只對答案位置算 logits（num_logits_to_keep），labels 同步截取；
               被截掉的位置 label 本來就是 -100，不參與 loss，因此 loss 數值相同
程式會在同一規模下比較兩種方式的 loss 是否一致（固定亂數種子，學習率設 0，權重不會被改動）。

用法（在專案根目錄、phi4 環境執行；單張 GPU）：
  python asr/finetune/probe_train_memory.py --sizes orig 1000 1500 2000 2500 3000 --device cuda:0

判斷標準：peak_reserved_gb ≤ 顯存總量 × --safe_ratio（預設 0.85）視為「安全」。
"""

import argparse
import gc
import os
import sys

import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

KEYWORD_INSTRUCTION_TEMPLATE = "Transcribe the audio clip into text. Pay attention to these keywords: {keywords}"
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-4-multimodal-instruct")
    p.add_argument("--sizes", nargs="+", default=["orig", "1000", "1500", "2000", "2500", "3000"])
    p.add_argument("--orig_dir", default="data/catslu")
    p.add_argument("--scale_dir", default="data/keyword_scale")
    p.add_argument("--domain", default="video", choices=["video", "music", "city"])
    p.add_argument("--train_csv", default="data/catslu/train.csv", help="從中挑最長的一句音訊")
    p.add_argument("--modes", nargs="+", default=["answer_only", "full"], choices=["answer_only", "full"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--safe_ratio", type=float, default=0.85)
    p.add_argument("--out_csv", default="asr/exp/keyword_scale/train_memory_probe.csv")
    return p.parse_args()


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def main():
    args = parse_args()
    dev = torch.device(args.device)
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3

    # 最長的一句音訊
    df = pd.read_csv(args.train_csv)
    df["dur"] = df["path"].map(lambda p: sf.info(p).duration)
    row = df.sort_values("dur").iloc[-1]
    audio, sr = sf.read(row["path"])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    transcript = row["manual_transcript"]
    print(f"GPU 顯存 {total_gb:.1f} GB；最長音訊 {row['dur']:.2f} 秒：{transcript}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, _attn_implementation=args.attn,
        trust_remote_code=True, device_map=args.device,
    )
    model.set_lora_adapter("speech")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"可訓練參數 {n_train / 1e6:.1f} M；模型權重佔用 {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB")
    # 學習率 0：會建立 AdamW 狀態（佔顯存），但不改動權重，方便比較兩種 loss
    opt = torch.optim.AdamW(trainable, lr=0.0, betas=(0.9, 0.95), eps=1e-7, weight_decay=0.01)

    answer_ids = processor.tokenizer(f"{transcript}{ANSWER_SUFFIX}", return_tensors="pt").input_ids
    results = []
    for size in args.sizes:
        folder = args.orig_dir if size == "orig" else os.path.join(args.scale_dir, f"kw{size}")
        words = read_lines(os.path.join(folder, f"keyword_{args.domain}.txt"))
        instruction = KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=" ".join(words))
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|audio_1|>\n" + instruction}], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, audios=[(audio, sr)], return_tensors="pt")
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1]:] = answer_ids
        seq_len = input_ids.shape[1]

        losses = {}
        for mode in args.modes:
            batch = {
                "input_ids": input_ids.to(dev),
                "labels": labels.to(dev),
                "attention_mask": torch.ones_like(input_ids).to(dev),
                "input_audio_embeds": inputs.input_audio_embeds.to(dev),
                "audio_embed_sizes": inputs.audio_embed_sizes.to(dev),
                "audio_attention_mask": None,
                "input_mode": 2,
            }
            if mode == "answer_only":
                k = answer_ids.shape[1] + 1
                batch["labels"] = batch["labels"][:, -k:]
                batch["num_logits_to_keep"] = k
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            status, loss_val = "ok", None
            try:
                torch.manual_seed(0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                loss_val = float(out.loss.detach())
                del out
            except Exception as e:  # noqa: BLE001
                if not is_oom(e):
                    raise
                status = "OOM"
                opt.zero_grad(set_to_none=True)
            peak_alloc = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
            peak_res = torch.cuda.max_memory_reserved(dev) / 1024 ** 3
            safe = status == "ok" and peak_res <= total_gb * args.safe_ratio
            losses[mode] = loss_val
            results.append({
                "size": size, "domain": args.domain, "n_keywords": len(words), "seq_len": seq_len, "mode": mode,
                "status": status, "loss": loss_val, "peak_allocated_gb": round(peak_alloc, 2),
                "peak_reserved_gb": round(peak_res, 2), "safe": safe,
            })
            print(f"[{size:>5}] {mode:<11} 序列 {seq_len:>6} token  {status:<3}  "
                  f"峰值 allocated {peak_alloc:5.1f} GB / reserved {peak_res:5.1f} GB  "
                  f"{'安全' if safe else '不安全'}  loss={loss_val}")
            gc.collect()
            torch.cuda.empty_cache()
        if all(losses.get(m) is not None for m in ("answer_only", "full")):
            diff = abs(losses["answer_only"] - losses["full"])
            print(f"        兩種方式 loss 差異 {diff:.2e}（應接近 0）")

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n已輸出 {args.out_csv}")
    for mode in args.modes:
        ok = out[(out["mode"] == mode) & out["safe"]]
        best = ok.iloc[-1]["size"] if len(ok) else "（無）"
        print(f"{mode}：可安全微調的最大規模 = {best}")


if __name__ == "__main__":
    sys.exit(main())
