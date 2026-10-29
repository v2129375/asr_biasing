#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 5（Keyword list 規模的影響）—— 步驟 2：不同提示條件下的 ASR 推論、評分與運算量測

只載入一次模型，依序在測試集上跑多個 keyword list 規模：
  orig     原 keyword list（data/catslu，與先前實驗相同）
  kw1000   data/keyword_scale/kw1000（原 list 完整保留，隨機補干擾詞到每領域 1000 個）
  kw1500 / kw2000 / kw2500 / kw3000   同上，每領域 1500 / 2000 / 2500 / 3000 個
提示詞格式、生成參數與輸出後處理都與 asr/phi4_keywords.py 相同，CER/KER 直接呼叫 asr/score.py 的
evaluate_asr，因此結果可以和先前實驗直接比較。

除了 CER、KER，另外記錄（回應「list 很大時模型能否執行、干擾是否嚴重」）：
  distractor_insertion_rate  輸出中出現「提示 list 裡有、但正確文本沒有」的詞的句子比例（干擾程度）
  prompt_tokens_mean         提示詞（含音訊 token）的平均長度
  latency_mean_s / rtf       每句平均推論時間 / 推論時間 ÷ 音訊長度
  peak_mem_gb                該條件下所有 GPU 中最大的顯存峰值
  n_oom                      顯存不足（OOM）而無法推論的句數；連續 OOM 達 --max_oom 次即停止該條件

用法（在專案根目錄執行，需先完成意圖辨識推論，取得含 predict 欄的結果檔）：
  python asr/phi4_keywords_scale.py \
      --input intent/exp/phi4_intent_result_finetuned.csv \
      --model_path asr/model/<動態關鍵詞微調模型> \
      --conditions orig kw1000 kw1500 kw2000 kw2500 kw3000 \
      --gpus 0,1

輸出（預設 asr/exp/keyword_scale/<模型資料夾名>_<domain_col>/）：
  <條件>.csv / <條件>.json   每句結果 / evaluate_asr 的評分
  summary.csv / summary.json  各條件的 CER、KER、干擾詞插入率、提示長度、時間、顯存、OOM
"""

import argparse
import gc
import json
import os
import re
import sys
import time

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import evaluate_asr, normalize_for_score  # noqa: E402  與 phi4_keywords.py 相同的評分

DOMAINS = ["video", "music", "city"]
BASE_INSTRUCTION = "Transcribe the audio clip into text."
KEYWORD_INSTRUCTION_TEMPLATE = "Transcribe the audio clip into text. Pay attention to these keywords: {keywords}"
CHINESE_NUMS = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七',
                '8': '八', '9': '九', '10': '十', '11': '十一', '12': '十二', '13': '十三', '14': '十四',
                '15': '十五', '16': '十六', '17': '十七', '18': '十八', '19': '十九', '20': '二十'}


def parse_args():
    p = argparse.ArgumentParser(description="Keyword list 規模實驗（口試意見 5）")
    p.add_argument("--input", default="intent/exp/tts_phi4_intent_result.csv",
                   help="意圖辨識結果 CSV（需含 path、manual_transcript、keyword、source 與 predict 欄）")
    p.add_argument("--domain_col", default="predict", choices=["predict", "source"],
                   help="選擇關鍵詞 list 依據的領域：predict＝意圖辨識預測（完整系統），source＝真實領域（只看 ASR）")
    p.add_argument("--model_path", required=True, help="ASR 模型（微調後資料夾，或 microsoft/Phi-4-multimodal-instruct）")
    p.add_argument("--base_model", default="microsoft/Phi-4-multimodal-instruct")
    p.add_argument("--conditions", nargs="+", default=["orig", "kw1000", "kw1500", "kw2000", "kw2500", "kw3000"])
    p.add_argument("--orig_dir", default="data/catslu", help="原 keyword list 資料夾（orig 條件）")
    p.add_argument("--scale_dir", default="data/keyword_scale", help="make_keyword_scale_lists.py 的輸出資料夾")
    p.add_argument("--gpus", default="0,1", help="使用的 GPU，例如 0 或 0,1")
    p.add_argument("--device_map", default="asr/finetune/device_map.json", help="多 GPU 時的模型分配")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                   help="attention 實作；RTX 5080 等不支援現成 flash_attn 的顯卡請用 sdpa")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--max_oom", type=int, default=5, help="連續 OOM 達此次數即停止該條件")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 句（除錯用，0＝全部）")
    p.add_argument("--out_dir", default="", help="輸出資料夾（預設 asr/exp/keyword_scale/<模型名>_<domain_col>）")
    p.add_argument("--overwrite", action="store_true", help="重新推論已存在結果的條件")
    return p.parse_args()


def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_condition_lists(cond, args):
    """回傳 {domain: [keywords]}"""
    folder = args.orig_dir if cond == "orig" else os.path.join(args.scale_dir, cond)
    lists = {}
    for d in DOMAINS:
        path = os.path.join(folder, f"keyword_{d}.txt")
        if not os.path.exists(path):
            raise SystemExit(f"錯誤：找不到 {path}，請先執行 data/make_keyword_scale_lists.py")
        lists[d] = read_list(path)  # orig 保留原檔順序與重複，與 phi4_keywords.py 完全一致
    return lists


def to_domain(intent):
    """與 phi4_keywords.py 的 get_domain_keywords 對應規則相同"""
    s = str(intent).lower()
    for d in DOMAINS:
        if d in s:
            return d
    return "video"


def build_instruction(cond, domain, lists):
    words = lists.get(domain, [])
    if not words:
        return BASE_INSTRUCTION
    return KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=" ".join(words))


def postprocess(text):
    """與 phi4_keywords.py 相同：去標點、阿拉伯數字轉中文"""
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\d+", lambda m: CHINESE_NUMS.get(m.group(0), m.group(0)), text)


def distractor_inserted(hyp, ref, words):
    """輸出中是否出現「list 裡有、正確文本沒有」的詞（長度 ≥ 2）"""
    h, r = normalize_for_score(hyp), normalize_for_score(ref)
    if not h:
        return False
    return any(len(w) >= 2 and w in h and w not in r for w in words)


class Runner:
    def __init__(self, args):
        self.args = args
        self.gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
        self.main_device = f"cuda:{self.gpu_ids[0]}"
        if len(self.gpu_ids) > 1:
            with open(args.device_map) as f:
                device_map = json.load(f)
        else:
            device_map = self.main_device
        self.processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        print(f"載入模型 {args.model_path}（attention={args.attn}）...")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation=args.attn,
        )
        self.model.eval()
        self.generation_config = GenerationConfig.from_pretrained(args.model_path)

    def sync(self):
        for g in self.gpu_ids:
            torch.cuda.synchronize(g)

    def reset_peak(self):
        for g in self.gpu_ids:
            torch.cuda.reset_peak_memory_stats(g)

    def peak_gb(self):
        return max(torch.cuda.max_memory_allocated(g) for g in self.gpu_ids) / 1024 ** 3

    def free(self):
        gc.collect()
        for g in self.gpu_ids:
            with torch.cuda.device(g):
                torch.cuda.empty_cache()

    @torch.no_grad()
    def transcribe(self, audio, sr, instruction):
        prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|audio_1|>\n" + instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[prompt], audios=[(audio, sr)], return_tensors="pt").to(self.main_device)
        n_tokens = int(inputs["input_ids"].shape[1])
        self.sync()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                num_logits_to_keep=1,
                generation_config=self.generation_config,
            )
        self.sync()
        latency = time.perf_counter() - t0
        ids = ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        del inputs, ids
        return text, n_tokens, latency


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def summarize(cond, df, lists, eval_res, peak_gb, n_cond_list):
    ok = df[~df["oom"]]
    row = {
        "condition": cond,
        "list_size_video": n_cond_list.get("video"),
        "list_size_music": n_cond_list.get("music"),
        "list_size_city": n_cond_list.get("city"),
        "n_utts": len(df),
        "n_oom": int(df["oom"].sum()),
        "cer": eval_res["overall"].get("cer"),
        "ker": eval_res["overall"].get("keyword_wer"),
        "distractor_insertion_rate": None,
        "prompt_tokens_mean": round(ok["prompt_tokens"].mean(), 1) if len(ok) else None,
        "prompt_tokens_max": int(ok["prompt_tokens"].max()) if len(ok) else None,
        "latency_mean_s": round(ok["latency_s"].mean(), 4) if len(ok) else None,
        "rtf": round(ok["latency_s"].sum() / ok["audio_s"].sum(), 4) if len(ok) else None,
        "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else None,
    }
    for d in DOMAINS:
        row[f"cer_{d}"] = eval_res.get(d, {}).get("cer")
        row[f"ker_{d}"] = eval_res.get(d, {}).get("keyword_wer")
    if lists is not None:
        row["distractor_insertion_rate"] = round(
            df.apply(lambda r: distractor_inserted(r["asr"], r["manual_transcript"], lists[r["domain_used"]]),
                     axis=1).mean(), 4)
    return row


def main():
    args = parse_args()
    df0 = pd.read_csv(args.input)
    need = {"path", "manual_transcript", "keyword", "source", args.domain_col}
    if not need.issubset(df0.columns):
        raise SystemExit(f"錯誤：{args.input} 需包含欄位 {sorted(need)}")
    if args.limit:
        df0 = df0.head(args.limit)
    df0 = df0.reset_index(drop=True)
    df0["domain_used"] = df0[args.domain_col].map(to_domain)

    name = os.path.basename(os.path.normpath(args.model_path))
    out_dir = args.out_dir or os.path.join("asr", "exp", "keyword_scale", f"{name}_{args.domain_col}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"測試資料 {args.input}：{len(df0)} 句；關鍵詞依 {args.domain_col} 欄選擇；輸出到 {out_dir}")

    # 預先讀音訊（每個條件共用）
    audios = []
    for p in tqdm(df0["path"], desc="讀取音訊"):
        a, sr = sf.read(p)
        if a.ndim > 1:
            a = a.mean(axis=1)
        audios.append((a, sr))

    runner = None
    summary = []
    for cond in args.conditions:
        lists = load_condition_lists(cond, args)
        n_cond_list = {d: len(lists[d]) for d in DOMAINS}
        csv_path = os.path.join(out_dir, f"{cond}.csv")
        json_path = os.path.join(out_dir, f"{cond}.json")
        peak = None

        if os.path.exists(csv_path) and not args.overwrite:
            print(f"\n[{cond}] 已有結果，直接重新評分（加 --overwrite 可重跑推論）")
            df = pd.read_csv(csv_path, keep_default_na=False)
            df["oom"] = df["oom"].astype(str).str.lower() == "true"
            meta_path = os.path.join(out_dir, f"{cond}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    peak = json.load(f).get("peak_mem_gb")
        else:
            if runner is None:
                runner = Runner(args)
            print(f"\n[{cond}] list 大小：{n_cond_list}")
            df = df0.copy()
            df["asr"], df["prompt_tokens"], df["latency_s"], df["audio_s"], df["oom"] = "", 0, 0.0, 0.0, False
            runner.free()
            runner.reset_peak()
            consecutive_oom = 0
            for i in tqdm(range(len(df)), desc=cond):
                audio, sr = audios[i]
                df.at[i, "audio_s"] = len(audio) / sr
                if consecutive_oom >= args.max_oom:
                    df.at[i, "oom"] = True
                    continue
                instruction = build_instruction(cond, df.at[i, "domain_used"], lists)
                try:
                    text, n_tok, lat = runner.transcribe(audio, sr, instruction)
                    df.at[i, "asr"] = postprocess(text)
                    df.at[i, "prompt_tokens"] = n_tok
                    df.at[i, "latency_s"] = lat
                    consecutive_oom = 0
                except Exception as e:  # noqa: BLE001
                    if not is_oom(e):
                        raise
                    df.at[i, "oom"] = True
                    consecutive_oom += 1
                    runner.free()
                    if consecutive_oom == args.max_oom:
                        print(f"\n[{cond}] 連續 {args.max_oom} 次 OOM，停止此條件（視為超過運算上限）")
            peak = runner.peak_gb()
            df.to_csv(csv_path, index=False)
            with open(os.path.join(out_dir, f"{cond}_meta.json"), "w") as f:
                json.dump({"peak_mem_gb": peak, "list_sizes": n_cond_list}, f, ensure_ascii=False, indent=2)

        # OOM 的句子輸出為空字串，評分時算錯（CER=1、關鍵詞錯誤），反映「無法執行」
        eval_res = evaluate_asr(df.reset_index(drop=True), cal_keyword_wer=True, print_errors=False,
                                output_file=json_path)
        summary.append(summarize(cond, df, lists, eval_res, peak, n_cond_list))

    if runner is not None:
        runner.free()
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    cols = ["condition", "list_size_video", "list_size_music", "list_size_city", "cer", "ker",
            "distractor_insertion_rate", "prompt_tokens_mean", "latency_mean_s", "rtf", "peak_mem_gb", "n_oom"]
    print("\n=== 結果彙整 ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(sm[cols].to_string(index=False))
    print(f"\n已輸出到 {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
