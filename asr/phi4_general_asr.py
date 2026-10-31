#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 8（災難性遺忘）—— 步驟 2：在通用語音辨識測試集上推論與評分

同一支程式分別跑「原始 Phi-4」與「微調後模型」，比較微調前後的通用辨識能力。
提示詞一律用原始 ASR 指令 "Transcribe the audio clip into text."（不給關鍵詞），
也就是一般使用者使用模型的方式；提示詞格式與生成方式與 asr/phi4_keywords_scale.py 相同
（直接匯入它的 Runner）。

測試集（--datasets，可任選）：
  test-clean   LibriSpeech test-clean（英文，2620 句）→ WER
  test-other   LibriSpeech test-other（英文、較難，2939 句）→ WER
  aishell      AISHELL-1 test（中文通用，7176 句；檔案 data/aishell/test.csv 已在專案中）→ CER

評分：
  英文：Whisper 的 EnglishTextNormalizer（轉小寫、去標點、數字與縮寫統一，例如 21→twenty one 的差異、
        Mr.→mister），再算 corpus-level WER 與替換／刪除／插入比例。這是 LibriSpeech 文獻常用的做法。
  中文：與 CATSLU 實驗相同（去標點、0–20 阿拉伯數字轉中文、asr/score.py 的 normalize_for_score），算 CER。
另外記錄「語言漂移」：英文測試集輸出中含中文字的句子比例（微調只用中文資料，最可能出現的遺忘現象）。

用法（在專案根目錄執行）：
  python asr/phi4_general_asr.py --model_path microsoft/Phi-4-multimodal-instruct
  python asr/phi4_general_asr.py --model_path asr/model/KworigKeywords
  # 先試跑：--limit 20

輸出：asr/exp/forgetting/<模型資料夾名>/
  <測試集>.csv      每句：ref、hyp、正規化後文字、錯誤數（S/D/I/N）、是否含中文、推論時間
  summary.csv       各測試集的 WER/CER、S/D/I、整句錯誤率、語言漂移比例、推論時間
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import normalize_for_score  # noqa: E402
from phi4_keywords_scale import BASE_INSTRUCTION, Runner, is_oom, postprocess  # noqa: E402

DATASETS = {
    "test-clean": ("data/librispeech/test-clean.csv", "en"),
    "test-other": ("data/librispeech/test-other.csv", "en"),
    "aishell": ("data/aishell/test.csv", "zh"),
}
CJK = re.compile(r"[一-鿿]")


def parse_args():
    p = argparse.ArgumentParser(description="通用 ASR 測試（災難性遺忘）")
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default="microsoft/Phi-4-multimodal-instruct")
    p.add_argument("--datasets", nargs="+", default=["test-clean", "test-other", "aishell"],
                   choices=list(DATASETS))
    p.add_argument("--gpus", default="0")
    p.add_argument("--device_map", default="asr/finetune/device_map.json")
    p.add_argument("--attn", default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--max_new_tokens", type=int, default=256, help="LibriSpeech 最長約 35 秒，需較長的生成上限")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 句（除錯用）")
    p.add_argument("--out_dir", default="")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


_en_norm = None


def norm_en(text):
    global _en_norm
    if _en_norm is None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        _en_norm = EnglishTextNormalizer({})
    return _en_norm(str(text))


def edit_ops(ref, hyp):
    """回傳 (S, D, I)：ref → hyp 的最少替換、刪除、插入數"""
    n, m = len(ref), len(hyp)
    # 每格存 (總距離, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            s, d, ins = prev[j - 1], prev[j], cur[j - 1]
            cur[j] = min((s[0] + 1, s[1] + 1, s[2], s[3]),
                         (d[0] + 1, d[1], d[2] + 1, d[3]),
                         (ins[0] + 1, ins[1], ins[2], ins[3] + 1))
        prev = cur
    return prev[m][1:]


def units(text, lang):
    if lang == "en":
        return norm_en(text).split()
    return list(normalize_for_score(text))


def score_df(df, lang):
    refs = [units(t, lang) for t in df["manual_transcript"]]
    hyps = [units(t, lang) for t in df["hyp"]]
    ops = [edit_ops(r, h) for r, h in zip(refs, hyps)]
    df["ref_norm"] = [" ".join(r) if lang == "en" else "".join(r) for r in refs]
    df["hyp_norm"] = [" ".join(h) if lang == "en" else "".join(h) for h in hyps]
    df["S"], df["D"], df["I"] = zip(*ops) if ops else ([], [], [])
    df["N"] = [len(r) for r in refs]
    df["errors"] = df["S"] + df["D"] + df["I"]
    df["has_cjk"] = df["hyp"].astype(str).map(lambda s: bool(CJK.search(s)))
    df["has_latin"] = df["hyp"].astype(str).map(lambda s: bool(re.search(r"[A-Za-z]", s)))
    return df


def summarize(name, lang, df, peak):
    N = df["N"].sum()
    row = {"dataset": name, "metric": "WER" if lang == "en" else "CER", "n_utts": len(df),
           "n_oom": int(df["oom"].sum()),
           "error_rate": round(df["errors"].sum() / N, 5),
           "sub_rate": round(df["S"].sum() / N, 5), "del_rate": round(df["D"].sum() / N, 5),
           "ins_rate": round(df["I"].sum() / N, 5),
           "sentence_error_rate": round((df["errors"] > 0).mean(), 4),
           "empty_output": int((df["hyp_norm"] == "").sum())}
    if lang == "en":
        row["cjk_output_rate"] = round(df["has_cjk"].mean(), 4)  # 英文音訊卻輸出中文 → 語言漂移
    else:
        row["latin_output_rate"] = round(df["has_latin"].mean(), 4)
    ok = df[~df["oom"]]
    row["latency_mean_s"] = round(ok["latency_s"].mean(), 4) if len(ok) else None
    row["rtf"] = round(ok["latency_s"].sum() / ok["audio_s"].sum(), 4) if len(ok) else None
    row["peak_mem_gb"] = round(peak, 2) if peak is not None else None
    return row


def main():
    args = parse_args()
    name = os.path.basename(os.path.normpath(args.model_path))
    out_dir = args.out_dir or os.path.join("asr", "exp", "forgetting", name)
    os.makedirs(out_dir, exist_ok=True)
    runner, summary = None, []
    for ds in args.datasets:
        csv_in, lang = DATASETS[ds]
        if not os.path.exists(csv_in):
            raise SystemExit(f"錯誤：找不到 {csv_in}（LibriSpeech 請先執行 data/prepare_librispeech.py）")
        csv_out = os.path.join(out_dir, f"{ds}.csv")
        meta = os.path.join(out_dir, f"{ds}_meta.json")
        peak = None
        if os.path.exists(csv_out) and not args.overwrite:
            print(f"\n[{ds}] 已有結果，直接重新評分（加 --overwrite 可重跑）")
            df = pd.read_csv(csv_out, keep_default_na=False)
            df["oom"] = df["oom"].astype(str).str.lower() == "true"
            if os.path.exists(meta):
                peak = json.load(open(meta)).get("peak_mem_gb")
        else:
            df = pd.read_csv(csv_in, keep_default_na=False)
            if args.limit:
                df = df.head(args.limit)
            df = df.reset_index(drop=True)
            if runner is None:
                runner = Runner(args)
            df["hyp"], df["latency_s"], df["audio_s"], df["oom"] = "", 0.0, 0.0, False
            runner.free()
            runner.reset_peak()
            for i in tqdm(range(len(df)), desc=ds):
                audio, sr = sf.read(df.at[i, "path"])
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                df.at[i, "audio_s"] = len(audio) / sr
                try:
                    text, _, lat = runner.transcribe(audio, sr, BASE_INSTRUCTION)
                except Exception as e:  # noqa: BLE001
                    if not is_oom(e):
                        print(f"\n[{ds}] 第 {i} 句錯誤，以空輸出計分：{df.at[i, 'path']}  {type(e).__name__}: {e}")
                    else:
                        runner.free()
                    df.at[i, "oom"] = True
                    continue
                # 中文沿用 CATSLU 的後處理；英文保留原文，交給 Whisper 正規化
                df.at[i, "hyp"] = postprocess(text) if lang == "zh" else text.strip()
                df.at[i, "latency_s"] = lat
            peak = runner.peak_gb()
            json.dump({"peak_mem_gb": peak}, open(meta, "w"))
        df = score_df(df, lang)
        df.to_csv(csv_out, index=False)
        row = summarize(ds, lang, df, peak)
        summary.append(row)
        print(f"[{ds}] {row['metric']} = {row['error_rate'] * 100:.2f}%")
    # 與已存在的 summary 合併（分次跑不同測試集時不會互相覆蓋）
    sm_path = os.path.join(out_dir, "summary.csv")
    sm = pd.DataFrame(summary)
    if os.path.exists(sm_path):
        old = pd.read_csv(sm_path)
        sm = pd.concat([old[~old["dataset"].isin(sm["dataset"])], sm], ignore_index=True)
    sm.to_csv(sm_path, index=False)
    print("\n=== 結果彙整 ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(sm.to_string(index=False))
    print(f"\n已輸出到 {sm_path}")


if __name__ == "__main__":
    main()
