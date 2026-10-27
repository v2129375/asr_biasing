#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口试意见 4（实验五变因控制）—— 第 3 步：检查真实训练集与合成训练集是否一致

比较项目：句数、各领域句数、关键词是否逐句一致、总时数、平均每句秒数、语速（字/秒）、
与真实文本完全相同的比例。可同时放入旧版合成资料做对照，结果可直接整理进论文表格。

用法（在 repo 根目录执行）：
  python tts/compare_real_tts_stats.py \
      --real_csv data/catslu/train.csv \
      --syn_csv tts/tts_data/sentences_audio_matched.csv tts/tts_data/sentences_audio_diff_ref.csv \
      --save tts/tts_data/real_vs_tts_stats.csv
"""

import argparse
import re
from collections import Counter

import pandas as pd
import soundfile as sf

NON_CHINESE_RE = re.compile(r"[^一-鿿]")


def to_label(text):
    return NON_CHINESE_RE.sub("", str(text))


def duration(path):
    try:
        return sf.info(path).duration
    except Exception:  # noqa: BLE001
        return None


def summarize(name, df, real_df=None):
    df = df.copy()
    if "tts_duration" in df.columns:
        df["dur"] = df["tts_duration"]
    else:
        df["dur"] = df["path"].map(duration)
    missing = int(df["dur"].isna().sum())
    d = df.dropna(subset=["dur"])
    n_chars = d["manual_transcript"].map(lambda t: len(to_label(t))).sum()
    total_sec = d["dur"].sum()

    info = {
        "dataset": name,
        "n_utts": len(df),
        "n_video": int((df["source"] == "video").sum()),
        "n_music": int((df["source"] == "music").sum()),
        "n_city": int((df["source"] == "city").sum()),
        "n_unique_keywords": df["keyword"].nunique() if "keyword" in df.columns else None,
        "total_hours": round(total_sec / 3600, 3),
        "mean_sec_per_utt": round(d["dur"].mean(), 3) if len(d) else None,
        "chars_per_sec": round(n_chars / total_sec, 3) if total_sec else None,
        "audio_missing": missing,
    }
    if "keyword" in df.columns:
        info["keyword_in_text_rate"] = round(
            df.apply(lambda r: str(r["keyword"]) in str(r["manual_transcript"]), axis=1).mean(), 4)
    if real_df is not None and "keyword" in df.columns:
        # 关键词「多重集合」是否与真实训练集完全一致（即逐句关键词相同、出现次数相同）
        info["keyword_multiset_equal_real"] = Counter(df["keyword"]) == Counter(real_df["keyword"])
        real_texts = set(real_df["manual_transcript"].map(to_label))
        info["text_identical_to_real_rate"] = round(
            df["manual_transcript"].map(lambda t: to_label(t) in real_texts).mean(), 4)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_csv", default="data/catslu/train.csv")
    p.add_argument("--syn_csv", nargs="+", default=["tts/tts_data/sentences_audio_matched.csv"])
    p.add_argument("--save", default="", help="将结果另存为 CSV（可选）")
    args = p.parse_args()

    real = pd.read_csv(args.real_csv)
    results = [summarize(f"real:{args.real_csv}", real)]
    for path in args.syn_csv:
        results.append(summarize(f"tts:{path}", pd.read_csv(path), real))

    table = pd.DataFrame(results).set_index("dataset").T
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(table.to_string())
    if args.save:
        table.to_csv(args.save)
        print(f"\n已保存：{args.save}")


if __name__ == "__main__":
    main()
