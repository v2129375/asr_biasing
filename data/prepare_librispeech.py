#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 8（災難性遺忘）—— 步驟 1：把 LibriSpeech 測試集整理成 CSV

LibriSpeech 的資料夾結構：<root>/<子集>/<說話者>/<章節>/<說話者>-<章節>.trans.txt
trans.txt 每行：<utt_id> <大寫英文轉寫>，音訊為同資料夾的 <utt_id>.flac

用法（在專案根目錄執行）：
  # 下載（OpenSLR 官方，各約 330 MB）
  mkdir -p /home/v2129375/dataset/librispeech && cd /home/v2129375/dataset/librispeech
  wget https://www.openslr.org/resources/12/test-clean.tar.gz
  wget https://www.openslr.org/resources/12/test-other.tar.gz
  tar xzf test-clean.tar.gz && tar xzf test-other.tar.gz      # 解壓後在 LibriSpeech/ 底下
  cd -
  python data/prepare_librispeech.py --root /home/v2129375/dataset/librispeech/LibriSpeech

輸出：data/librispeech/test-clean.csv、data/librispeech/test-other.csv
      欄位 path, manual_transcript, source（＝子集名稱）, speaker
"""

import argparse
import glob
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/v2129375/dataset/librispeech/LibriSpeech")
    p.add_argument("--subsets", nargs="+", default=["test-clean", "test-other"])
    p.add_argument("--out_dir", default="data/librispeech")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for sub in args.subsets:
        rows = []
        for tf in sorted(glob.glob(os.path.join(args.root, sub, "*", "*", "*.trans.txt"))):
            d = os.path.dirname(tf)
            with open(tf, encoding="utf-8") as f:
                for line in f:
                    uid, text = line.strip().split(" ", 1)
                    path = os.path.join(d, f"{uid}.flac")
                    if not os.path.exists(path):
                        print(f"警告：找不到 {path}")
                        continue
                    rows.append({"path": path, "manual_transcript": text, "source": sub,
                                 "speaker": uid.split("-")[0]})
        if not rows:
            raise SystemExit(f"錯誤：{os.path.join(args.root, sub)} 下找不到任何 trans.txt，請確認 --root")
        out = os.path.join(args.out_dir, f"{sub}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{sub}：{len(rows)} 句，{len(set(r['speaker'] for r in rows))} 位說話者 → {out}")


if __name__ == "__main__":
    main()
