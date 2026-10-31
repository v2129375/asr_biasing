#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 8（災難性遺忘）—— 步驟 3：比較微調前後，並做統計檢定

對每個測試集，以「同一批句子」比較原始模型與微調模型：
  - 錯誤率（WER/CER）的絕對差與相對變化（正值＝微調後變差＝遺忘）
  - 成對 bootstrap（以句子為單位重抽 --n_boot 次）求差值的 95% 信賴區間與雙尾 p 值
  - 每句錯誤數變多／變少／相同的句數
  - 語言漂移：英文音訊輸出中文的句數

用法：
  python asr/compare_forgetting.py \
      --base asr/exp/forgetting/Phi-4-multimodal-instruct \
      --ft asr/exp/forgetting/KworigKeywords
  # 可同時比較多個微調模型：--ft dirA dirB

輸出：asr/exp/forgetting/compare.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--ft", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", default=["test-clean", "test-other", "aishell"])
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="asr/exp/forgetting/compare.csv")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for ft in args.ft:
        for ds in args.datasets:
            fa, fb = os.path.join(args.base, f"{ds}.csv"), os.path.join(ft, f"{ds}.csv")
            if not (os.path.exists(fa) and os.path.exists(fb)):
                print(f"略過 {ds}：{fa} 或 {fb} 不存在")
                continue
            a = pd.read_csv(fa, keep_default_na=False)
            b = pd.read_csv(fb, keep_default_na=False)
            m = a.merge(b, on="path", suffixes=("_a", "_b"))
            if len(m) != len(a) or len(m) != len(b):
                print(f"注意 {ds}：兩邊句數不同（{len(a)} vs {len(b)}），只比較共同的 {len(m)} 句")
            ea, eb, n = m["errors_a"].to_numpy(float), m["errors_b"].to_numpy(float), m["N_a"].to_numpy(float)
            ra, rb = ea.sum() / n.sum(), eb.sum() / n.sum()
            idx = rng.integers(0, len(m), size=(args.n_boot, len(m)))
            nn = n[idx].sum(1)
            diff = eb[idx].sum(1) / nn - ea[idx].sum(1) / nn
            lo, hi = np.percentile(diff, [2.5, 97.5])
            pval = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
            metric = "WER" if ds.startswith("test-") else "CER"
            row = {"model": os.path.basename(os.path.normpath(ft)), "dataset": ds, "metric": metric,
                   "n_utts": len(m), "base_rate": round(ra, 5), "ft_rate": round(rb, 5),
                   "abs_diff": round(rb - ra, 5), "rel_change": round((rb - ra) / ra, 4) if ra else None,
                   "ci95_low": round(lo, 5), "ci95_high": round(hi, 5), "bootstrap_p": round(pval, 4),
                   "utts_worse": int((eb > ea).sum()), "utts_better": int((eb < ea).sum()),
                   "utts_same": int((eb == ea).sum())}
            if "has_cjk_b" in m and ds.startswith("test-"):
                row["base_cjk_outputs"] = int(m["has_cjk_a"].astype(str).str.lower().eq("true").sum())
                row["ft_cjk_outputs"] = int(m["has_cjk_b"].astype(str).str.lower().eq("true").sum())
            rows.append(row)
            print(f"[{row['model']} | {ds}] {metric} {ra * 100:.2f}% → {rb * 100:.2f}%  "
                  f"差 {(rb - ra) * 100:+.2f}（95% CI {lo * 100:+.2f} ~ {hi * 100:+.2f}，p={pval:.4f}）")
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n已輸出 {args.out}")


if __name__ == "__main__":
    main()
