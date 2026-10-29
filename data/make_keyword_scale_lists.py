#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 5（Keyword list 規模的影響）—— 步驟 1：產生不同規模的 keyword list

規則（每個規模 N、每個領域）：
  1. 原 keyword list（data/catslu/keyword_{領域}.txt）**完整保留**：內容、順序、重複行都不改動；
  2. 從額外關鍵詞池隨機取 (N − 原行數) 個干擾詞，**隨機插入**原 list 的各個位置，
     湊滿 N 行（N 指提示詞中的關鍵詞個數，與原程式「一行一個關鍵詞、以空白連接」的算法一致）；
  3. 巢狀：較小規模的干擾詞一定包含在較大規模中（1000 ⊂ 1500 ⊂ 2000 …），規模越大只是「多加」干擾詞。

因此：
  - 把任何一個規模的 list 中的干擾詞拿掉，剩下的就是原 list（同內容、同順序）；
  - 測試集關鍵詞的涵蓋率固定為 100%，只有干擾詞數量改變；
  - 干擾詞散布在整個 list 中，正確關鍵詞不會固定排在最前面或最後面。

輸出格式與 data/catslu/keyword_*.txt 相同，原有的訓練程式 finetune_catslu_keywords.py
與推論程式 phi4_keywords.py 只要把 --keywords_dir 指到新資料夾就能直接使用。

用法（在專案根目錄執行）：
  python data/make_keyword_scale_lists.py                       # 預設 1000 1500 2000 2500 3000
  python data/make_keyword_scale_lists.py --sizes 1000 1500 2000

輸出：
  data/keyword_scale/kw1000/keyword_video.txt、keyword_music.txt、keyword_city.txt
  data/keyword_scale/kw1500/ …
  data/keyword_scale/summary.csv（各規模、各領域的原行數、干擾詞數、總行數、總字數）
"""

import argparse
import csv
import os
import random

DOMAINS = ["video", "music", "city"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig_dir", default="data/catslu", help="原 keyword list 資料夾")
    p.add_argument("--pool_dir", default="data/keyword_pool", help="額外關鍵詞池資料夾")
    p.add_argument("--out_dir", default="data/keyword_scale")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 1500, 2000, 2500, 3000],
                   help="每個領域 list 的關鍵詞個數（行數）")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    args = parse_args()
    orig = {d: read_lines(os.path.join(args.orig_dir, f"keyword_{d}.txt")) for d in DOMAINS}
    pool = {}
    for d in DOMAINS:
        words = list(dict.fromkeys(read_lines(os.path.join(args.pool_dir, f"extra_{d}.txt"))))
        orig_set = set(orig[d])
        pool[d] = [w for w in words if w not in orig_set]
        # 干擾詞的抽取順序：固定種子隨機排列，依序取用 → 規模之間自然巢狀
        random.Random(f"{args.seed}-pool-{d}").shuffle(pool[d])

    rows = []
    for n in sorted(args.sizes):
        out = os.path.join(args.out_dir, f"kw{n}")
        os.makedirs(out, exist_ok=True)
        for d in DOMAINS:
            need = n - len(orig[d])
            if need < 0:
                raise SystemExit(f"錯誤：{d} 原 list 已有 {len(orig[d])} 行，大於規模 {n}")
            if need > len(pool[d]):
                raise SystemExit(f"錯誤：{d} 額外關鍵詞池只有 {len(pool[d])} 個，不足以擴充到 {n}")
            extras = pool[d][:need]
            # 把干擾詞隨機插入原 list：先決定 N 個位置中哪些放干擾詞，其餘依序放原關鍵詞
            rng = random.Random(f"{args.seed}-insert-{d}-{n}")
            extra_pos = set(rng.sample(range(n), need))
            it_orig, it_extra = iter(orig[d]), iter(extras)
            words = [next(it_extra) if i in extra_pos else next(it_orig) for i in range(n)]
            # 自我檢查：拿掉干擾詞後必須與原 list 完全相同
            extra_set = set(extras)
            assert [w for w in words if w not in extra_set] == orig[d]
            with open(os.path.join(out, f"keyword_{d}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(words) + "\n")
            rows.append([n, d, len(orig[d]), need, len(words), sum(len(w) for w in words)])
            print(f"kw{n} {d}: 原 list {len(orig[d])} 行（完整保留） + 干擾詞 {need} = {len(words)}"
                  f"（共 {rows[-1][-1]} 字）")

    with open(os.path.join(args.out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "domain", "n_original_lines", "n_distractor", "n_total", "n_chars"])
        w.writerows(rows)
    print(f"已輸出到 {args.out_dir}/")


if __name__ == "__main__":
    main()
