#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新實驗：地圖（map）領域 zero-shot 測試
目的：只用 video／music／city 訓練的 ASR 模型，在完全沒看過的地圖領域上，
     能不能「利用提示詞中的關鍵詞清單」把地點名稱辨識對。

測試資料：data/prepare_map_test_data.py 輸出的 data/map/test.csv（全部都是測試句）
關鍵詞表：data/map/keyword.txt（poi_name lexicon，純中文，4923 個）

提示條件（--conditions，可自由組合）：
  none       不給關鍵詞（"Transcribe the audio clip into text."），作為基準
  full       整份 keyword.txt（每句都相同，約 4923 個）
  randN      該句正確關鍵詞 ＋ 從 keyword.txt 其餘詞隨機抽 N−1 個，打亂順序（例如 rand1000、rand3000）
             - 亂數只由「句子編號」決定，所以每個模型看到的清單完全相同，可做成對比較（McNemar）
             - 巢狀抽樣：rand1000 的清單一定包含在 rand2000 之中
  rand1      只給正確關鍵詞（上限參考）
提示詞格式、生成參數、後處理、CER/KER 計算都與 asr/phi4_keywords_scale.py 相同
（直接匯入它的 Runner、postprocess、distractor_inserted 與 asr/score.py 的 evaluate_asr）。

用法（在專案根目錄執行；需要 asr/phi4_keywords_scale.py 在同一資料夾）：
  python asr/phi4_map_zeroshot.py \
      --model_path asr/model/<模型資料夾> \
      --conditions none rand1 rand1000 rand3000 full \
      --gpus 0 --attn flash_attention_2

  # 未微調的原始模型（對照組）
  python asr/phi4_map_zeroshot.py --model_path microsoft/Phi-4-multimodal-instruct ...

  # 先用少量句子確認流程／顯存：--limit 20；想縮短時間可隨機抽樣：--sample 1000

輸出（預設 asr/exp/map_zeroshot/<模型資料夾名>/）：
  <條件>.csv / <條件>.json   每句結果（含 hit 欄：關鍵詞是否辨識正確）/ evaluate_asr 評分
  summary.csv / summary.json  各條件的 CER、corpus CER、KER、依槽位的 KER、提示詞誤插入率、
                              提示長度、推論時間、顯存、OOM
"""

import argparse
import json
import os
import random
import sys

import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import evaluate_asr, normalize_for_score, cer  # noqa: E402
from phi4_keywords_scale import (  # noqa: E402
    BASE_INSTRUCTION, KEYWORD_INSTRUCTION_TEMPLATE, Runner, distractor_inserted, is_oom, postprocess,
)


def parse_args():
    p = argparse.ArgumentParser(description="地圖領域 zero-shot 測試")
    p.add_argument("--input", default="data/map/test.csv")
    p.add_argument("--keyword_file", default="data/map/keyword.txt")
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default="microsoft/Phi-4-multimodal-instruct")
    p.add_argument("--conditions", nargs="+", default=["none", "rand1", "rand1000", "rand3000", "full"])
    p.add_argument("--seed", type=int, default=42, help="randN 的亂數種子（所有模型請用相同值）")
    p.add_argument("--gpus", default="0")
    p.add_argument("--device_map", default="asr/finetune/device_map.json")
    p.add_argument("--attn", default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--max_oom", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 句（除錯用）")
    p.add_argument("--sample", type=int, default=0, help="隨機抽 N 句（固定種子，所有模型抽到同一批）")
    p.add_argument("--out_dir", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min_audio_s", type=float, default=0.1,
                   help="短於此秒數（或空白）的音訊無法抽特徵，從測試中排除（所有模型排除同一批）")
    return p.parse_args()


def read_list(path):
    with open(path, encoding="utf-8") as f:
        return list(dict.fromkeys(l.strip() for l in f if l.strip()))


def utt_keywords(cond, i, keyword, lexicon, seed):
    """回傳第 i 句在此條件下的關鍵詞清單"""
    if cond == "none":
        return []
    if cond == "full":
        return lexicon
    n = int(cond[4:])
    rng = random.Random(f"{seed}-{i}")
    others = [w for w in lexicon if w != keyword]
    rng.shuffle(others)  # 每句固定一個排列，取前 N−1 個 → 不同 N 之間巢狀
    chosen = [keyword] + others[: max(n - 1, 0)]
    random.Random(f"{seed}-{i}-{n}").shuffle(chosen)
    return chosen


def keyword_hit(hyp, keyword):
    h, k = normalize_for_score(hyp), normalize_for_score(keyword)
    return bool(h) and bool(k) and k in h


def corpus_cer(df):
    edits = total = 0
    for ref, hyp in zip(df["manual_transcript"], df["asr"]):
        r, h = list(normalize_for_score(ref)), list(normalize_for_score(hyp))
        if r:
            edits += cer(r, h) * len(r)
            total += len(r)
    return edits / total if total else None


def main():
    args = parse_args()
    for c in args.conditions:
        if c not in ("none", "full") and not (c.startswith("rand") and c[4:].isdigit()):
            raise SystemExit(f"錯誤：不認得的條件 {c}（可用 none、full、randN）")
    df0 = pd.read_csv(args.input, keep_default_na=False)
    if args.sample:
        df0 = df0.sample(n=min(args.sample, len(df0)), random_state=args.seed).sort_index()
    if args.limit:
        df0 = df0.head(args.limit)
    df0 = df0.reset_index().rename(columns={"index": "utt_idx"})  # utt_idx＝在 test.csv 的列號，決定亂數
    lexicon = read_list(args.keyword_file)
    miss = (~df0["keyword"].isin(set(lexicon))).sum()
    print(f"測試句 {len(df0)}；關鍵詞表 {len(lexicon)} 個；正確關鍵詞不在表中 {miss} 句")

    name = os.path.basename(os.path.normpath(args.model_path))
    out_dir = args.out_dir or os.path.join("asr", "exp", "map_zeroshot", name)
    os.makedirs(out_dir, exist_ok=True)

    audios, keep = [], []
    for p in tqdm(df0["path"], desc="讀取音訊"):
        a, sr = sf.read(p)
        if a.ndim > 1:
            a = a.mean(axis=1)
        ok = len(a) / sr >= args.min_audio_s
        keep.append(ok)
        if ok:
            audios.append((a, sr))
    bad = df0[[not k for k in keep]]
    if len(bad):
        print(f"排除 {len(bad)} 句過短／空白音訊（< {args.min_audio_s} 秒）：")
        for _, r in bad.iterrows():
            print(f"  {r['path']}  {r['manual_transcript']}")
        bad.to_csv(os.path.join(out_dir, "excluded_audio.csv"), index=False)
    df0 = df0[keep].reset_index(drop=True)
    print(f"實際測試 {len(df0)} 句")

    runner, summary = None, []
    for cond in args.conditions:
        csv_path = os.path.join(out_dir, f"{cond}.csv")
        meta_path = os.path.join(out_dir, f"{cond}_meta.json")
        peak = None
        if os.path.exists(csv_path) and not args.overwrite:
            print(f"\n[{cond}] 已有結果，直接重新評分（加 --overwrite 可重跑）")
            df = pd.read_csv(csv_path, keep_default_na=False)
            df["oom"] = df["oom"].astype(str).str.lower() == "true"
            df["error"] = df["error"].astype(str).str.lower() == "true" if "error" in df else False
            if os.path.exists(meta_path):
                peak = json.load(open(meta_path)).get("peak_mem_gb")
        else:
            if runner is None:
                runner = Runner(args)
            df = df0.copy()
            df["asr"], df["prompt_tokens"], df["latency_s"], df["audio_s"], df["oom"] = "", 0, 0.0, 0.0, False
            df["list_size"] = 0
            df["error"] = False
            runner.free()
            runner.reset_peak()
            consecutive = 0
            for i in tqdm(range(len(df)), desc=cond):
                audio, sr = audios[i]
                df.at[i, "audio_s"] = len(audio) / sr
                if consecutive >= args.max_oom:
                    df.at[i, "oom"] = True
                    continue
                words = utt_keywords(cond, int(df.at[i, "utt_idx"]), df.at[i, "keyword"], lexicon, args.seed)
                df.at[i, "list_size"] = len(words)
                instr = KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=" ".join(words)) if words else BASE_INSTRUCTION
                try:
                    text, n_tok, lat = runner.transcribe(audio, sr, instr)
                    df.at[i, "asr"] = postprocess(text)
                    df.at[i, "prompt_tokens"], df.at[i, "latency_s"] = n_tok, lat
                    consecutive = 0
                except Exception as e:  # noqa: BLE001
                    if not is_oom(e):
                        # 其他錯誤（例如音訊特徵抽取失敗）：記錄後繼續，不中斷整個條件
                        print(f"\n[{cond}] 第 {i} 句發生錯誤，略過：{df.at[i, 'path']}  {type(e).__name__}: {e}")
                        df.at[i, "error"] = True
                        continue
                    df.at[i, "oom"] = True
                    consecutive += 1
                    runner.free()
                    if consecutive == args.max_oom:
                        print(f"\n[{cond}] 連續 {args.max_oom} 次 OOM，停止此條件")
            peak = runner.peak_gb()
            df["hit"] = [keyword_hit(h, k) for h, k in zip(df["asr"], df["keyword"])]
            df.to_csv(csv_path, index=False)
            json.dump({"peak_mem_gb": peak}, open(meta_path, "w"))

        n_error = int(df["error"].sum())
        if n_error:
            print(f"[{cond}] 有 {n_error} 句推論出錯，已從評分中排除（各條件請確認句數一致）")
        df = df[~df["error"]].reset_index(drop=True)
        df["hit"] = [keyword_hit(h, k) for h, k in zip(df["asr"], df["keyword"])]
        eval_res = evaluate_asr(df.reset_index(drop=True), cal_keyword_wer=True, print_errors=False,
                                output_file=os.path.join(out_dir, f"{cond}.json"))
        ok = df[~df["oom"]]
        row = {
            "condition": cond,
            "list_size": int(df["list_size"].max()) if "list_size" in df else None,
            "n_utts": len(df), "n_oom": int(df["oom"].sum()), "n_error": n_error,
            "cer_mean": eval_res["overall"].get("cer"),
            "cer_corpus": round(corpus_cer(df), 4),
            "ker": round(1 - df["hit"].mean(), 4),
        }
        for slot, g in df.groupby("slot"):
            row[f"ker_{slot}"] = round(1 - g["hit"].mean(), 4)
        if cond != "none":
            row["prompt_insertion_rate"] = round(df.apply(
                lambda r: distractor_inserted(
                    r["asr"], r["manual_transcript"],
                    utt_keywords(cond, int(r["utt_idx"]), r["keyword"], lexicon, args.seed)), axis=1).mean(), 4)
        row.update({
            "prompt_tokens_mean": round(ok["prompt_tokens"].mean(), 1) if len(ok) else None,
            "latency_mean_s": round(ok["latency_s"].mean(), 4) if len(ok) else None,
            "peak_mem_gb": round(peak, 2) if peak is not None else None,
        })
        summary.append(row)
        print(f"[{cond}] KER={row['ker']:.4f}  CER(corpus)={row['cer_corpus']:.4f}")

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n=== 結果彙整 ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(sm.to_string(index=False))
    print(f"\n已輸出到 {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
