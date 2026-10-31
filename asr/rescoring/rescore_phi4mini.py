#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 10 —— 步驟 2：以 Phi-4-mini（文字 LLM，不微調）搭配 keyword list 對 N-best 重新評分，
並與動態關鍵詞微調方法（既有結果，不重跑）比較

關鍵詞清單：與動態關鍵詞方法的 orig 條件完全相同（data/catslu/keyword_<領域>.txt，依真實領域 source 選擇），
因此兩種方法拿到的關鍵詞資訊一樣，差別只在「關鍵詞在解碼時使用」或「解碼後只看文字重新評分」。

兩種 rescoring（都使用同一個 Phi-4-mini）：
  select  選擇式：把 N 個候選（標為 A、B、C…）與關鍵詞清單交給 LLM，比較下一個 token 為各字母的機率，
          選機率最高者。不需調參。
  score   分數式：LLM 計算「給定關鍵詞清單」時每個候選文字的 log-prob（lm_logp），
          總分 = asr_logp + λ × lm_logp，選總分最高者。λ 只在訓練集 N-best 上調（以關鍵詞正確率為準，
          平手看 CER），測試集只用調好的 λ。沒有訓練集 N-best 時，改報各 λ 的測試結果（僅供參考）。

另外報告：
  1best   ASR 原本的第一名（不 rescoring）
  oracle  N 個候選中只要有一個包含正確關鍵詞就算對（任何 rescoring 方法的上限）

用法（在專案根目錄執行）：
  預設 --lm_backend phi4mm：Phi-4-multimodal 的語言模型就是凍結的 Phi-4-mini，以純文字模式（input_mode=0，
  不掛 LoRA）執行即可在原環境（transformers 4.48.2）使用；可先用 check_phi4mini_weights.py 確認權重相同。
  python asr/rescoring/rescore_phi4mini.py --asr phi4
  python asr/rescoring/rescore_phi4mini.py --asr whisper
  # 與動態關鍵詞方法比較的結果檔（預設為意見 5 的 Kworig orig 條件）：
  #   --compare asr/exp/keyword_scale/KworigKeywords_source/orig.csv

輸出：asr/exp/rescoring/<asr>/
  test_results.csv     每句：參考、關鍵詞、1best/select/score/oracle 的輸出與是否命中
  lambda_tuning.csv    訓練集上各 λ 的 KER/CER
  summary.csv          各方法 KER、CER（corpus）、各領域 KER、平均時間
  mcnemar.csv          與 1best、與動態關鍵詞方法的成對檢定
  lm_cache_<split>.jsonl  LLM 分數快取（重跑時不必重算）
"""

import argparse
import json
import os
import string
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from score import cer, normalize_for_score  # noqa: E402

LETTERS = list(string.ascii_uppercase)
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--asr", required=True, choices=["phi4", "whisper"])
    p.add_argument("--nbest_dir", default="asr/exp/rescoring")
    p.add_argument("--keywords_dir", default="data/catslu")
    p.add_argument("--lm_backend", default="phi4mm", choices=["phi4mm", "phi4mini"],
                   help="phi4mm：用 Phi-4-multimodal 的純文字模式（不掛任何 LoRA，即凍結的 Phi-4-mini 語言模型），"
                        "可在 transformers 4.48.2 下執行；phi4mini：載入 microsoft/Phi-4-mini-instruct（需 transformers ≥ 4.49）")
    p.add_argument("--lm", default="", help="模型路徑；預設依 --lm_backend 自動選擇")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--compare", default="asr/exp/keyword_scale/KworigKeywords_source/orig.csv",
                   help="動態關鍵詞方法的逐句結果（需含 path、asr、keyword 欄）")
    p.add_argument("--methods", nargs="+", default=["select", "score"], choices=["select", "score"])
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def hit(text, kw):
    h, k = normalize_for_score(text), normalize_for_score(kw)
    return bool(h) and bool(k) and k in h


def corpus_cer(refs, hyps):
    e = t = 0
    for r, h in zip(refs, hyps):
        r, h = list(normalize_for_score(r)), list(normalize_for_score(h))
        if r:
            e += cer(r, h) * len(r)
            t += len(r)
    return e / t if t else None


def load_nbest(path, limit=0):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[:limit] if limit else rows


class LM:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = args.device
        if args.lm_backend == "phi4mm":
            path = args.lm or "microsoft/Phi-4-multimodal-instruct"
            self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
                _attn_implementation=args.attn).eval()
            # 純文字模式：明確卸下 vision/speech LoRA，只用凍結的語言模型（input_mode=0＝LANGUAGE）
            self.model.unset_lora_adapter()
            self.extra = {"input_mode": 0}
            self._checked = False
        else:
            path = args.lm or "microsoft/Phi-4-mini-instruct"
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=False,
                attn_implementation=args.attn).eval()
            self.extra = {}
        print(f"文字 LLM：{path}（backend={args.lm_backend}）")

    def check_lora_off(self):
        """第一次 forward 後確認所有 LoRA 層都已停用（phi4mm backend 才需要）"""
        if getattr(self, "_checked", True):
            return
        n_lora = n_on = 0
        for m in self.model.modules():
            if hasattr(m, "lora_A") and hasattr(m, "disable_adapters"):
                n_lora += 1
                if not m.disable_adapters:
                    n_on += 1
        print(f"LoRA 檢查：共 {n_lora} 個 LoRA 層，啟用中 {n_on} 個")
        if n_on:
            raise SystemExit("錯誤：純文字模式下仍有 LoRA 啟用，結果不等於 Phi-4-mini，請把這段訊息貼給我")
        self._checked = True
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok.encode(L, add_special_tokens=False)
            self.letter_ids.append(ids[0])

    def chat(self, content):
        return self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                            add_generation_prompt=True)

    @torch.no_grad()
    def select(self, cands, keywords):
        """回傳各候選的選擇機率（log-softmax over 候選字母）"""
        lines = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cands))
        content = ("下面是同一段中文语音的多个语音识别候选结果。说话内容很可能包含以下关键词之一：\n"
                   f"{' '.join(keywords)}\n\n候选结果：\n{lines}\n\n"
                   "请选出最可能正确的识别结果，只回答一个字母。")
        ids = self.tok(self.chat(content), return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        logits = self.model(input_ids=ids, **self.extra).logits[0, -1].float()
        self.check_lora_off()
        sub = logits[self.letter_ids[:len(cands)]]
        return torch.log_softmax(sub, -1).tolist()

    @torch.no_grad()
    def logp(self, cands, keywords):
        """回傳各候選文字在「給定關鍵詞清單」條件下的 log-prob 總和"""
        prefix = self.chat("以下是一句中文语音识别结果，内容很可能包含这些关键词之一：\n"
                           f"{' '.join(keywords)}\n请写出这句话。")
        p_ids = self.tok(prefix, add_special_tokens=False).input_ids
        out = []
        for c in cands:
            c_ids = self.tok(c, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            ids = torch.tensor([p_ids + c_ids], device=self.device)
            logits = self.model(input_ids=ids, **self.extra).logits[0, len(p_ids) - 1:-1].float()
            self.check_lora_off()
            lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(c_ids, device=self.device)[:, None])
            out.append(float(lp.sum()))
        return out


def lm_scores(rows, lm, lists, methods, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                cache[d["path"]] = d
    with open(cache_path, "a", encoding="utf-8") as fout:
        for r in tqdm(rows, desc=os.path.basename(cache_path)):
            c = cache.get(r["path"], {"path": r["path"]})
            need = [m for m in methods if m not in c]
            if need:
                cands = [h["text"] for h in r["hyps"]]
                kws = lists[r["source"]]
                for m in need:
                    t0 = time.perf_counter()
                    c[m] = lm.select(cands, kws) if m == "select" else lm.logp(cands, kws)
                    c[f"{m}_time"] = time.perf_counter() - t0
                cache[r["path"]] = c
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                fout.flush()
    return cache


def pick_score(r, c, lam):
    tot = [h["asr_logp"] + lam * l for h, l in zip(r["hyps"], c["score"])]
    return int(np.argmax(tot))


def evaluate(rows, picks):
    refs = [r["manual_transcript"] for r in rows]
    hyps = [r["hyps"][i]["text"] if r["hyps"] else "" for r, i in zip(rows, picks)]
    hits = [hit(h, r["keyword"]) for h, r in zip(hyps, rows)]
    return hyps, hits, 1 - np.mean(hits), corpus_cer(refs, hyps)


def mcnemar(a, b):
    a, b = np.asarray(a), np.asarray(b)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return x, y, (binomtest(min(x, y), x + y).pvalue if x + y else 1.0)


def main():
    args = parse_args()
    out_dir = os.path.join(args.nbest_dir, args.asr)
    os.makedirs(out_dir, exist_ok=True)
    lists = {d: read_list(os.path.join(args.keywords_dir, f"keyword_{d}.txt")) for d in ["video", "music", "city"]}
    test = load_nbest(os.path.join(args.nbest_dir, f"nbest_{args.asr}_test.jsonl"), args.limit)
    train_path = os.path.join(args.nbest_dir, f"nbest_{args.asr}_train.jsonl")
    train = load_nbest(train_path, args.limit) if os.path.exists(train_path) and "score" in args.methods else None
    print(f"測試 {len(test)} 句；訓練（調 λ）{len(train) if train else 0} 句；"
          f"平均候選數 {np.mean([len(r['hyps']) for r in test]):.1f}")

    lm = LM(args)
    c_test = lm_scores(test, lm, lists, args.methods, os.path.join(out_dir, "lm_cache_test.jsonl"))

    res = pd.DataFrame({"path": [r["path"] for r in test], "manual_transcript": [r["manual_transcript"] for r in test],
                        "keyword": [r["keyword"] for r in test], "source": [r["source"] for r in test],
                        "n_hyps": [len(r["hyps"]) for r in test]})
    summary = []

    def add(name, picks=None, hyps=None, hits=None, extra_time=None):
        if picks is not None:
            hyps, hits, ker, c = evaluate(test, picks)
        else:
            ker, c = 1 - np.mean(hits), corpus_cer(res["manual_transcript"], hyps)
        res[f"{name}_hyp"], res[f"{name}_hit"] = hyps, hits
        row = {"method": name, "ker": round(ker, 4), "cer_corpus": round(c, 4)}
        for d in ["video", "music", "city"]:
            m = res["source"] == d
            row[f"ker_{d}"] = round(1 - np.mean(np.asarray(hits)[m.values]), 4)
        asr_t = np.mean([r.get("latency_s", 0) for r in test])
        row["asr_time_s"] = round(asr_t, 3)
        row["rescore_time_s"] = round(extra_time, 3) if extra_time is not None else 0.0
        summary.append(row)

    add("1best", picks=[0] * len(test))
    oracle_hits = [any(hit(h["text"], r["keyword"]) for h in r["hyps"]) for r in test]
    oracle_hyps = [next((h["text"] for h in r["hyps"] if hit(h["text"], r["keyword"])),
                        r["hyps"][0]["text"] if r["hyps"] else "") for r in test]
    add("oracle", hyps=oracle_hyps, hits=oracle_hits)

    if "select" in args.methods:
        picks = [int(np.argmax(c_test[r["path"]]["select"])) for r in test]
        add("select", picks=picks, extra_time=np.mean([c_test[r["path"]]["select_time"] for r in test]))

    if "score" in args.methods:
        tune = []
        if train:
            c_train = lm_scores(train, lm, lists, ["score"], os.path.join(out_dir, "lm_cache_train.jsonl"))
            for lam in LAMBDAS:
                _, _, ker, c = evaluate(train, [pick_score(r, c_train[r["path"]], lam) for r in train])
                tune.append({"lambda": lam, "train_ker": round(ker, 4), "train_cer": round(c, 4)})
            tdf = pd.DataFrame(tune).sort_values(["train_ker", "train_cer", "lambda"])
            best = float(tdf.iloc[0]["lambda"])
            print(f"訓練集選出 λ = {best}")
        else:
            best = None
            print("沒有訓練集 N-best：改報各 λ 在測試集上的結果（僅供參考，不可當作正式結果）")
        for lam in LAMBDAS:
            _, _, ker, c = evaluate(test, [pick_score(r, c_test[r["path"]], lam) for r in test])
            match = [t for t in tune if t["lambda"] == lam]
            if match:
                match[0].update({"test_ker": round(ker, 4), "test_cer": round(c, 4)})
            else:
                tune.append({"lambda": lam, "test_ker": round(ker, 4), "test_cer": round(c, 4)})
        pd.DataFrame(tune).to_csv(os.path.join(out_dir, "lambda_tuning.csv"), index=False)
        if best is not None:
            add(f"score", picks=[pick_score(r, c_test[r["path"]], best) for r in test],
                extra_time=np.mean([c_test[r["path"]]["score_time"] for r in test]))
            summary[-1]["lambda"] = best

    # 動態關鍵詞方法（既有結果）
    if args.compare and os.path.exists(args.compare):
        cmp = pd.read_csv(args.compare, keep_default_na=False)[["path", "asr", "latency_s"]]
        cmp = res[["path"]].merge(cmp, on="path", how="left")
        if cmp["asr"].isna().any():
            print(f"注意：{args.compare} 缺少 {int(cmp['asr'].isna().sum())} 句，這些句子以空輸出計")
        hy = cmp["asr"].fillna("").tolist()
        add("dynamic_keyword", hyps=hy, hits=[hit(h, k) for h, k in zip(hy, res["keyword"])])
        summary[-1]["asr_time_s"] = round(cmp["latency_s"].astype(float).mean(), 3)
    else:
        print(f"找不到動態關鍵詞方法的結果 {args.compare}，略過比較")

    res.to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    mc = []
    names = [s["method"] for s in summary if s["method"] != "oracle"]
    for a in names:
        for b in names:
            if a < b and ("1best" in (a, b) or "dynamic_keyword" in (a, b)):
                x, y, pv = mcnemar(res[f"{a}_hit"], res[f"{b}_hit"])
                mc.append({"A": a, "B": b, "only_A_correct": x, "only_B_correct": y, "p": float(f"{pv:.3g}")})
    pd.DataFrame(mc).to_csv(os.path.join(out_dir, "mcnemar.csv"), index=False)
    print("\n=== 結果彙整 ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sm.to_string(index=False))
        print(pd.DataFrame(mc).to_string(index=False))
    print(f"\n已輸出到 {out_dir}/")


if __name__ == "__main__":
    main()
