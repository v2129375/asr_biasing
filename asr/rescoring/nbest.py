#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 10（不需微調的 LLM rescoring baseline）—— 步驟 1：產生 N-best

兩個語音辨識模型（都不微調、都不給關鍵詞）：
  --asr phi4      原始 Phi-4-multimodal-instruct（提示詞 "Transcribe the audio clip into text."，
                  與先前實驗的格式相同）
  --asr whisper   openai/whisper-large-v3（language=zh, task=transcribe；輸出以 OpenCC 轉成簡體）
解碼：beam search，num_beams = num_return_sequences = --beams（預設 30），正規化後取前 --n（預設 10）個不重複候選，記錄每個候選的
      sequences_scores 與 token 數。預設 --length_penalty 0：以整句總 log-prob 排序，避免重複／幻覺的長句排在前面。
後處理：與先前實驗相同（去標點、0–20 阿拉伯數字轉中文）；正規化後相同的候選合併，保留分數最高者。

用法（在專案根目錄執行）：
  # 測試集（必要）
  python asr/rescoring/nbest.py --asr phi4    --input data/catslu/test.csv  --split test
  python asr/rescoring/nbest.py --asr whisper --input data/catslu/test.csv  --split test
  # 訓練集（用來調 rescoring 的權重 λ，測試集不參與調參）
  python asr/rescoring/nbest.py --asr phi4    --input data/catslu/train.csv --split train
  python asr/rescoring/nbest.py --asr whisper --input data/catslu/train.csv --split train
  # 先試跑：--limit 5

輸出：asr/exp/rescoring/nbest_<asr>_<split>.jsonl（每行一句：path, manual_transcript, keyword, source, hyps）
     支援中斷後續跑（已完成的句子會跳過）。
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from score import normalize_for_score  # noqa: E402

BASE_INSTRUCTION = "Transcribe the audio clip into text."
CHINESE_NUMS = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七',
                '8': '八', '9': '九', '10': '十', '11': '十一', '12': '十二', '13': '十三', '14': '十四',
                '15': '十五', '16': '十六', '17': '十七', '18': '十八', '19': '十九', '20': '二十'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--asr", required=True, choices=["phi4", "whisper"])
    p.add_argument("--input", default="data/catslu/test.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=10, help="保留的 N-best 數（正規化後不重複的候選）")
    p.add_argument("--beams", type=int, default=30,
                   help="beam 數（＝回傳序列數）。短句的 beam 常只差標點，正規化後會合併，因此要比 --n 大")
    p.add_argument("--length_penalty", type=float, default=0.0,
                   help="beam 排序的長度懲罰。預設 0＝以整句總 log-prob 排序（標準 N-best 定義）；"
                        "1.0（transformers 預設）會以平均每 token 分數排序，容易讓重複、幻覺的長句排在前面")
    p.add_argument("--debug", action="store_true", help="印出每句合併前的原始 beam，檢查候選多樣性")
    p.add_argument("--phi4_model", default="microsoft/Phi-4-multimodal-instruct")
    p.add_argument("--whisper_model", default="openai/whisper-large-v3")
    p.add_argument("--attn", default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out_dir", default="asr/exp/rescoring")
    return p.parse_args()


def postprocess(text):
    """與 phi4_keywords.py 相同：去標點、阿拉伯數字轉中文"""
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\d+", lambda m: CHINESE_NUMS.get(m.group(0), m.group(0)), text)


def load_audio(path, target_sr=None):
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if target_sr and sr != target_sr:
        import librosa
        a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return a, sr


class Phi4NBest:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.phi4_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.phi4_model, device_map=args.device, torch_dtype="auto", trust_remote_code=True,
            _attn_implementation=args.attn).eval()
        self.gen_cfg = GenerationConfig.from_pretrained(args.phi4_model)
        self.prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|audio_1|>\n" + BASE_INSTRUCTION}], tokenize=False,
            add_generation_prompt=True)

    @torch.no_grad()
    def __call__(self, path):
        audio, sr = load_audio(path)
        inputs = self.processor(text=[self.prompt], audios=[(audio, sr)], return_tensors="pt").to(self.args.device)
        plen = inputs["input_ids"].shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                **inputs, max_new_tokens=self.args.max_new_tokens, generation_config=self.gen_cfg,
                num_beams=self.args.beams, num_return_sequences=self.args.beams, do_sample=False,
                early_stopping=True, length_penalty=self.args.length_penalty, return_dict_in_generate=True, output_scores=True, num_logits_to_keep=1)
        seqs = out.sequences[:, plen:]
        texts = self.processor.batch_decode(seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = {self.processor.tokenizer.eos_token_id, self.processor.tokenizer.pad_token_id}
        ntoks = [int(sum(1 for t in s.tolist() if t not in eos)) + 1 for s in seqs]
        return texts, out.sequences_scores.float().tolist(), ntoks


class WhisperNBest:
    """Whisper 自己的 generate() 包了長音訊／時間戳記的處理，num_return_sequences 在短音訊上只會回傳
    同一個結果；因此這裡繞過它，直接用 transformers 標準的 beam search（GenerationMixin.generate），
    以 <|startoftranscript|><|zh|><|transcribe|><|notimestamps|> 作為解碼起始序列。"""

    def __init__(self, args):
        import copy
        from opencc import OpenCC
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.args = args
        self.cc = OpenCC("t2s")
        self.processor = WhisperProcessor.from_pretrained(args.whisper_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            args.whisper_model, torch_dtype=torch.float16).to(args.device).eval()
        tok = self.processor.tokenizer
        ids = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|zh|>", "<|transcribe|>", "<|notimestamps|>"])
        self.prompt = torch.tensor([ids], device=args.device)
        self.gen_cfg = copy.deepcopy(self.model.generation_config)
        self.gen_cfg.forced_decoder_ids = None

    @torch.no_grad()
    def __call__(self, path):
        from transformers.generation.utils import GenerationMixin
        audio, sr = load_audio(path, 16000)
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        out = GenerationMixin.generate(
            self.model, input_features=feats.to(self.args.device, torch.float16),
            decoder_input_ids=self.prompt, generation_config=self.gen_cfg,
            num_beams=self.args.beams, num_return_sequences=self.args.beams, do_sample=False,
            early_stopping=True, length_penalty=self.args.length_penalty,
            max_new_tokens=self.args.max_new_tokens, return_dict_in_generate=True, output_scores=True)
        seqs = out.sequences[:, self.prompt.shape[1]:]
        texts = [self.cc.convert(t) for t in self.processor.batch_decode(seqs, skip_special_tokens=True)]
        special = set(self.processor.tokenizer.all_special_ids)
        ntoks = [int(sum(1 for t in s.tolist() if t not in special)) + 1 for s in seqs]
        return texts, out.sequences_scores.float().tolist(), ntoks


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"nbest_{args.asr}_{args.split}.jsonl")
    df = pd.read_csv(args.input, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["path"] for l in f if l.strip()}
        print(f"已有 {len(done)} 句，續跑")
    model = Phi4NBest(args) if args.asr == "phi4" else WhisperNBest(args)
    n_distinct = []
    with open(out_path, "a", encoding="utf-8") as fout:
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{args.asr} {args.split}"):
            if r["path"] in done:
                continue
            t0 = time.perf_counter()
            texts, scores, ntoks = model(r["path"])
            lat = time.perf_counter() - t0
            if args.debug:
                print(f"\n{r['manual_transcript']}")
                for t, s in zip(texts, scores):
                    print(f"   {s:8.4f}  {t!r}")
            merged = {}
            for rank, (t, s, n) in enumerate(zip(texts, scores, ntoks)):
                text = postprocess(t.strip())
                key = normalize_for_score(text)
                if key in merged:  # 正規化後相同的候選只留第一個（beam 分數較高者）
                    continue
                merged[key] = {"text": text, "raw": t, "rank": rank, "asr_score": s, "n_tokens": n,
                               "asr_logp": s * (n ** args.length_penalty)}  # 還原成整句總 log-prob
                if len(merged) >= args.n:
                    break
            n_distinct.append(len(merged))
            fout.write(json.dumps({"path": r["path"], "manual_transcript": r["manual_transcript"],
                                   "keyword": r["keyword"], "source": r["source"], "latency_s": lat,
                                   "n_raw": len(texts),
                                   "hyps": list(merged.values())}, ensure_ascii=False) + "\n")
            fout.flush()
    if n_distinct:
        print(f"本次 {len(n_distinct)} 句：每句不重複候選平均 {np.mean(n_distinct):.2f} 個，"
              f"只有 1 個候選的句子 {sum(1 for x in n_distinct if x == 1)} 句")
        if np.mean(n_distinct) < 2:
            print("警告：候選幾乎都只有 1 個，N-best 沒有正常產生，請加 --debug 檢查")
    print(f"已輸出 {out_path}")


if __name__ == "__main__":
    main()
