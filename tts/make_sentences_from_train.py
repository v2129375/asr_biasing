#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口试意见 4（实验五变因控制）—— 第 1 步：以真实训练集的「每句关键词」造句

与旧版 tts/phi4mini_make_sentance.py 的差别：
  旧版：读取 keyword_video/music/city.txt 中的「全部」关键词（930+390+555=1875 个），
        每个关键词造一句 → 1875 句，与真实训练集（1511 句）的关键词、筆数都不同，
        而且包含了测试集里才出现的关键词。
  新版：逐行读取真实训练集 data/catslu/train.csv（1511 句），
        但「只读取每句的 keyword 与 source 两栏」，不读取真实文本与音频，
        模拟「只有关键词、没有文本」的情况；每个 (source, keyword) 造一句 → 输出 1511 句：
          - 筆数相同（1511）
          - 每句的关键词与领域相同（因此各领域句数 video/music/city 也相同）
          - 句子内容完全由 Phi-4-mini 生成，不使用真实文本

造句方法沿用论文（Phi-4-mini + few-shot，system prompt 与示例与旧版完全相同），
只做了以下必要的改进：
  1. 训练集中同一个关键词会出现多次（1511 句只有 514 个不同关键词），旧版 do_sample=False
     会让同一关键词生成完全相同的句子，因此改为带随机种子的采样，并尽量避免同关键词重复句。
  2. 检查生成句：必须原样包含关键词、只能有中文字和标点（避免阿拉伯数字/英文导致标注不一致）、
     长度合理；不合格则重试，重试多次仍失败则使用简单模板句，并在 gen_status 栏记录。
  3. 输出两种文本：
       tts_text          —— 保留标点，送进 TTS（标点有助于韵律）
       manual_transcript —— 只保留中文字，与真实训练集标注格式一致（作为 ASR 训练标签）
  4. 支持断点续跑（输出档已存在时跳过已完成的行）。

用法（在 repo 根目录执行）：
  python tts/make_sentences_from_train.py
  python tts/make_sentences_from_train.py --train_csv data/catslu/train.csv \
         --output tts/tts_data/sentences_train_kw.csv --seed 42
"""

import argparse
import os
import re
import random

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ========== 默认配置 ==========
TRAIN_CSV = "data/catslu/train.csv"                    # 真实训练集（1511 句）
OUTPUT_FILE = "tts/tts_data/sentences_train_kw.csv"    # 输出：与训练集一一对应的造句结果
MODEL_PATH = "microsoft/Phi-4-mini-instruct"
SAVE_EVERY = 50
# ==============================

# system prompt 与 few-shot 示例沿用旧版 phi4mini_make_sentance.py（即论文 Figure 6 的做法）
SYSTEM_MESSAGE = """我会给你一些专有名词和领域，请根据这些专有名词和它的领域进行造句，句子是在进行询问天气、点播音乐、点播视频或者电视剧
其中video领域是点播视频或者电视剧，music领域是点播音乐，city领域是询问天气
如果是播放音乐，则造句和音乐播放相关的点播命令
如果是询问天气则造句应该和天气询问相关的疑问句
注意点播视频或者音乐应该是命令的句式不应该有疑问语气，而询问天气是疑问句
"""

FEW_SHOT = [
    {"role": "user", "content": "video:何以笙箫默"},
    {"role": "assistant", "content": "我想看电视剧何以笙箫默"},
    {"role": "user", "content": "music:温岚"},
    {"role": "assistant", "content": "我要听温岚的囚鸟"},
    {"role": "user", "content": "city:深圳"},
    {"role": "assistant", "content": "深圳天气怎么样"},
]

# 重试多次仍失败时的保底模板（只在极少数情况使用，会记录在 gen_status 栏）
FALLBACK_TEMPLATES = {
    "video": "我想看{kw}",
    "music": "我要听{kw}的歌",
    "city": "{kw}天气怎么样",
}

CHINESE_RE = re.compile(r"[一-鿿]")
NON_CHINESE_RE = re.compile(r"[^一-鿿]")
# 允许出现在 TTS 文本中的标点（会在 manual_transcript 中被去掉）
ALLOWED_PUNCT = set("，。？！、,.?!；;：: ")


def parse_args():
    p = argparse.ArgumentParser(description="以真实训练集每句的关键词造句（口试意见 4）")
    p.add_argument("--train_csv", default=TRAIN_CSV, help="真实训练集 CSV，需含 keyword、source 栏")
    p.add_argument("--output", default=OUTPUT_FILE, help="输出 CSV 路径")
    p.add_argument("--model", default=MODEL_PATH, help="造句用的语言模型")
    p.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    p.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--max_retries", type=int, default=6, help="每句最多重试次数")
    p.add_argument("--max_len", type=int, default=30, help="生成句（纯中文字）最大长度")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="使用 HuggingFace 上的远程模型代码（仅 transformers < 4.49 时需要）")
    return p.parse_args()


def to_label(text: str) -> str:
    """只保留中文字，与真实训练集 manual_transcript 的格式一致"""
    return NON_CHINESE_RE.sub("", text)


def clean_output(raw: str) -> str:
    """取模型输出的第一行，去掉引号、前缀等多余内容"""
    text = raw.strip().split("\n")[0].strip()
    # 去掉模型偶尔加上的「video:」「句子：」之类前缀
    text = re.sub(r"^(video|music|city)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"^(句子|造句|答案)\s*[:：]\s*", "", text)
    text = text.strip("\"'“”‘’「」『』 ")
    return text


def check_sentence(text: str, keyword: str, max_len: int):
    """检查生成句是否合格，返回 (是否合格, 原因)"""
    if not text:
        return False, "empty"
    if keyword not in text:
        return False, "keyword_missing"
    # 只允许中文字 + 常见标点；出现数字/英文会导致 TTS 读法与标注不一致
    for ch in text:
        if not (CHINESE_RE.match(ch) or ch in ALLOWED_PUNCT):
            return False, f"bad_char:{ch}"
    label = to_label(text)
    if len(label) <= len(to_label(keyword)):
        return False, "too_short"      # 只有关键词本身，没有造句
    if len(label) > max_len:
        return False, "too_long"
    return True, "ok"


def build_pipe(model_path, trust_remote_code=False):
    # Phi-4-mini 的架构是 Phi3，transformers >= 4.49 已原生支持，不需要下载远程代码。
    # 远程代码 modeling_phi3.py 会 import 新版 transformers 已移除的 LossKwargs，
    # 因此默认 trust_remote_code=False；若 transformers < 4.49，再加 --trust_remote_code。
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_one(pipe, keyword, domain, args, row_seed, used_labels):
    """为一个 (domain, keyword) 生成一句合格的句子；返回 (tts_text, status, n_try)"""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + FEW_SHOT + [
        {"role": "user", "content": f"{domain}:{keyword}"}
    ]
    gen_args = {
        "max_new_tokens": args.max_new_tokens,
        "return_full_text": False,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    backup = None  # 合格但与已有句子重复的候选，重试用尽时退而求其次
    for attempt in range(args.max_retries):
        torch.manual_seed(row_seed + attempt)
        try:
            out = pipe(messages, **gen_args)
            text = clean_output(out[0]["generated_text"])
        except Exception as e:  # noqa: BLE001
            print(f"  生成出错（第 {attempt + 1} 次）：{e}")
            continue
        ok, reason = check_sentence(text, keyword, args.max_len)
        if not ok:
            print(f"  不合格（{reason}）：{text}")
            continue
        if to_label(text) in used_labels:
            backup = backup or text
            print(f"  与同关键词已有句子重复，重试：{text}")
            continue
        return text, "ok", attempt + 1
    if backup is not None:
        return backup, "duplicate", args.max_retries
    template = FALLBACK_TEMPLATES.get(domain, "{kw}").format(kw=keyword)
    return template, "template", args.max_retries


def main():
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.train_csv)
    for col in ("keyword", "source"):
        if col not in df.columns:
            raise ValueError(f"{args.train_csv} 缺少 {col} 栏")
    # 只保留关键词与领域，刻意丢弃真实文本与音频路径，避免任何真实句子信息进入合成流程
    df = df[["keyword", "source"]].reset_index(drop=True)
    print(f"读取真实训练集 {args.train_csv}：{len(df)} 句")
    print("各领域句数：", df["source"].value_counts().to_dict())
    print(f"不同关键词数：{df['keyword'].nunique()}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 断点续跑
    results = {}
    if os.path.exists(args.output):
        old = pd.read_csv(args.output)
        for _, r in old.iterrows():
            results[int(r["idx"])] = r.to_dict()
        print(f"发现已有输出，已完成 {len(results)} 句，将跳过这些句子")

    # 记录每个关键词已生成过的句子（含续跑读入的），用来避免重复
    used = {}
    for r in results.values():
        used.setdefault(r["keyword"], set()).add(r["manual_transcript"])

    pipe = build_pipe(args.model, args.trust_remote_code)

    def save():
        rows = [results[i] for i in sorted(results)]
        pd.DataFrame(rows, columns=[
            "idx", "manual_transcript", "tts_text", "keyword", "source",
            "gen_status", "n_try",
        ]).to_csv(args.output, index=False)

    n_new = 0
    for idx, row in df.iterrows():
        if idx in results:
            continue
        keyword = str(row["keyword"]).strip()
        domain = str(row["source"]).strip()
        print(f"[{idx + 1}/{len(df)}] {domain}:{keyword}")
        used_labels = used.setdefault(keyword, set())
        tts_text, status, n_try = generate_one(
            pipe, keyword, domain, args, row_seed=args.seed * 100000 + idx * 10, used_labels=used_labels
        )
        label = to_label(tts_text)
        used_labels.add(label)
        results[idx] = {
            "idx": idx,
            "manual_transcript": label,        # ASR 训练标签（纯中文，与真实资料一致）
            "tts_text": tts_text,              # 送进 TTS 的文本（保留标点）
            "keyword": keyword,
            "source": domain,
            "gen_status": status,
            "n_try": n_try,
        }
        print(f"  -> {tts_text}（{status}）")
        n_new += 1
        if n_new % SAVE_EVERY == 0:
            save()
    save()

    out = pd.read_csv(args.output)
    print("\n=== 完成 ===")
    print(f"输出：{args.output}，共 {len(out)} 句（真实训练集 {len(df)} 句）")
    print("各领域句数：", out["source"].value_counts().to_dict())
    print("生成状态：", out["gen_status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
