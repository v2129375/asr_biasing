#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口试意见 4（实验五变因控制）—— 第 2 步：用 F5-TTS 合成「造出来的句子」，语速参考真实语料

情境：模拟「目标领域只有关键词、没有文本」。
  输入是 make_sentences_from_train.py 的输出（以真实训练集每句的关键词造出的 1511 句），
  本程序只合成这些造出来的句子，不使用真实训练集的文本，也不使用逐句的真实音频信息。

与旧版 tts/F5_TTS.py 的差别：
  1. 语速控制（意见 4 的重点）：
     旧版没有控制语速，F5-TTS 依参考音频的语速推算长度；而且 F5-TTS 内部对很短的句子
     （utf-8 少于 10 bytes，约 3 个汉字以下）会自动把语速降到 0.3，容易产生过慢的音频。
     新版先算出真实训练集的「整体平均语速」（总字数 / 总秒数，只是语料层级的统计量），
     再让每一句的长度 = 该句字数 / 平均语速（用 F5-TTS 的 fix_duration 控制）。
     由于造句的筆数、关键词与真实训练集相同，合成资料的总时数会与真实资料（约 1.5 小时）相近；
     如需让总时数更精确地一致，可加 --match_total_hours（会微调整体语速并印出调整后的语速）。
  2. 参考说话人：
       single —— 单一说话人（论文做法：从训练集选一位咬字清晰的女声），需给 --ref_audio 与 --ref_text
       random —— 每句随机从 --ref_csv（默认训练集，而不是旧版的 test.csv）挑一位说话人当音色，
                 并用 CSV 中的 manual_transcript 当参考文本（只用于音色提示，不是合成内容）
     有提供参考文本时不会再调用 Whisper 转写参考音频（旧版 REF_TEXT="" 会触发自动转写）。
  3. 只载入一次模型（旧版每一句都用 subprocess 调用 f5-tts_infer-cli，每句重新载入模型）。
  4. 输出音频重采样为 16 kHz（与 CATSLU 真实音频相同），并支持断点续跑。

用法（在 repo 根目录执行，先执行 make_sentences_from_train.py）：
  python tts/F5_TTS_matched.py \
      --input_csv tts/tts_data/sentences_train_kw.csv \
      --ref_mode single --ref_audio /path/to/female_ref.wav --ref_text "参考音频的文字" \
      --output_audio_dir /home/v2129375/dataset/tts_audio_matched \
      --output_csv tts/tts_data/sentences_audio_matched.csv
"""

import argparse
import os
import random
import re
import sys

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ===================== 默认配置 =====================
INPUT_CSV = "tts/tts_data/sentences_train_kw.csv"
OUTPUT_AUDIO_DIR = "/home/v2129375/dataset/tts_audio_matched"
OUTPUT_CSV = "tts/tts_data/sentences_audio_matched.csv"
F5TTS_MODEL = "F5TTS_v1_Base"
RATE_CSV = "data/catslu/train.csv"  # 用来计算真实语料平均语速（只取整体统计量）
REF_CSV = "data/catslu/train.csv"   # random 模式的参考说话人来源（训练集，不要用测试集）
# ====================================================

NON_CHINESE_RE = re.compile(r"[^一-鿿]")


def parse_args():
    p = argparse.ArgumentParser(description="F5-TTS 合成造句结果（语速参考真实语料，口试意见 4）")
    p.add_argument("--input_csv", default=INPUT_CSV, help="make_sentences_from_train.py 的输出")
    p.add_argument("--output_audio_dir", default=OUTPUT_AUDIO_DIR)
    p.add_argument("--output_csv", default=OUTPUT_CSV)
    p.add_argument("--model", default=F5TTS_MODEL)
    p.add_argument("--ckpt_file", default="", help="本地 F5-TTS 权重（可选）")
    p.add_argument("--vocab_file", default="", help="本地 vocab（可选）")

    # 参考说话人
    p.add_argument("--ref_mode", choices=["single", "random"], default="single")
    p.add_argument("--ref_audio", default="", help="single 模式：参考音频路径")
    p.add_argument("--ref_text", default="", help="single 模式：参考音频的文字内容（强烈建议提供）")
    p.add_argument("--ref_csv", default=REF_CSV, help="random 模式：参考说话人 CSV（需含 path、manual_transcript）")
    p.add_argument("--ref_min_dur", type=float, default=2.0, help="random 模式：参考音频最短秒数")
    p.add_argument("--ref_max_dur", type=float, default=10.0, help="random 模式：参考音频最长秒数")

    # 语速控制
    p.add_argument("--duration_mode", choices=["rate", "speed"], default="rate",
                   help="rate：依真实语料平均语速决定每句长度；speed：不控制时长，只用 --speed（旧版行为）")
    p.add_argument("--rate_csv", default=RATE_CSV,
                   help="计算平均语速用的真实语料 CSV（需含 path、manual_transcript）")
    p.add_argument("--chars_per_sec", type=float, default=0.0,
                   help="直接指定平均语速（字/秒）；>0 时不读取 --rate_csv")
    p.add_argument("--match_total_hours", action="store_true",
                   help="微调整体语速，使合成总时数 = 真实语料总时数（会印出调整后的语速）")
    p.add_argument("--speed", type=float, default=1.0, help="speed 模式下的语速")
    p.add_argument("--min_dur", type=float, default=0.8, help="合成音频最短秒数")

    # 其他
    p.add_argument("--out_sr", type=int, default=16000, help="输出采样率（CATSLU 为 16 kHz；0 表示保持 24 kHz）")
    p.add_argument("--nfe_step", type=int, default=32)
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--remove_silence", action="store_true",
                   help="去除合成音频中的静音（会改变时长，rate 模式下不建议开启）")
    p.add_argument("--overwrite", action="store_true", help="重新合成已存在的音频")
    return p.parse_args()


def to_label(text: str) -> str:
    return NON_CHINESE_RE.sub("", str(text))


def audio_duration(path):
    try:
        return sf.info(path).duration
    except Exception:  # noqa: BLE001
        return None


def real_corpus_stats(csv_path):
    """真实语料的整体统计：总字数、总秒数（只用于语速，不涉及任何逐句对应）"""
    df = pd.read_csv(csv_path)
    total_chars, total_sec, missing = 0, 0.0, 0
    for _, r in df.iterrows():
        d = audio_duration(r["path"])
        if d is None:
            missing += 1
            continue
        total_chars += len(to_label(r["manual_transcript"]))
        total_sec += d
    if total_sec == 0:
        sys.exit(f"错误：无法从 {csv_path} 读取任何真实音频，请确认 path 栏的音频存在，或改用 --chars_per_sec")
    if missing:
        print(f"警告：{csv_path} 中有 {missing} 条音频读取失败，已略过")
    return total_chars, total_sec


def load_input(csv_path):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    for col in ("manual_transcript", "source"):
        if col not in df.columns:
            sys.exit(f"错误：输入 CSV 缺少 {col} 栏（请使用 make_sentences_from_train.py 的输出）")
    if "tts_text" not in df.columns:
        df["tts_text"] = df["manual_transcript"]
    if "keyword" not in df.columns:
        df["keyword"] = ""
    df["manual_transcript"] = df["manual_transcript"].map(to_label)
    return df


class RefPicker:
    """根据 ref_mode 为每一句提供 (参考音频, 参考文本)"""

    def __init__(self, args):
        self.mode = args.ref_mode
        self.rng = random.Random(args.seed)
        if self.mode == "single":
            if not args.ref_audio or not os.path.exists(args.ref_audio):
                sys.exit("错误：single 模式需要 --ref_audio（存在的音频文件）")
            if not args.ref_text.strip():
                print("警告：未提供 --ref_text，F5-TTS 会用 Whisper 自动转写参考音频，"
                      "建议手动提供以免转写错误影响合成")
            self.single = (args.ref_audio, args.ref_text)
        else:
            ref = pd.read_csv(args.ref_csv)
            if "path" not in ref.columns or "manual_transcript" not in ref.columns:
                sys.exit("错误：--ref_csv 需含 path 与 manual_transcript 栏")
            cands = []
            for _, r in ref.iterrows():
                d = audio_duration(r["path"])
                if d is not None and args.ref_min_dur <= d <= args.ref_max_dur:
                    cands.append((r["path"], str(r["manual_transcript"])))
            if not cands:
                sys.exit("错误：--ref_csv 中没有符合时长条件的参考音频")
            print(f"random 模式：可用参考说话人音频 {len(cands)} 条")
            self.cands = cands

    def get(self):
        if self.mode == "single":
            return self.single
        return self.rng.choice(self.cands)


def main():
    args = parse_args()

    df = load_input(args.input_csv)
    print(f"输入 {args.input_csv}：{len(df)} 句；各领域：{df['source'].value_counts().to_dict()}")

    # ---------- 决定语速 ----------
    rate = None
    real_hours = None
    if args.duration_mode == "rate":
        if args.chars_per_sec > 0:
            rate = args.chars_per_sec
            print(f"使用指定的平均语速：{rate:.3f} 字/秒")
        else:
            real_chars, real_sec = real_corpus_stats(args.rate_csv)
            rate = real_chars / real_sec
            real_hours = real_sec / 3600
            print(f"真实语料 {args.rate_csv}：{real_chars} 字，{real_hours:.3f} 小时，"
                  f"平均语速 {rate:.3f} 字/秒（含句首尾静音）")
        if args.match_total_hours:
            if real_hours is None:
                sys.exit("错误：--match_total_hours 需要从 --rate_csv 计算真实总时数（不要同时指定 --chars_per_sec）")
            syn_chars = df["manual_transcript"].map(len).sum()
            rate = syn_chars / (real_hours * 3600)
            print(f"--match_total_hours：造句共 {syn_chars} 字，调整后语速 {rate:.3f} 字/秒，"
                  f"使总时数 ≈ {real_hours:.3f} 小时")

    # F5-TTS 在这里才 import，避免只看 --help 时也要载入 torch
    import torch
    import torchaudio
    from f5_tts.api import F5TTS
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    os.makedirs(args.output_audio_dir, exist_ok=True)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    picker = RefPicker(args)
    tts = F5TTS(model=args.model, ckpt_file=args.ckpt_file, vocab_file=args.vocab_file)
    quiet = lambda *a, **k: None  # noqa: E731

    ref_cache = {}  # 原始参考音频 -> (处理后路径, 处理后文本, 处理后秒数)

    def prepare_ref(ref_audio, ref_text):
        key = (ref_audio, ref_text)
        if key not in ref_cache:
            p_audio, p_text = preprocess_ref_audio_text(ref_audio, ref_text, show_info=quiet)
            ref_cache[key] = (p_audio, p_text, sf.info(p_audio).duration)
        return ref_cache[key]

    rows, failed = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="合成音频"):
        label = row["manual_transcript"]
        tts_text = str(row["tts_text"]).strip()
        wav_path = os.path.join(args.output_audio_dir, f"tts_{idx:04d}.wav")

        target = max(len(label) / rate, args.min_dur) if rate else None
        ref_audio, ref_text = picker.get()  # 每句都抽一次，保证续跑时说话人分配不变

        if args.overwrite or audio_duration(wav_path) is None:
            try:
                p_audio, p_text, ref_sec = prepare_ref(ref_audio, ref_text)
                # fix_duration = 参考音频长度 + 希望生成的长度（F5-TTS 的定义）
                fix_duration = ref_sec + target if target is not None else None
                wav, sr, _ = tts.infer(
                    ref_file=p_audio,
                    ref_text=p_text,
                    gen_text=tts_text,
                    show_info=quiet,
                    progress=None,
                    nfe_step=args.nfe_step,
                    cfg_strength=args.cfg_strength,
                    speed=args.speed,
                    fix_duration=fix_duration,
                    seed=args.seed + idx,
                )
                if wav is None:
                    raise RuntimeError("F5-TTS 未返回音频")
                wav_t = torch.as_tensor(wav, dtype=torch.float32).reshape(1, -1)
                if args.out_sr and args.out_sr != sr:
                    wav_t = torchaudio.functional.resample(wav_t, sr, args.out_sr)
                    sr = args.out_sr
                sf.write(wav_path, wav_t.squeeze(0).numpy(), sr, subtype="PCM_16")
                if args.remove_silence:
                    from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
                    remove_silence_for_generated_wav(wav_path)
            except Exception as e:  # noqa: BLE001
                failed.append((idx, str(e)))
                print(f"\n第 {idx} 句合成失败：{e}")
                continue

        rows.append({
            "path": wav_path,
            "manual_transcript": label,
            "keyword": row["keyword"],
            "source": row["source"],
            "tts_text": tts_text,
            "target_duration": round(target, 3) if target else None,
            "tts_duration": round(audio_duration(wav_path), 3),
            "ref_audio": ref_audio,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)

    print("\n=== 合成完成 ===")
    print(f"输出 CSV：{args.output_csv}（{len(out)}/{len(df)} 句）")
    if len(out):
        tts_sec = out["tts_duration"].sum()
        print(f"合成音频总时数：{tts_sec / 3600:.3f} 小时"
              + (f"（真实语料 {real_hours:.3f} 小时）" if real_hours else ""))
        print(f"合成音频平均语速：{out['manual_transcript'].map(len).sum() / tts_sec:.3f} 字/秒")
        print("各领域句数：", out["source"].value_counts().to_dict())
    if failed:
        print(f"警告：{len(failed)} 句失败（重新执行本脚本会自动补跑缺少的句子）：")
        for i, msg in failed[:20]:
            print(f"  第 {i} 句：{msg}")


if __name__ == "__main__":
    main()
