#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新實驗：地圖導航（map）領域的 zero-shot 測試資料

目的：模型只用 video、music、city 三個領域訓練過；地圖領域完全沒看過。
     把地圖資料「全部」當作測試集，檢驗模型是否學會「從提示詞的關鍵詞清單中找出正確名稱」
     這種通用能力（對應口試意見 7，王新民老師的建議）。

整理方式與原本的 prepare_video_data.py／prepare_audio_data.py／prepare_map_data.py 相同：
  - manual_transcript 只保留中文字（find_chinese）
  - 從 semantic 標註中取出指定槽位的值當作關鍵詞，只保留純中文的關鍵詞（isChinese）
  - keyword.txt 取自 CATSLU 該領域的 lexicon（只保留純中文的詞）
與原程式不同的地方：
  1. 地圖資料不分訓練／測試，train.json、development.json、test.json 全部收進 test.csv
     （欄位 orig_split 記錄原本屬於哪一份）。
  2. 一句話可能同時有起點、終點、途經點，keyword 欄放「主要關鍵詞」
     （優先順序：终点名称 → poi名称 → 途经点名称 → 起点名称，可用 --slots 調整），
     「这里」「家」等指示詞／泛稱，以及少於 2 個字的值不當作關鍵詞（--exclude、--min_len），
     keywords_all 欄放這句全部的地點名稱（以 | 分隔）。
  3. 只保留「關鍵詞原樣出現在 manual_transcript 中」的句子，否則 KER 無法計算；
     被排除的句數會列在統計中。
  4. 地點名稱多半只出現一次，除了 lexicon 版的 keyword.txt，另外輸出
     keyword_from_data.txt（資料中實際出現過的所有地點名稱，去重），供後續決定提示詞用哪一份。

用法（在專案根目錄執行）：
  # 1. 先檢查資料：列出地圖領域有哪些槽位、各槽位的值數量、lexicon 檔案與詞數
  python data/prepare_map_test_data.py --inspect

  # 2. 確認槽位名稱後正式整理
  python data/prepare_map_test_data.py
  python data/prepare_map_test_data.py --slots 终点名称 poi名称 途经点名称 起点名称 --lexicon poi_name.txt

輸出（data/map/）：
  test.csv               path, manual_transcript, keyword, keywords_all, slot, source(=map), orig_split
  keyword.txt            lexicon 版關鍵詞（與其他領域相同的做法）
  keyword_map.txt        與 keyword.txt 相同內容，命名方式與 data/catslu/keyword_{領域}.txt 一致
  keyword_from_data.txt  資料中實際出現過的地點名稱（去重）
  stats.txt              整理過程的統計
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

import pandas as pd

TRAIN_ROOT = "/home/v2129375/dataset/catslu_traindev"
TEST_ROOT = "/home/v2129375/dataset/catslu_test"
OUT_DIR = "data/map"
DEFAULT_SLOTS = ["终点名称", "poi名称", "途经点名称", "起点名称"]
# 指示詞、泛稱，不是具體地名，不當作關鍵詞
DEFAULT_EXCLUDE = ["这里", "那里", "这儿", "那儿", "这边", "那边", "这", "那", "家", "家里", "公司", "单位", "附近", "当前位置", "我的位置"]


def parse_args():
    p = argparse.ArgumentParser(description="整理 CATSLU 地圖領域為 zero-shot 測試資料")
    p.add_argument("--train_root", default=TRAIN_ROOT)
    p.add_argument("--test_root", default=TEST_ROOT)
    p.add_argument("--domain_dir", default="map", help="CATSLU 中地圖領域的資料夾名稱")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS,
                   help="當作關鍵詞的槽位，依優先順序排列（第一個找到的當作主要關鍵詞）")
    p.add_argument("--lexicon", nargs="*", default=None,
                   help="lexicon 檔名（位於 data/map/lexicon/）；不指定時依 ontology.json 中上述槽位對應的詞表")
    p.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE, help="不當作關鍵詞的值（指示詞、泛稱）")
    p.add_argument("--min_len", type=int, default=2, help="關鍵詞最少字數")
    p.add_argument("--inspect", action="store_true", help="只檢查資料結構，不輸出檔案")
    return p.parse_args()


def find_chinese(text):
    return re.sub(r"[^一-龥]", "", str(text))


def is_chinese(word):
    return bool(word) and all("一" <= ch <= "鿿" for ch in word)


def json_files(args):
    """地圖領域所有的標註檔：(原本的 split, json 路徑, 音訊資料夾)"""
    out = []
    for root in (args.train_root, args.test_root):
        ddir = os.path.join(root, "data", args.domain_dir)
        for name in ("train.json", "development.json", "dev.json", "test.json"):
            path = os.path.join(ddir, name)
            if os.path.exists(path):
                out.append((name.replace(".json", ""), path, os.path.join(ddir, "audios")))
    return out


def load_ontology(args):
    for root in (args.train_root, args.test_root):
        path = os.path.join(root, "data", args.domain_dir, "ontology.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return path, json.load(f)
    return None, None


def lexicon_dirs(args):
    return [os.path.join(r, "data", args.domain_dir, "lexicon") for r in (args.train_root, args.test_root)]


def find_lexicon_file(args, name):
    name = os.path.basename(name)
    for d in lexicon_dirs(args):
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    return None


def slot_lexicons(ontology, slots):
    """從 ontology.json 找出槽位對應的 lexicon 檔名（CATSLU 的 informable 槽位值可能是檔案路徑或值列表）"""
    found = {}
    if not isinstance(ontology, dict):
        return found
    stack = [ontology]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k in slots and isinstance(v, str) and v.endswith(".txt"):
                    found[k] = v
                elif k in slots and isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    files = [x for x in v if x.endswith(".txt")]
                    if files:
                        found[k] = files[0]
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return found


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def inspect(args):
    files = json_files(args)
    if not files:
        sys.exit(f"錯誤：在 {args.train_root} 與 {args.test_root} 的 data/{args.domain_dir}/ 下找不到任何 json")
    print("== 標註檔")
    slot_counter = collections.Counter()
    slot_values = collections.defaultdict(set)
    n_utt = 0
    example = None
    for split, path, _ in files:
        data = json.load(open(path, encoding="utf-8"))
        n = sum(len(d["utterances"]) for d in data)
        n_utt += n
        print(f"  {path}：{len(data)} 段對話，{n} 句")
        for d in data:
            for u in d["utterances"]:
                for s in u.get("semantic", []):
                    if len(s) >= 2:
                        slot_counter[s[1]] += 1
                        if len(s) >= 3:
                            slot_values[s[1]].add(s[2])
                        if example is None and len(s) == 3:
                            example = u
    print(f"  合計 {n_utt} 句")
    print("\n== 槽位（出現次數／不同值數量／值範例）")
    for slot, c in slot_counter.most_common():
        vals = list(slot_values[slot])[:6]
        print(f"  {slot}: {c} 次，{len(slot_values[slot])} 種值，例：{vals}")
    if example:
        print("\n== 一句標註範例")
        print("  ", json.dumps(example, ensure_ascii=False)[:400])
    path, onto = load_ontology(args)
    print(f"\n== ontology：{path or '找不到'}")
    if onto is not None:
        print("  ", json.dumps(onto, ensure_ascii=False)[:800])
        print("  槽位 → lexicon：", slot_lexicons(onto, set(slot_counter)))
    print("\n== lexicon 檔案")
    for d in lexicon_dirs(args):
        for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
            lines = read_lines(f)
            print(f"  {f}：{len(lines)} 詞，純中文 {sum(map(is_chinese, lines))} 詞，例：{lines[:5]}")
    print("\n確認要當作關鍵詞的槽位與 lexicon 後，以 --slots 與 --lexicon 指定再正式執行。")


def main():
    args = parse_args()
    if args.inspect:
        inspect(args)
        return

    files = json_files(args)
    if not files:
        sys.exit(f"錯誤：找不到地圖領域的 json，請先用 --inspect 確認路徑與 --domain_dir")
    os.makedirs(args.out_dir, exist_ok=True)
    stats = collections.Counter()
    rows = []
    for split, path, audio_dir in files:
        data = json.load(open(path, encoding="utf-8"))
        for d in data:
            for u in d["utterances"]:
                stats["總句數"] += 1
                transcript = find_chinese(u.get("manual_transcript", ""))
                found = {}  # slot -> [values]
                for s in u.get("semantic", []):
                    if len(s) == 3 and s[1] in args.slots:
                        v = str(s[2]).strip()
                        if not is_chinese(v):
                            stats["關鍵詞含非中文而排除"] += 1
                            continue
                        if v in args.exclude or len(v) < args.min_len:
                            stats["關鍵詞為指示詞/泛稱或太短而排除"] += 1
                            continue
                        found.setdefault(s[1], []).append(v)
                if not found:
                    stats["沒有地點名稱槽位"] += 1
                    continue
                # 主要關鍵詞：依 --slots 的優先順序，且必須原樣出現在文本中
                keyword, slot = "", ""
                for sl in args.slots:
                    for v in found.get(sl, []):
                        if v in transcript:
                            keyword, slot = v, sl
                            break
                    if keyword:
                        break
                if not keyword:
                    stats["關鍵詞未原樣出現在文本中而排除"] += 1
                    continue
                all_kw = list(dict.fromkeys(v for sl in args.slots for v in found.get(sl, [])))
                rows.append({
                    "path": os.path.join(audio_dir, f"{u['wav_id']}.wav"),
                    "manual_transcript": transcript,
                    "keyword": keyword,
                    "keywords_all": "|".join(all_kw),
                    "slot": slot,
                    "source": "map",
                    "orig_split": split,
                })
                stats["保留句數"] += 1
                stats[f"主要關鍵詞槽位：{slot}"] += 1
                if len(all_kw) > 1:
                    stats["一句含多個地點名稱"] += 1

    df = pd.DataFrame(rows)
    missing_audio = int((~df["path"].map(os.path.exists)).sum()) if len(df) else 0
    df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    # 資料中實際出現過的地點名稱
    data_kw = list(dict.fromkeys(k for ks in df["keywords_all"] for k in ks.split("|") if k)) if len(df) else []
    with open(os.path.join(args.out_dir, "keyword_from_data.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(data_kw) + "\n")

    # lexicon 版關鍵詞（與其他領域相同：讀 lexicon，只留純中文）
    lex_files = []
    if args.lexicon:
        lex_files = [find_lexicon_file(args, n) for n in args.lexicon]
    else:
        _, onto = load_ontology(args)
        m = slot_lexicons(onto, set(args.slots)) if onto else {}
        lex_files = list(dict.fromkeys(find_lexicon_file(args, v) for v in m.values()))
    lex_files = [p for p in lex_files if p]
    lexicon = []
    for p in lex_files:
        lexicon += [w for w in read_lines(p) if is_chinese(w) and w not in args.exclude and len(w) >= args.min_len]
    lexicon = list(dict.fromkeys(lexicon))
    if lexicon:
        for name in ("keyword.txt", "keyword_map.txt"):
            with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
                f.write("\n".join(lexicon) + "\n")

    test_kw = set(df["keyword"]) if len(df) else set()
    freq = collections.Counter(df["keyword"]) if len(df) else collections.Counter()
    lines = [f"標註檔：{', '.join(p for _, p, _ in files)}",
             f"關鍵詞槽位（優先順序）：{' > '.join(args.slots)}"]
    lines += [f"{k}：{v}" for k, v in stats.items()]
    lines += [f"音訊檔不存在：{missing_audio} 句",
              f"主要關鍵詞種類數：{len(test_kw)}",
              f"只出現一次的主要關鍵詞：{sum(1 for c in freq.values() if c == 1)} 種",
              f"資料中出現過的地點名稱（keyword_from_data.txt）：{len(data_kw)} 個",
              f"lexicon 檔：{', '.join(lex_files) if lex_files else '未找到（未輸出 keyword.txt，請用 --lexicon 指定）'}",
              f"lexicon 版關鍵詞（keyword.txt）：{len(lexicon)} 個"]
    if lexicon:
        cov = sum(1 for k in df["keyword"] if k in set(lexicon)) / max(len(df), 1)
        lines.append(f"測試句主要關鍵詞被 lexicon 涵蓋的比例：{cov:.2%}")
    lines.append("各原始 split 句數：" + str(df["orig_split"].value_counts().to_dict() if len(df) else {}))
    with open(os.path.join(args.out_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n已輸出到 {args.out_dir}/")


if __name__ == "__main__":
    main()
