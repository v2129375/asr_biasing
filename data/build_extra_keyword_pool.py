#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 5（Keyword list 規模的影響）—— 步驟 0：建立各領域的「額外關鍵詞池」

目的：現有 keyword list 每個領域只有幾百個（去重後 video 552、music 389、city 810），
為了把 list 擴大到 1000 / 2000 / 3000，需要大量「不在原 list 中」的同類名稱當作干擾詞。
本程式從公開資料整理真實名稱，不足的部分（video）再以 AI 生成的影視名稱補足：

  領域   來源（優先使用真實名稱）
  music  華語歌手名：dengxiuqi/ChineseLyrics（4,019 位歌手，2019 年前的華語歌曲）
  city   中國行政區劃名稱：modood/Administrative-divisions-of-China（地級市、縣、縣級市、市轄區）
  video  電影片名：bbbbbbbin/MOVIE_DATA（2010–2017 年中國上映電影）、XueXiangKM/douban-movies（豆瓣高分電影）
         ＋ AI 生成的電視劇、綜藝、動畫、紀錄片、兒歌等名稱（llm_video_titles.txt，於 source 欄標註 llm）

處理規則：
  - 只保留純中文名稱（與 CATSLU 前處理 isChinese 的規則一致），長度 2～10 字；
  - 去除與原 keyword list 完全相同的名稱；city 另外比對去掉「市/縣/區/自治州…」後綴的名稱，
    避免「泰安」與「泰安市」這種同一地點的重複；
  - 以固定種子打亂順序後輸出。後續 make_keyword_scale_lists.py 依序從池中取詞，
    因此 1000 ⊂ 2000 ⊂ 3000（巢狀），且每個規模中干擾詞的來源比例相同，只有「數量」改變。

取得資料（在任一可以連 GitHub 的電腦執行；本專案已附上整理好的結果，一般不需重跑）：
  git clone --depth 1 https://github.com/dengxiuqi/ChineseLyrics.git
  git clone --depth 1 https://github.com/bbbbbbbin/MOVIE_DATA.git
  git clone --depth 1 https://github.com/XueXiangKM/douban-movies.git
  mkdir admin && for f in cities areas; do \
    curl -L -o admin/$f.csv https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/master/dist/$f.csv; done

用法：
  python data/build_extra_keyword_pool.py \
      --lyrics_dir ChineseLyrics --movie_dir MOVIE_DATA/data/movie \
      --douban_csv douban-movies/movies.csv --admin_dir admin \
      --llm_video data/keyword_pool/llm_video_titles.txt \
      --orig_dir data/catslu --out_dir data/keyword_pool
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import re

CH = re.compile(r"^[一-鿿]+$")
CITY_SUFFIX = re.compile(r"(市|县|区|自治州|自治县|自治旗|地区|盟|旗|林区|特区|州)$")
SKIP_ADMIN = {"市辖区", "县", "省直辖县级行政区划", "自治区直辖县级行政区划"}
DOMAINS = ["video", "music", "city"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lyrics_dir", required=True)
    p.add_argument("--movie_dir", required=True)
    p.add_argument("--douban_csv", required=True)
    p.add_argument("--admin_dir", required=True)
    p.add_argument("--llm_video", required=True)
    p.add_argument("--orig_dir", default="data/catslu")
    p.add_argument("--out_dir", default="data/keyword_pool")
    p.add_argument("--min_len", type=int, default=2)
    p.add_argument("--max_len", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def ok(name, args):
    return bool(CH.match(name)) and args.min_len <= len(name) <= args.max_len


def stem(name):
    return CITY_SUFFIX.sub("", name)


def main():
    args = parse_args()
    orig = {}
    for d in DOMAINS:
        with open(os.path.join(args.orig_dir, f"keyword_{d}.txt"), encoding="utf-8") as f:
            orig[d] = {l.strip() for l in f if l.strip()}
    orig_city_stems = {stem(x) for x in orig["city"]} | orig["city"]

    pool = {d: {} for d in DOMAINS}  # name -> source

    # music：歌手名（依作品數排序只是參考，最後會打亂）
    cnt = collections.Counter()
    for f in sorted(glob.glob(os.path.join(args.lyrics_dir, "lyrics*.json"))):
        with open(f, encoding="utf-8") as fh:
            for x in json.load(fh):
                cnt[str(x.get("singer", "")).strip()] += 1
    for name, _ in cnt.most_common():
        if ok(name, args) and name not in orig["music"]:
            pool["music"].setdefault(name, "ChineseLyrics_singer")

    # video：真實電影片名
    for f in sorted(glob.glob(os.path.join(args.movie_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            for x in json.load(fh).values():
                name = str(x.get("title", "")).strip()
                if ok(name, args) and name not in orig["video"]:
                    pool["video"].setdefault(name, "MOVIE_DATA_2010_2017")
    with open(args.douban_csv, encoding="gbk", errors="ignore") as fh:
        for row in csv.reader(fh):
            name = row[0].strip() if row else ""
            if ok(name, args) and name not in orig["video"]:
                pool["video"].setdefault(name, "douban_movies")
    # video：AI 生成的影視名稱（補足電視劇、綜藝、動畫等）
    with open(args.llm_video, encoding="utf-8") as fh:
        for line in fh:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            if ok(name, args) and name not in orig["video"]:
                pool["video"].setdefault(name, "llm")

    # city：行政區劃名稱
    for fname in ("cities", "areas"):
        with open(os.path.join(args.admin_dir, f"{fname}.csv"), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = row["name"].strip()
                if name in SKIP_ADMIN or not ok(name, args):
                    continue
                if name in orig["city"] or stem(name) in orig_city_stems:
                    continue
                pool["city"].setdefault(name, f"admin_{fname}")

    # 固定種子打亂後輸出
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for d in DOMAINS:
        names = sorted(pool[d])
        rng.shuffle(names)
        with open(os.path.join(args.out_dir, f"extra_{d}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        rows += [(n, d, pool[d][n]) for n in names]
        src = collections.Counter(pool[d].values())
        print(f"{d}: 原 list {len(orig[d])} 個（去重），額外關鍵詞池 {len(names)} 個，來源 {dict(src)}")
    with open(os.path.join(args.out_dir, "extra_keyword_sources.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["keyword", "domain", "source"])
        w.writerows(rows)
    print(f"已輸出到 {args.out_dir}/")


if __name__ == "__main__":
    main()
