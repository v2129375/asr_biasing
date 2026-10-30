#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口試意見 6：以「較小的 keyword list」微調（每句保證包含正確關鍵詞）

用途：訓練時每一句只放 N 個關鍵詞（例如 100、200），測試時改用完整的原 keyword list，
驗證「訓練時與推論時 list 大小不一致」的影響。

與原程式 finetune_catslu_keywords.py 的唯一差別是「每一句怎麼選關鍵詞」：
  原程式 --num_keywords N：從該領域全部關鍵詞隨機抽 N 個，不保證包含正確關鍵詞
                         （video 只有約 N/555 的機率抽到）。
  本程式 --num_keywords N：該句的正確關鍵詞 ＋ 從其餘關鍵詞隨機抽 N−1 個，再打亂順序。
                         每個 epoch、每一句都重新抽，因此模型會看過各種組合。
這樣訓練時與測試時的正確關鍵詞涵蓋率都是 100%，比較時只剩「list 大小」一個變因。

其餘所有設定（模型、LoRA、bf16、gradient checkpointing、提示詞格式、超參數、
你在原程式加入的 num_logits_to_keep 修改、attention 設定）都直接沿用原程式：
本程式在執行時載入原程式，只替換選關鍵詞的那一步，原程式檔案不做任何修改。

用法（在專案根目錄執行，參數與原程式完全相同）：
  python asr/finetune/finetune_catslu_keywords_subset.py \
      --data_path data/catslu/train.csv --keywords_dir data/catslu \
      --num_keywords 100 --output_dir asr/model/Sub100Keywords --gpu_ids 0
"""

import os
import random
import sys

import pandas as pd

# 載入同資料夾的原訓練程式（它在 import 時就會解析命令列參數，參數直接沿用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_catslu_keywords as base  # noqa: E402

if base.NUM_KEYWORDS <= 0:
    sys.exit("錯誤：請用 --num_keywords 指定每句的關鍵詞數量，例如 --num_keywords 100")

_orig_getitem = base.CatsluKeywordsDataset.__getitem__


def _getitem(self, idx):
    """先記下這一句的正確關鍵詞，再交給原本的 __getitem__ 處理"""
    kw = self.data.iloc[idx].get("keyword", "")
    self._current_keyword = "" if pd.isna(kw) else str(kw).strip()
    return _orig_getitem(self, idx)


def _get_domain_keywords(self, source):
    """正確關鍵詞 ＋ 隨機 N−1 個其他關鍵詞，打亂順序"""
    if source not in self.keywords_dict:
        print(f"Warning: No keywords found for source '{source}', using empty keyword list")
        return []
    pool = list(dict.fromkeys(self.keywords_dict[source]))  # 去重，保留順序
    n = min(base.NUM_KEYWORDS, len(pool))
    target = getattr(self, "_current_keyword", "")
    if target:
        others = [w for w in pool if w != target]
        chosen = [target] + random.sample(others, min(n - 1, len(others)))
    else:  # 沒有關鍵詞欄位時退回原程式的做法
        chosen = random.sample(pool, n)
    random.shuffle(chosen)
    return chosen


base.CatsluKeywordsDataset.__getitem__ = _getitem
base.CatsluKeywordsDataset._get_domain_keywords = _get_domain_keywords

if __name__ == "__main__":
    print(f"[subset] 每句關鍵詞數 = {base.NUM_KEYWORDS}（保證包含該句正確關鍵詞）")
    base.main()
