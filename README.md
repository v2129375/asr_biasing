# 基於多模態大型語言模型改善中文語音辨識之語境偏移效果

本專案以 **Phi-4 Multimodal** 為核心，在 CATSLU 資料集（video / music / city 三個領域）上建立一套語境偏移（contextual biasing）語音辨識系統。系統分成三個部分：

1. **意圖辨識（Intent）**：輸入語音，判斷屬於 video、music 或 city 哪一個領域；
2. **語音辨識（ASR）**：依意圖辨識的結果，把該領域的關鍵詞放進提示詞（動態關鍵詞提示），引導模型正確辨識專有名詞；
3. **語音合成（TTS）**：在沒有目標領域語料、只有關鍵詞的情況下，用 Phi-4-mini 造句、F5-TTS 合成語音，產生訓練資料。

評估指標：意圖辨識看各類別與總體正確率；ASR 看字錯誤率 **CER** 與關鍵詞錯誤率 **KER**（程式中名為 `keyword_wer`）。

> 所有指令都在專案根目錄執行，例如 `/home/v2129375/asr_biasing`。
> 多數程式的路徑寫在程式開頭的常數或 `argparse` 預設值中，下文會標出需要修改的檔案與行號。

---

## 目錄結構（主要部分）

```
asr_biasing/
├── data/
│   ├── prepare_video_data.py / prepare_audio_data.py / prepare_map_data.py   # 從 CATSLU 擷取三個領域
│   ├── combine_catslu_data.py                  # 合併三個領域並切分訓練/測試集
│   └── catslu/
│       ├── train.csv（1,511 句） test.csv（378 句）
│       └── keyword_video.txt / keyword_music.txt / keyword_city.txt   # 各領域關鍵詞清單
├── intent/
│   ├── phi4.py                                  # 未微調模型推論
│   ├── phi4_local.py                            # 微調後模型推論
│   ├── intent_score.py                          # 評分
│   ├── finetune_phi4/sample_finetune_speech_intent.py   # 意圖辨識微調
│   ├── model/                                   # 微調後的模型
│   └── exp/                                     # 推論結果
├── asr/
│   ├── phi4.py                                  # 未微調模型推論（基線）
│   ├── phi4_local.py                            # 一般微調模型推論（基線，無關鍵詞）
│   ├── phi4_keywords.py                         # 本方法：動態關鍵詞提示推論＋自動評分
│   ├── score.py                                 # CER / KER 評分
│   ├── finetune/sample_finetune_speech_asr.py   # 一般微調（基線）
│   ├── finetune/finetune_catslu_keywords.py     # 本方法：動態關鍵詞提示微調
│   ├── model/                                   # 微調後的模型
│   └── exp/                                     # 推論與評分結果
└── tts/
    ├── phi4mini_make_sentance.py / F5_TTS.py    # 舊版合成流程（論文原始實驗五）
    ├── make_sentences_from_train.py             # 新版：以訓練集每句的關鍵詞造句
    ├── F5_TTS_matched.py                        # 新版：F5-TTS 合成，語速對齊真實語料
    ├── compare_real_tts_stats.py                # 新版：真實 vs. 合成資料統計
    └── tts_data/                                # 造句與合成結果 CSV
```

資料 CSV 的共同欄位：`path`（音訊路徑）、`manual_transcript`（文字標註）、`keyword`（該句關鍵詞）、`source`（領域）。

---

## 1. 環境安裝

Phi-4 Multimodal 的程式碼需要 `transformers==4.48.2`，而 F5-TTS 安裝時會把 transformers 升級到新版。**建議建立兩個 conda 環境**，避免互相影響。

### 1.1 `phi4` 環境：意圖辨識與 ASR 的微調、推論

```bash
conda create -n phi4 python=3.12 -y
conda activate phi4
# 先依 CUDA 版本安裝 PyTorch（見 https://pytorch.org）
pip install torch torchvision torchaudio
pip install -r phi4_requirement.txt     # transformers==4.48.2、peft、accelerate、flash_attn 等
pip install pandas numpy scikit-learn tqdm requests
```

- `flash_attn` 需要 Ampere 以上的 GPU（RTX 30 系列以後）。若無法使用，推論程式中的 `_attn_implementation='flash_attention_2'` 需改為 `'sdpa'` 或 `'eager'`。
- 預設使用兩張 GPU，模型分配寫在 `intent/finetune_phi4/device_map.json` 與 `asr/finetune/device_map.json`。

### 1.2 `tts` 環境：造句與語音合成

```bash
conda create -n tts python=3.12 -y
conda activate tts
pip install torch torchaudio            # 依 CUDA 版本安裝
pip install f5-tts pandas soundfile tqdm
pip install "transformers>=4.49" accelerate   # Phi-4-mini 造句用
```

- F5-TTS 的模型權重第一次執行時會自動從 HuggingFace 下載。
- 若 wandb 匯入出錯，`F5_TTS_matched.py` 會自動略過（合成不需要 wandb）；也可執行 `pip install --upgrade --force-reinstall wandb` 修復。

### 1.3 資料準備

1. 下載 CATSLU 資料集，解壓到 `/home/v2129375/dataset/catslu_traindev` 與 `/home/v2129375/dataset/catslu_test`。
2. 依序執行以下程式。若資料集放在其他位置，請先修改各程式開頭的 `train_data_root`、`test_data_root`。

```bash
python data/prepare_video_data.py     # → data/video/
python data/prepare_audio_data.py     # → data/music/（歌手名）
python data/prepare_map_data.py       # → data/city/（天氣領域的城市名）
python data/combine_catslu_data.py    # → data/catslu/train.csv、test.csv、keyword_*.txt
```

`combine_catslu_data.py` 會依領域做分層抽樣，80% 為訓練集、20% 為測試集（`random_state=42`）。

---

## 2. 意圖辨識

使用 `phi4` 環境。

### 2.1 未微調模型推論（基線）

```bash
python intent/phi4.py
```

讀取 `data/catslu/test.csv`，結果寫到 `intent/exp/phi4_intent_result.csv`（第 101 行 `output_filename`），新增一欄 `predict`。

### 2.2 微調

程式：`intent/finetune_phi4/sample_finetune_speech_intent.py`

| 行號 | 參數 | 說明 |
|---|---|---|
| 445 | `--catslu_data_path` | 訓練資料 CSV；以 `source` 欄作為意圖標籤 |
| 451 | `--eval_size` | 從訓練資料中隨機抽出多少筆當驗證集（預設 200） |
| 455 | `--output_dir` | 模型輸出資料夾（預設 `intent/model/new/`） |
| 464 / 466 | `--num_train_epochs` / `--learning_rate` | 訓練輪數 / 學習率 |

```bash
python intent/finetune_phi4/sample_finetune_speech_intent.py
```

- 訓練前、後會各在驗證集上評估一次，結果存到 `<output_dir>/eval_before.json`、`eval_after.json`。這只是訓練資料內的驗證，正式成績請看 2.3、2.4。
- 驗證集是隨機抽取的，訓練期間會在訓練 CSV 同一個資料夾暫存 `train_temp.csv`、`eval_temp.csv`。
- 每次訓練請指定不同的 `--output_dir`，避免覆蓋之前的模型。

### 2.3 微調後模型推論

程式：`intent/phi4_local.py`

| 行號 | 內容 |
|---|---|
| 22 | `model_path`：要載入的微調模型資料夾，需與 2.2 的 `--output_dir` 相同 |
| 103 | `input_data_path`：測試資料夾（讀取其中的 `test.csv`） |
| 105 | `output_filename`：結果檔名（寫到 `intent/exp/`） |

```bash
python intent/phi4_local.py
```

逐句推論，從輸出擷取 `video|music|city`（擷取不到時預設為 `city`），存入 `predict` 欄。**這個結果檔也是 ASR 推論（3.4）的輸入。**

### 2.4 評分

程式：`intent/intent_score.py`，修改第 5 行 `input_data_path` 為 2.1 或 2.3 的結果檔。

```bash
python intent/intent_score.py
```

印出 video / music / city 各類別正確率與總體正確率。

---

## 3. 語音辨識（ASR）

使用 `phi4` 環境。

### 3.1 未微調模型推論（基線）

```bash
python asr/phi4.py
```

提示詞為 `Transcribe the audio clip into text.`，讀取 `data/catslu/test.csv`，結果寫到 `asr/exp/phi4_ori_asr_result.csv`（第 77 行）。評分見 3.5。

### 3.2 一般微調（基線，不使用關鍵詞）

- 微調：`asr/finetune/sample_finetune_speech_asr.py`。訓練資料為第 500 行 `--catslu_data_path`（預設 `data/catslu/train.csv`），模型輸出為第 510 行 `--output_dir`（預設 `asr/model/`）。
- 推論：`asr/phi4_local.py`。第 43 行 `model_path` 為模型資料夾，第 105 行為測試資料，第 106 行為結果檔 `asr/exp/phi4_finetuned_asr_result.csv`。

```bash
python asr/finetune/sample_finetune_speech_asr.py
python asr/phi4_local.py
```

評分見 3.5。

### 3.3 本方法：動態關鍵詞提示微調

程式：`asr/finetune/finetune_catslu_keywords.py`

| 行號 | 參數 | 說明 |
|---|---|---|
| 36 | `--gpu_ids` | 使用的 GPU（預設 `[0,1]`；多張時依 `asr/finetune/device_map.json` 分配） |
| 38 | `--data_path` | 訓練資料 CSV（真實資料為 `data/catslu/train.csv`） |
| 39 | `--keywords_dir` | 關鍵詞清單資料夾（預設 `data/catslu`） |
| 41 | `--output_dir` | 模型輸出資料夾；**資料夾名稱會成為 3.4 結果檔的檔名** |
| 42 / 43 | `--batch_size` / `--batch_size_per_gpu` | 總批次需能被「GPU 數 × 每張 GPU 批次」整除 |
| 44 / 45 | `--num_train_epochs` / `--learning_rate` | 訓練輪數 / 學習率 |
| 50 | `--num_keywords` | 每句提示中放入的關鍵詞數量；0 表示放入該領域全部關鍵詞 |
| 51 | `--num_sentences` | 訓練使用的句數；0 表示全部 |

```bash
python asr/finetune/finetune_catslu_keywords.py
```

訓練時依每句的 `source`（真實領域）從 `keyword_{領域}.txt` 取出關鍵詞，組成提示
`Transcribe the audio clip into text. Pay attention to these keywords: 關鍵詞1 關鍵詞2 ...`，
並以 LoRA（`speech` adapter）微調。此程式不切驗證集，全部句子都用於訓練。

### 3.4 本方法：推論＋評分

程式：`asr/phi4_keywords.py`

| 行號 | 內容 |
|---|---|
| 23 | `NUM_KEYWORDS`：需與訓練時的 `--num_keywords` 一致 |
| 24 | `input_data_path`：**意圖辨識結果檔**（2.3 的輸出，需含 `predict` 欄） |
| 25 | `model_path`：3.3 訓練出的模型資料夾 |

```bash
python asr/phi4_keywords.py
```

- 推論時依 **`predict` 欄（預測的領域）** 選擇關鍵詞放入提示，模擬完整系統「意圖辨識 → 動態關鍵詞提示 → ASR」的串接；意圖預測錯誤時，提示中就會是錯誤領域的關鍵詞。
- 輸出會去除標點，並把阿拉伯數字轉成中文數字。
- 結果寫到 `asr/exp/<model_path 資料夾名>.csv`，並**自動評分**，寫到同名 `.json`。
- 也可用命令列參數 `--input`、`--model_path`、`--num_keywords`、`--gpus` 指定，不必改程式。`--use_keywords` 為 `type=bool`，寫 `False` 仍會被視為 True，請勿使用。

### 3.5 評分

程式：`asr/score.py`，函式 `evaluate_asr`。

- 3.4 會自動呼叫，不需另外執行。
- 3.1、3.2 的結果需手動評分：修改第 205 行 `input_file` 為要評分的 CSV，再執行 `python asr/score.py`。

評分內容：

- **CER**：逐句字錯誤率的平均；
- **KER（`keyword_wer`）**：ASR 結果中沒有完整出現該句關鍵詞的比例；
- 另外提供總體及 video / music / city 各類別的數值，並印出關鍵詞辨識錯誤的句子。結果存成與輸入同名的 `.json`。

---

## 4. 語音合成（實驗五：只有關鍵詞時產生訓練資料）

使用 `tts` 環境產生資料，再切回 `phi4` 環境訓練與測試。

本版本依口試意見 4 修正：**只取真實訓練集每一句的關鍵詞與領域**（不使用真實文本與音訊）來造句與合成，使合成資料的筆數（1,511）、每句關鍵詞、各領域句數、總時數都與真實訓練集一致。這樣比較時只剩「真實 vs. 合成」一個變因。
（舊版 `phi4mini_make_sentance.py` 是對 `keyword_*.txt` 的全部 1,875 個關鍵詞各造一句，筆數與關鍵詞都和真實訓練集不同。）

### 4.1 造句

```bash
conda activate tts
python tts/make_sentences_from_train.py
```

- 輸入：`data/catslu/train.csv`，只讀取 `keyword`、`source` 兩欄。
- 方法：Phi-4-mini 加上 few-shot 提示（與論文相同），依「領域:關鍵詞」造句。句子必須含關鍵詞、不得有數字或英文，不合格就重試；同一關鍵詞盡量不產生重複句。
- 輸出：`tts/tts_data/sentences_train_kw.csv`，其中 `tts_text` 含標點（給 TTS），`manual_transcript` 為純中文（作為 ASR 標籤），`gen_status` 記錄生成狀態（ok / duplicate / template）。
- 若在 transformers < 4.49 的環境執行，需加 `--trust_remote_code`。

### 4.2 語音合成

```bash
python tts/F5_TTS_matched.py \
    --input_csv tts/tts_data/sentences_train_kw.csv \
    --ref_mode single \
    --ref_audio /home/v2129375/dataset/catslu_traindev/data/video/audios/1cfdff9db4788e7e8b0729b8236c6594_59cca4de3327930c51000687.wav \
    --ref_text "我想看白雪公主" \
    --output_audio_dir /home/v2129375/dataset/tts_audio_matched \
    --output_csv tts/tts_data/sentences_audio_matched.csv
```

- 參考說話人：論文使用訓練集中一位咬字清晰的女聲（即上面的音訊）。`--ref_mode random` 則每句從訓練集隨機挑一位說話人。
- 語速：先計算真實訓練集的整體平均語速（總字數 ÷ 總秒數），每句長度 = 字數 ÷ 平均語速。加 `--match_total_hours` 會微調整體語速，使總時數與真實資料完全相同（重新合成時需同時加 `--overwrite`）。
- 輸出：16 kHz 音訊與 `tts/tts_data/sentences_audio_matched.csv`。欄位與真實資料相容，可直接用於微調。中斷後重新執行會自動跳過已合成的句子。

### 4.3 核對統計

```bash
python tts/compare_real_tts_stats.py \
    --real_csv data/catslu/train.csv \
    --syn_csv tts/tts_data/sentences_audio_matched.csv \
    --save tts/tts_data/real_vs_tts_stats.csv
```

確認 `n_utts` = 1511、各領域句數與真實相同、`keyword_multiset_equal_real` = True、`total_hours` 相近。

### 4.4 用合成資料訓練與測試

切回 `phi4` 環境，依第 2、3 節的流程執行，只需修改下列位置。超參數請與真實資料那組相同。

| 步驟 | 檔案：行號 | 修改為 |
|---|---|---|
| 意圖微調 | `intent/finetune_phi4/sample_finetune_speech_intent.py`：445 | `default="tts/tts_data/sentences_audio_matched.csv"` |
| 意圖微調 | 同上：451 | `default=` 與真實資料那組相同的驗證集筆數 |
| 意圖微調 | 同上：455 | `default='intent/model/tts_matched/'` |
| 意圖推論 | `intent/phi4_local.py`：22 | `model_path = "intent/model/tts_matched"` |
| 意圖推論 | 同上：105 | `output_filename = "tts_matched_phi4_intent_result.csv"` |
| 意圖評分 | `intent/intent_score.py`：5 | `input_data_path = "intent/exp/tts_matched_phi4_intent_result.csv"` |
| ASR 微調 | `asr/finetune/finetune_catslu_keywords.py`：38 | `default="tts/tts_data/sentences_audio_matched.csv"` |
| ASR 微調 | 同上：41 | `default='asr/model/TTSMatchedKeywords'` |
| ASR 推論＋評分 | `asr/phi4_keywords.py`：24 | `input_data_path = "intent/exp/tts_matched_phi4_intent_result.csv"` |
| ASR 推論＋評分 | 同上：25 | `model_path = "asr/model/TTSMatchedKeywords"` |

執行順序：

```bash
conda activate phi4
python intent/finetune_phi4/sample_finetune_speech_intent.py   # 意圖微調（合成資料）
python intent/phi4_local.py                                      # 在真實測試集推論
python intent/intent_score.py                                    # 意圖評分
python asr/finetune/finetune_catslu_keywords.py                  # ASR 微調（合成資料）
python asr/phi4_keywords.py                                      # 推論＋評分 → asr/exp/TTSMatchedKeywords.json
```

ASR 推論使用的是「以合成資料訓練的意圖模型」的預測結果，因此整個流程沒有任何真實標註資料參與訓練。

---

## 5. 常見問題

- **`ImportError: cannot import name 'LossKwargs'`**：transformers 版本與模型附帶的程式碼不相容。Phi-4 Multimodal 請在 `phi4` 環境（`transformers==4.48.2`）執行；Phi-4-mini 造句在新版 transformers 下不需加 `--trust_remote_code`。
- **只有一張 GPU**：ASR 微調改 `--gpu_ids 0`，ASR 推論改 `--gpus 0`；意圖微調需修改 `intent/finetune_phi4/device_map.json`，把所有層都設為 0，或讓它使用 `'auto'`。
- **結果被覆蓋**：推論結果檔名與模型資料夾都寫死在程式中，每次實驗前請確認已改成新的名稱。
