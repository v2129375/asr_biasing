"""
使用 Gemma-3N 做中文 ASR 任务，模仿 phi4_keywords.py 的处理方式：

- 支持从 CSV 文件读取音频路径和意图信息
- 支持关键词功能（可选）
- 使用 messages 格式，在 content 中直接包含音频和文本
- 批量处理音频并保存结果到 CSV
- 最后调用 score.evaluate_asr 进行评估

示例调用：
    python asr/gemma2.py \
        --input intent/exp/tts_phi4_intent_result.csv \
        --model_path google/gemma-3n-e2b-it \
        --base_model google/gemma-3n-e2b-it \
        --gpus 0 \
        --batch_size 1
"""

import json
import os
import random
import gc
import re
import argparse
from typing import List, Tuple
from pathlib import Path

import torch
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoProcessor, GenerationConfig, Gemma3nForConditionalGeneration

from score import evaluate_asr


# =======================
# 全局参数（与 phi4_keywords.py 风格一致）
# =======================

# 是否使用关键词
USE_KEYWORDS = True

# 随机选择的关键词数量，设为 0 表示使用全部关键词
NUM_KEYWORDS = 0

# 输入数据路径（默认复用 phi4 的输入）
input_data_path = "intent/exp/tts_phi4_intent_result.csv"

# Gemma-3N 模型路径（可以是微调后的目录，也可以是原始 HF ID）
# 如果是本地路径，使用绝对路径或相对于项目根目录的路径
model_path = "asr/model/gemma_keywords2"

# 基础模型路径（用于加载 processor 等配置）
base_model_path = "google/gemma-3n-e2b-it"

# 设备映射文件路径（多卡时使用，格式与 phi4_keywords 一致）
DEVICE_MAP_PATH = "asr/finetune/device_map.json"

# 默认使用的 GPU（可以是多卡，例如 "0,1"）
DEFAULT_GPUS = "0"

# 默认 batch 大小
DEFAULT_BATCH_SIZE = 1

# ASR 生成长度上限（越大越慢）
MAX_NEW_TOKENS_ASR = 32

# 基础任务指令（中文 ASR）
BASE_INSTRUCTION = "请将这段中文语音准确转写为中文文本，只输出转写结果本身，不要翻译、不需要解释。"

# 带关键词的任务指令模板
KEYWORD_INSTRUCTION_TEMPLATE = (
    "请将这段中文语音准确转写为中文文本，注意这些关键词：{keywords}。只输出转写结果本身，不要翻译、不需要解释。"
)

# 关键词目录
KEYWORDS_DIR = "data/catslu"


def parse_args():
    parser = argparse.ArgumentParser(description="ASR with Gemma-3N (audio input, keywords)")
    parser.add_argument(
        "--use_keywords", type=bool, default=USE_KEYWORDS, help="Whether to use keywords"
    )
    parser.add_argument(
        "--num_keywords",
        type=int,
        default=NUM_KEYWORDS,
        help="Number of keywords to randomly select (0 for all)",
    )
    parser.add_argument(
        "--input", type=str, default=input_data_path, help="Input data path (CSV)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_path,
        help="Gemma-3N model path (fine-tuned or base)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=base_model_path,
        help="Base Gemma-3N model path (for processor, etc.)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=DEVICE_MAP_PATH,
        help="Device map path for multi-GPU (JSON file, same format as phi4_keywords)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=DEFAULT_GPUS,
        help='GPU IDs to use (comma-separated, e.g., "0" or "0,1")',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--max_new_tokens_asr",
        type=int,
        default=MAX_NEW_TOKENS_ASR,
        help="Maximum new tokens for Gemma ASR decoding (smaller is faster)",
    )
    parser.add_argument(
        "--keywords_dir",
        type=str,
        default=KEYWORDS_DIR,
        help="Keywords directory",
    )
    parser.add_argument(
        "--base_instruction",
        type=str,
        default=BASE_INSTRUCTION,
        help="Base instruction for ASR",
    )
    parser.add_argument(
        "--keyword_template",
        type=str,
        default=KEYWORD_INSTRUCTION_TEMPLATE,
        help="Keyword instruction template",
    )
    return parser.parse_args()


# 解析命令行参数并更新全局变量（完全仿照 phi4_keywords.py）
args = parse_args()

USE_KEYWORDS = args.use_keywords
NUM_KEYWORDS = args.num_keywords
input_data_path = args.input
model_path = args.model_path
output_data_path = f"asr/exp/{os.path.basename(model_path)}.csv"
base_model_path = args.base_model
DEVICE_MAP_PATH = args.device_map
DEFAULT_GPUS = args.gpus
DEFAULT_BATCH_SIZE = args.batch_size
KEYWORDS_DIR = args.keywords_dir
BASE_INSTRUCTION = args.base_instruction
KEYWORD_INSTRUCTION_TEMPLATE = args.keyword_template
MAX_NEW_TOKENS_ASR = args.max_new_tokens_asr


class GemmaASR:
    """
    使用 Gemma-3N 做语音识别（ASR），接口风格尽量与 Phi4 类似：

    - `load_gemma()`：加载模型和 processor，支持多卡；
    - `load_keywords()` / `get_domain_keywords()` / `build_instruction_with_keywords()`：
       与 phi4_keywords 中的逻辑保持一致；
    - `process_batch()`：批量处理音频（audio, samplerate, intent）；
    - `process_audio()`：单条音频封装。
    """

    def __init__(self, gpu_ids: List[int] | None = None, batch_size: int = 1):
        self.model: Gemma3nForConditionalGeneration | None = None
        self.processor: AutoProcessor | None = None
        self.generation_config: GenerationConfig | None = None

        # 关键词字典
        self.keywords_dict: dict = {}

        # GPU 配置
        if gpu_ids is None or len(gpu_ids) == 0:
            gpu_ids = [0]
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.main_device = f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"

    def load_gemma(self):
        """加载 Gemma-3N 模型和 AutoProcessor，支持单卡/多卡。"""
        num_gpus = len(self.gpu_ids)
        print(f"使用 GPU: {self.gpu_ids}")

        if num_gpus > 1 and torch.cuda.is_available():
            # 多卡：从 JSON 读取 device_map
            with open(DEVICE_MAP_PATH, "r") as f:
                device_map = json.load(f)
        else:
            # 单卡或无 CUDA：直接指定主设备
            device_map = self.main_device

        # processor 一般从基础模型加载，保证配置完整
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

        # 使用全局变量 model_path
        current_model_path = model_path
        
        print(f"正在加载 Gemma-3N 模型到 {device_map} ...")
        print(f"模型路径: {current_model_path}")
        
        # 检查是否是本地路径
        # 判断标准：路径不以 "http" 开头，且不是 HuggingFace Hub ID 格式（不包含 "/" 或格式为 "org/model"）
        is_hf_hub_id = "/" in current_model_path and not current_model_path.startswith("./") and not current_model_path.startswith("../") and not os.path.isabs(current_model_path) and not os.path.exists(current_model_path)
        
        if not is_hf_hub_id:
            # 本地路径：转换为绝对路径
            if not os.path.isabs(current_model_path):
                # 相对于当前工作目录
                if os.path.exists(current_model_path):
                    current_model_path = os.path.abspath(current_model_path)
                else:
                    # 尝试相对于项目根目录（asr_biasing）
                    current_file_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(current_file_dir)  # 上一级目录
                    potential_path = os.path.join(project_root, current_model_path)
                    if os.path.exists(potential_path):
                        current_model_path = potential_path
                    else:
                        # 如果还是不存在，使用绝对路径（可能路径错误）
                        current_model_path = os.path.abspath(current_model_path)
            print(f"使用本地模型路径: {current_model_path}")
        
        # 检查是否是 LoRA 适配器（只包含 adapter 文件，不包含完整模型）
        adapter_config_path = os.path.join(current_model_path, "adapter_config.json") if not is_hf_hub_id else None
        is_lora_adapter = adapter_config_path and os.path.exists(adapter_config_path)
        
        if is_lora_adapter:
            print("检测到 LoRA 适配器，加载基础模型和 LoRA 权重...")
            from peft import PeftModel
            # 先加载基础模型
            print(f"加载基础模型: {base_model_path}")
            base_model = Gemma3nForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map=device_map,
            )
            # 然后加载 LoRA 权重
            print(f"加载 LoRA 权重: {current_model_path}")
            self.model = PeftModel.from_pretrained(base_model, current_model_path)
            print("LoRA 权重加载完成")
        else:
            # 完整模型或 HuggingFace Hub ID
            print(f"加载完整模型: {current_model_path}")
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                current_model_path,
                device_map=device_map,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=not is_hf_hub_id,  # 本地路径时只使用本地文件
            )

        # 生成配置
        try:
            config_path = current_model_path if not is_hf_hub_id else current_model_path
            self.generation_config = GenerationConfig.from_pretrained(config_path)
        except Exception:
            # 若没有单独的 generation_config，则使用模型自带的配置
            self.generation_config = self.model.generation_config

        # 显式限制解码长度，加快推理
        if self.generation_config is not None:
            self.generation_config.max_new_tokens = MAX_NEW_TOKENS_ASR
            # 确保缓存使用正确的 dtype（与模型一致）
            if hasattr(self.generation_config, 'cache_implementation'):
                # 某些版本可能需要显式设置缓存实现
                pass

        # 打印模型分布情况
        if hasattr(self.model, "hf_device_map"):
            print("模型分布情况:")
            for key, device in self.model.hf_device_map.items():
                print(f"  {key}: {device}")

    def load_keywords(self, keywords_dir: str = KEYWORDS_DIR):
        """加载各领域的关键词文件（video/music/city）"""
        keyword_files = {
            "video": os.path.join(keywords_dir, "keyword_video.txt"),
            "music": os.path.join(keywords_dir, "keyword_music.txt"),
            "city": os.path.join(keywords_dir, "keyword_city.txt"),
        }

        for domain, file_path in keyword_files.items():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    keywords = [line.strip() for line in f.readlines() if line.strip()]
                self.keywords_dict[domain] = keywords
                print(f"Loaded {len(keywords)} keywords for {domain} domain")
            except FileNotFoundError:
                print(f"Warning: Keyword file not found: {file_path}")
                self.keywords_dict[domain] = []

    def get_domain_keywords(self, intent: str) -> List[str]:
        """根据意图获取对应领域的关键词（逻辑与 phi4_keywords.py 一致）"""
        if not isinstance(intent, str):
            intent = str(intent)
        intent_lower = intent.lower()

        if "video" in intent_lower:
            domain = "video"
        elif "music" in intent_lower:
            domain = "music"
        elif "city" in intent_lower:
            domain = "city"
        else:
            domain = "video"

        keywords = self.keywords_dict.get(domain, [])

        if NUM_KEYWORDS > 0 and USE_KEYWORDS and keywords:
            num_to_select = min(len(keywords), NUM_KEYWORDS)
            return random.sample(keywords, num_to_select)
        else:
            return keywords

    def build_instruction_with_keywords(self, intent: str) -> str:
        """根据意图构建包含关键词的指令"""
        domain_keywords = self.get_domain_keywords(intent)

        if domain_keywords and USE_KEYWORDS:
            keywords_str = " ".join(domain_keywords)
            return KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=keywords_str)
        else:
            return BASE_INSTRUCTION

    def release_gemma(self):
        """释放 Gemma 模型和显存"""
        del self.model, self.processor
        gc.collect()

        if torch.cuda.is_available():
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(f"cuda:{gpu_id}"):
                    torch.cuda.empty_cache()

    def process_batch(self, batch_data: List[Tuple]) -> List[str]:
        """
        批量处理音频数据。
        batch_data: List[(audio_array, samplerate, intent)]
        
        使用 Gemma-3N 的 messages 格式，在 content 中直接包含音频和文本。
        """
        results: List[str] = []

        # 为每个样本构建 messages
        batch_messages = []

        for audio, samplerate, intent in batch_data:
            # 根据意图构建指令
            instruction = self.build_instruction_with_keywords(intent)

            # 确保音频是 float32 格式，范围在 [-1, 1]
            if audio.dtype != "float32":
                audio = audio.astype("float32")

            # 使用 messages 格式，在 content 中直接包含音频和文本
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": instruction}
                    ]
                }
            ]

            batch_messages.append(messages)

        # 批量处理
        # 获取模型的 dtype，确保输入与模型 dtype 匹配
        model_dtype = next(self.model.parameters()).dtype
        print(f"模型 dtype: {model_dtype}")
        
        # 对每个样本使用 apply_chat_template 处理
        batch_inputs = []
        for messages in batch_messages:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs.append(inputs)

        # 合并 batch（注意：Gemma-3N 的 apply_chat_template 可能不支持直接 batch）
        # 如果 batch_size > 1，需要逐个处理或手动 padding
        if len(batch_inputs) == 1:
            # 单样本情况
            inputs = batch_inputs[0].to(self.main_device)
            # 确保正确的数据类型：整数类型保持 long，浮点类型匹配模型 dtype
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
            # 音频相关特征匹配模型的 dtype（避免 dtype 不匹配）
            # 确保所有浮点类型输入都匹配模型 dtype
            for key in inputs.keys():
                if inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype=model_dtype)
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                # 完全禁用混合精度，确保 dtype 一致
                # 使用 torch.no_grad() 和显式禁用 autocast
                with torch.cuda.amp.autocast(enabled=False):
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS_ASR,
                        generation_config=self.generation_config,
                        do_sample=False,
                    )
                generate_ids = generate_ids[0, input_len:]

            decoded = self.processor.decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.append(decoded)
        else:
            # 多样本情况：逐个处理（因为 apply_chat_template 可能不支持直接 batch）
            for inputs in batch_inputs:
                inputs = inputs.to(self.main_device)
                # 确保正确的数据类型：整数类型保持 long，浮点类型匹配模型 dtype
                if "input_ids" in inputs:
                    inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
                # 音频相关特征匹配模型的 dtype（避免 dtype 不匹配）
                # 确保所有浮点类型输入都匹配模型 dtype
                for key in inputs.keys():
                    if inputs[key].dtype.is_floating_point:
                        inputs[key] = inputs[key].to(dtype=model_dtype)
                # 打印调试信息
                if idx == 0:  # 只打印第一个样本的调试信息
                    print(f"输入 keys: {list(inputs.keys())}")
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: dtype={value.dtype}, shape={value.shape}")
                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    # 完全禁用混合精度，确保 dtype 一致
                    # 使用 torch.no_grad() 和显式禁用 autocast
                    with torch.cuda.amp.autocast(enabled=False):
                        generate_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS_ASR,
                            generation_config=self.generation_config,
                            do_sample=False,
                        )
                    generate_ids = generate_ids[0, input_len:]

                decoded = self.processor.decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                results.append(decoded)

        # 清理缓存
        del batch_inputs
        if torch.cuda.is_available():
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(f"cuda:{gpu_id}"):
                    torch.cuda.empty_cache()

        return results

    def process_audio(self, audio, samplerate: int, intent: str) -> str:
        """处理单个音频（封装成 batch=1）"""
        results = self.process_batch([(audio, samplerate, intent)])
        return results[0]


if __name__ == "__main__":
    # 输出关键词使用情况
    print(f"使用关键词: {USE_KEYWORDS}")
    if USE_KEYWORDS:
        if NUM_KEYWORDS > 0:
            print(f"随机选择关键词: 每个样本选择 {NUM_KEYWORDS} 个关键词")
        else:
            print("使用所有可用的关键词")

    # 解析 GPU ID 列表
    gpu_ids = [int(gpu_id.strip()) for gpu_id in DEFAULT_GPUS.split(",")]

    gemma_asr = GemmaASR(gpu_ids=gpu_ids, batch_size=DEFAULT_BATCH_SIZE)
    gemma_asr.load_gemma()

    # 加载关键词
    gemma_asr.load_keywords(KEYWORDS_DIR)

    # 读取输入数据
    df = pd.read_csv(input_data_path)

    # 添加 ASR 结果列
    df["asr"] = ""

    batch_data: List[Tuple] = []
    batch_indices: List[int] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频 (Gemma-3N ASR)"):
        # 读取音频
        audio, samplerate = sf.read(row["path"])
        # 如果是多通道，转为单通道
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 重采样到 16kHz（Gemma-3N 推荐）
        if samplerate != 16000:
            audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000

        # 预测意图列与 phi4_keywords 一致，使用 'predict'
        intents = row.get("predict", "")

        batch_data.append((audio, samplerate, intents))
        batch_indices.append(idx)

        # 累积到一个 batch 或到达末尾
        if len(batch_data) >= gemma_asr.batch_size or idx == len(df) - 1:
            responses = gemma_asr.process_batch(batch_data)

            for i, response in enumerate(responses):
                # 删除标点符号（与 phi4_keywords.py 一致）
                response = re.sub(r"[^\w\s]", "", response)

                # 数字转中文（与 phi4_keywords.py 保持一致）
                def num_to_chinese(matched):
                    num = matched.group(0)
                    chinese_nums = {
                        "0": "零",
                        "1": "一",
                        "2": "二",
                        "3": "三",
                        "4": "四",
                        "5": "五",
                        "6": "六",
                        "7": "七",
                        "8": "八",
                        "9": "九",
                        "10": "十",
                        "11": "十一",
                        "12": "十二",
                        "13": "十三",
                        "14": "十四",
                        "15": "十五",
                        "16": "十六",
                        "17": "十七",
                        "18": "十八",
                        "19": "十九",
                        "20": "二十",
                    }
                    return chinese_nums.get(num, num)

                response = re.sub(r"\d+", num_to_chinese, response)

                # 当前样本的文件路径
                file_path = df.loc[batch_indices[i], "path"]
                # 在控制台输出：文件路径 + 最终解码句子
                print(f"{file_path}\t{response}")

                df.at[batch_indices[i], "asr"] = response

            batch_data = []
            batch_indices = []

    # 确保输出目录存在并保存结果
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_csv(output_data_path, index=False)

    # 释放资源
    gemma_asr.release_gemma()

    # 评估 ASR 结果
    print("\n开始评估 Gemma-3N ASR 结果...")
    results_df = pd.read_csv(output_data_path)
    output_json_path = output_data_path.replace(".csv", ".json")
    evaluation_results = evaluate_asr(
        results_df, cal_keyword_wer=True, print_errors=True, output_file=output_json_path
    )
