import os
import re
import gc
import json
import argparse
import math
import random
import copy
from typing import List, Tuple

import torch
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2AudioForConditionalGeneration,
)
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoderLayer
from peft import PeftModel

from score import evaluate_asr


# =======================
# 全局参数（与 phi4_keywords.py 保持同样的结构）
# =======================

# 是否使用关键词
USE_KEYWORDS = True

# 随机选择的关键词数量，设为 0 表示使用全部关键词
NUM_KEYWORDS = 0

# 输入数据路径（默认为 phi4 的输入文件，可按需修改）
input_data_path = "intent/exp/tts_phi4_intent_result.csv"

# 微调后 Qwen2-Audio 模型路径（与 finetune_qwen.py 中 OUTPUT_DIR 保持一致）
model_path = "asr/model/qwen_finetune2"

# 基础模型路径
base_model_path = "Qwen/Qwen2-Audio-7B-Instruct"

# 设备映射文件路径（保留参数以与 phi4_keywords 接口一致，在单卡场景下不会实际使用）
DEVICE_MAP_PATH = "asr/finetune/device_map.json"

# 默认使用的 GPU（这里只有一张 RTX 5090，所以默认只用 0 号卡）
DEFAULT_GPUS = "0"

# 默认 batch 大小
DEFAULT_BATCH_SIZE = 1

# 基础任务指令（与微调代码保持一致）
BASE_INSTRUCTION = "Transcribe the audio clip into text."

# 带关键词的任务指令模板
KEYWORD_INSTRUCTION_TEMPLATE = (
    "Transcribe the audio clip into text. Pay attention to these keywords: {keywords}"
)

# 关键词目录（与训练脚本一致）
KEYWORDS_DIR = "data/catslu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR with Qwen2-Audio on a single RTX 5090 (fp16 + flash attention)"
    )
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
        "--input", type=str, default=input_data_path, help="Input data path"
    )
    parser.add_argument(
        "--model_path", type=str, default=model_path, help="Fine-tuned model path"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=base_model_path,
        help="Base model path (fallback for processor)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=DEVICE_MAP_PATH,
        help="Device map path (kept for API compatibility, not used in single-GPU mode)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=DEFAULT_GPUS,
        help='GPU IDs to use (comma-separated, e.g., "0")',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
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


# 解析命令行参数并更新全局变量
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


class QwenASR:
    """
    参考 finetune_qwen.py 和 phi4_keywords.py，
    使用 Qwen2-Audio-7B-Instruct 在单张 RTX 5090 上做 fp16 + flash attention 推理。
    """

    def __init__(self, gpu_ids: List[int] = None, batch_size: int = 1):
        self.model: Qwen2AudioForConditionalGeneration | None = None
        self.processor: AutoProcessor | None = None
        self.generation_config: GenerationConfig | None = None
        self.keywords_dict: dict = {}

        if gpu_ids is None or len(gpu_ids) == 0:
            gpu_ids = [0]

        # 虽然参数支持多个 GPU，但根据你的环境只有一张 5090，这里只使用第一张
        if len(gpu_ids) > 1:
            print(
                f"检测到传入多个 GPU {gpu_ids}，但当前脚本仅使用单卡模式，将只使用 {gpu_ids[0]}。"
            )

        self.gpu_ids = [gpu_ids[0]]
        self.batch_size = batch_size
        self.main_device = f"cuda:{self.gpu_ids[0]}"

    def load_qwen(self):
        """加载 Qwen2-Audio 模型和处理器，使用 fp16 + flash attention 在单卡上推理"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请确认已安装正确的 CUDA 和驱动")

        gpu_id = self.gpu_ids[0]
        torch.cuda.set_device(gpu_id)
        print(f"使用 GPU: cuda:{gpu_id}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"GPU 设备名称: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU 计算能力: {torch.cuda.get_device_capability(gpu_id)}")

        # 显式使用 fp16
        torch_dtype = torch.float16

        # 建议开启可扩展分段，减少显存碎片
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        # 先尝试从微调目录加载 processor（finetune_qwen.py 已将 processor 保存到 OUTPUT_DIR）
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            print(f"已从微调模型目录加载处理器: {model_path}")
        except Exception as e:
            print(
                f"从微调模型目录加载处理器失败，将回退到基础模型 {base_model_path}，错误: {e}"
            )
            self.processor = AutoProcessor.from_pretrained(
                base_model_path, trust_remote_code=True
            )

        # 先加载基座模型，再加载微调好的 LoRA 权重（finetune 只保存了 adapter）
        print(f"正在加载基座模型 {base_model_path} (fp16 + eager attention)...")
        try:
            base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                attn_implementation="eager",
                trust_remote_code=True,
                device_map={"" : self.main_device},
            )
        except TypeError:
            base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                attn_implementation="eager",
                trust_remote_code=True,
                device_map=self.main_device,
            )

        # 加载微调好的 LoRA adapter 权重
        if os.path.isfile(os.path.join(model_path, "adapter_config.json")) or os.path.isfile(
            os.path.join(model_path, "adapter_model.safetensors")
        ) or os.path.isfile(os.path.join(model_path, "adapter_model.bin")):
            print(f"正在加载 LoRA 权重: {model_path}")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # 目录下是完整模型（已合并 LoRA 或非 PEFT 保存），从 model_path 加载
            del base_model
            print(f"未检测到 LoRA adapter，从 {model_path} 加载完整模型")
            try:
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                    trust_remote_code=True,
                    device_map={"" : self.main_device},
                )
            except TypeError:
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                    trust_remote_code=True,
                    device_map=self.main_device,
                )

        # 关闭 cache，略省显存
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        # 生成配置
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception:
            self.generation_config = self.model.generation_config

        print("Qwen2-Audio 模型加载完成。")

    def load_keywords(self, keywords_dir: str = KEYWORDS_DIR):
        """加载各领域的关键词文件（与训练时的 video/music/city 一致）"""
        keyword_files = {
            "video": os.path.join(keywords_dir, "keyword_video.txt"),
            "music": os.path.join(keywords_dir, "keyword_music.txt"),
            "city": os.path.join(keywords_dir, "keyword_city.txt"),
        }

        for domain, file_path in keyword_files.items():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    keywords = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                self.keywords_dict[domain] = keywords
                print(f"Loaded {len(keywords)} keywords for {domain} domain")
            except FileNotFoundError:
                print(f"Warning: Keyword file not found: {file_path}")
                self.keywords_dict[domain] = []

    def get_domain_keywords(self, intent: str) -> List[str]:
        """
        根据意图获取对应领域的关键词。
        这里保持与 phi4_keywords.py 一致：根据 intent 文本中包含的 'video'/'music'/'city' 归到不同 domain。
        """
        intent_lower = intent.lower() if isinstance(intent, str) else ""
        if "video" in intent_lower:
            domain = "video"
        elif "music" in intent_lower:
            domain = "music"
        elif "city" in intent_lower:
            domain = "city"
        else:
            domain = "video"  # 默认使用 video 领域

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
            keywords_str = ", ".join(domain_keywords)
            return KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=keywords_str)
        else:
            return BASE_INSTRUCTION

    def release_qwen(self):
        """释放模型与 CUDA 显存"""
        del self.model, self.processor
        gc.collect()

        if torch.cuda.is_available():
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(f"cuda:{gpu_id}"):
                    torch.cuda.empty_cache()

    def process_batch(
        self, batch_data: List[Tuple[torch.Tensor, int, str, str]]
    ) -> List[str]:
        """
        批量处理音频数据。
        为避免 flash-attn 在变长 batch 上的 cu_seqlens 形状问题，这里对每条样本
        单独调用 processor + generate（逻辑批处理，但模型前向是 sample-by-sample）。

        batch_data: List[(audio_array, samplerate, intent, path)]
        """
        results: List[str] = []

        for audio, samplerate, intent, path in batch_data:
            # 构建带关键词的指令
            instruction = self.build_instruction_with_keywords(intent)

            # 按 finetune_qwen.py 的对话格式构建 chat 模板
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": path},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            # # 每次输入识别时输出 Prompt 和对应音频路径
            # print(f"[输入] 音频路径: {path}")
            # print(f"[输入] Prompt: {instruction}")

            sampling_rate = getattr(
                self.processor.feature_extractor, "sampling_rate", 16000
            )

            # 对单条样本做一次前向，仍然使用 fp16 + flash attention
            with autocast(enabled=True):
                processed = self.processor(
                    text=prompt,
                    audio=[audio],
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=sampling_rate,
                ).to(self.main_device)

                # 按官方 Qwen2-Audio 推理示例，直接将 processor 的输出全部传给 generate，
                # 让模型自己处理 audio/text 融合，并使用 flash_attention_2。
                generate_ids = self.model.generate(
                    **processed,
                    max_new_tokens=128,
                    generation_config=self.generation_config,
                )

                gen_only_ids = generate_ids[:, processed["input_ids"].shape[1] :]

                response = self.processor.batch_decode(
                    gen_only_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # 每次模型输出后打印原始 ASR 结果
                print(f"[模型原始输出] {response}")
                results.append(response)

            # 每条样本后做一次轻量清理，避免显存碎片
            del processed, generate_ids, gen_only_ids
            if torch.cuda.is_available():
                for gpu_id in self.gpu_ids:
                    with torch.cuda.device(f"cuda:{gpu_id}"):
                        torch.cuda.empty_cache()

        return results

    def process_audio(self, audio, samplerate: int, intent: str, path: str) -> str:
        """处理单个音频（封装成 batch=1 调用）"""
        results = self.process_batch([(audio, samplerate, intent, path)])
        return results[0]


if __name__ == "__main__":
    # 输出关键词使用情况
    print(f"使用关键词: {USE_KEYWORDS}")
    if USE_KEYWORDS:
        if NUM_KEYWORDS > 0:
            print(f"随机选择关键词: 每个样本选择 {NUM_KEYWORDS} 个关键词")
        else:
            print("使用所有可用的关键词")

    # 解析 GPU ID 列表（但实际只会使用第一张卡）
    gpu_ids = [int(gpu_id.strip()) for gpu_id in DEFAULT_GPUS.split(",")]

    qwen_asr = QwenASR(gpu_ids=gpu_ids, batch_size=DEFAULT_BATCH_SIZE)
    qwen_asr.load_qwen()

    # 加载关键词
    qwen_asr.load_keywords(KEYWORDS_DIR)

    # 读取输入数据文件
    df = pd.read_csv(input_data_path)

    # 添加 ASR 结果列
    df["asr"] = ""

    batch_data: List[Tuple[torch.Tensor, int, str, str]] = []
    batch_indices: List[int] = []

    # 进度条
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频(Qwen2-Audio)"):
        # 读取音频文件
        audio, samplerate = sf.read(row["path"])
        # 若为多通道，转为单通道
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 读取预测的 intents（与 phi4_keywords 保持一致，使用 'predict' 列）
        intents = row.get("predict", "")

        batch_data.append((audio, samplerate, intents, row["path"]))
        batch_indices.append(idx)

        # 批处理
        if len(batch_data) >= qwen_asr.batch_size or idx == len(df) - 1:
            responses = qwen_asr.process_batch(batch_data)

            for i, response in enumerate(responses):
                # 删除英文
                response = re.sub(r"[A-Za-z]+", "", response)
                # 删除标点（保留中文、数字和空白符）
                response = re.sub(r"[^\d\s\u4e00-\u9fff]", "", response)
                # 删除所有空白符（空格、制表符等）
                response = re.sub(r"\s+", "", response)

                # 将数字转换为中文数字（与 phi4_keywords 保持一致）
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

                # 打印本条最终 ASR 结果（写入 CSV 的内容）
                print(f"[{batch_indices[i]}] ASR: {response}")
                df.at[batch_indices[i], "asr"] = response

            batch_data = []
            batch_indices = []

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_csv(output_data_path, index=False)

    # 释放资源
    qwen_asr.release_qwen()

    # 评估 ASR 结果
    print("\n开始评估 Qwen2-Audio ASR 结果...")
    results_df = pd.read_csv(output_data_path)
    output_json_path = output_data_path.replace(".csv", ".json")
    evaluation_results = evaluate_asr(
        results_df, cal_keyword_wer=True, print_errors=True, output_file=output_json_path
    )

    # 控制台简单输出一下指标
    if isinstance(evaluation_results, dict):
        print("\n评估指标摘要：")
        for k, v in evaluation_results.items():
            print(f"{k}: {v}")

