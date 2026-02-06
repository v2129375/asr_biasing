"""
finetune Qwen2-Audio-7B-Instruct on an ASR (Automatic Speech Recognition) task with domain-specific keywords
在ASR（自动语音识别）任务上微调 Qwen2-Audio-7B-Instruct 模型，支持领域特定关键词

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
accelerate==1.3.0
"""

import os
import pandas as pd
from pathlib import Path
import random

import torch
import numpy as np
import soundfile as sf
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 全局参数设置
GPU_IDS = [0]  # 单显卡微调，只使用GPU 0
MODEL_NAME_OR_PATH = 'Qwen/Qwen2-Audio-7B-Instruct'
CATSLU_DATA_PATH = "data/catslu/train.csv"
KEYWORDS_DIR = "data/catslu"
USE_FLASH_ATTENTION = True
USE_QLORA_8BIT = False
OUTPUT_DIR = 'asr/model/qwen_finetune2'
BATCH_SIZE = 1
BATCH_SIZE_PER_GPU = 1
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 4.0e-5
WD = 0.01
TQDM_ENABLED = True

USE_KEYWORDS = False
# 关键词随机选择参数
NUM_KEYWORDS = 0  # 随机选择的关键词数量，设为0表示使用全部关键词
# 基础任务指令
BASE_INSTRUCTION = "Transcribe the audio clip into text."
# 带关键词的任务指令模板
KEYWORD_INSTRUCTION_TEMPLATE = "Transcribe the audio clip into text. Pay attention to these keywords: {keywords}"
# 答案后缀标记，用于标识生成结束
ANSWER_SUFFIX = "<|endoftext|>"
# 标签忽略索引值，用于损失计算中忽略某些位置
_IGNORE_INDEX = -100


class CatsluKeywordsDataset(Dataset):
    """支持关键词的CATSLU数据集类，用于ASR任务"""
    def __init__(self, processor, data_path, keywords_dir="data/catslu", split="train", rank=0, world_size=1):
        """
        初始化CATSLU关键词数据集
        
        Args:
            processor: 模型处理器
            data_path: CSV数据文件路径
            keywords_dir: 关键词文件目录路径
            split: 数据集划分，'train'或'eval'
            rank: 分布式训练的进程排名
            world_size: 分布式训练的总进程数
        """
        # 读取CSV文件
        self.data = pd.read_csv(data_path)
        self.training = "train" in split
        self.processor = processor
        
        # 验证数据集中是否包含source列
        if 'source' not in self.data.columns:
            raise ValueError(f"数据集 {data_path} 中缺少 'source' 列，请确保CSV文件包含source字段")
        
        # 检查source列的值
        unique_sources = self.data['source'].unique()
        print(f"数据集中发现的领域: {unique_sources}")
        
        # 验证source值是否在支持的范围内
        supported_sources = ['video', 'music', 'city']
        unsupported_sources = [s for s in unique_sources if s not in supported_sources]
        if unsupported_sources:
            print(f"警告: 发现不支持的领域: {unsupported_sources}")
            print(f"支持的领域: {supported_sources}")
        
        # 加载各领域的关键词
        self.keywords_dict = self._load_keywords(keywords_dir)
        
        # 如果在分布式环境中，分片数据集
        if world_size > 1:
            total_len = len(self.data)
            per_worker = total_len // world_size
            start_idx = rank * per_worker
            end_idx = start_idx + per_worker if rank < world_size - 1 else total_len
            self.data = self.data.iloc[start_idx:end_idx]

    def _load_keywords(self, keywords_dir):
        """加载各领域的关键词"""
        keywords_dict = {}
        
        # 定义关键词文件路径
        keyword_files = {
            'video': os.path.join(keywords_dir, 'keyword_video.txt'),
            'music': os.path.join(keywords_dir, 'keyword_music.txt'), 
            'city': os.path.join(keywords_dir, 'keyword_city.txt')
        }
        
        # 读取每个领域的关键词
        for domain, file_path in keyword_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    keywords = [line.strip() for line in f.readlines() if line.strip()]
                keywords_dict[domain] = keywords
                print(f"Loaded {len(keywords)} keywords for {domain} domain")
            except FileNotFoundError:
                print(f"Warning: Keyword file not found: {file_path}")
                keywords_dict[domain] = []
        
        return keywords_dict

    def _get_domain_keywords(self, source):
        """根据数据源获取对应领域的关键词"""
        if source in self.keywords_dict:
            keywords = self.keywords_dict[source]
            
            # 如果设置了随机选择关键词数量且有足够的关键词
            if NUM_KEYWORDS > 0 and USE_KEYWORDS and keywords:
                # 确保选择的关键词数量不超过可用关键词总数
                num_to_select = min(len(keywords), NUM_KEYWORDS)
                # 随机选择指定数量的关键词
                return random.sample(keywords, num_to_select)
            else:
                return keywords
        else:
            # 如果没有找到对应的关键词，打印警告并返回空列表
            print(f"Warning: No keywords found for source '{source}', using empty keyword list")
            return []

    def _build_instruction_with_keywords(self, source):
        """根据领域构建包含关键词的指令"""
        domain_keywords = self._get_domain_keywords(source)
        
        if domain_keywords and USE_KEYWORDS:
            # 使用该领域的关键词（可能是随机选择的）
            keywords_str = ', '.join(domain_keywords)
            return KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=keywords_str)
        else:
            return BASE_INSTRUCTION

    def __len__(self):
        return len(self.data)
    
    def load_audio(self, audio_path):
        """加载音频文件"""
        try:
            audio_array, sampling_rate = sf.read(audio_path)
            # 确保音频是单声道
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array, sampling_rate
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 返回一个非常短的静音音频
            return np.zeros(16000), 16000

    def __getitem__(self, idx):
        """获取单个样本并处理为模型可接受的格式"""
        data = self.data.iloc[idx]
        
        # 加载音频文件
        audio_array, sampling_rate = self.load_audio(data['path'])
        
        # 从数据中获取source字段，确保不为空
        source = data['source']
        if pd.isna(source) or source == '':
            print(f"Warning: Empty source field in row {idx}, defaulting to 'video'")
            source = 'video'
        
        # 根据数据源构建指令
        instruction = self._build_instruction_with_keywords(source)
        
        # Qwen2-Audio 使用不同的对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": data['path']},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        
        # 应用聊天模板
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        
        
        # 获取采样率
        sampling_rate = getattr(
            self.processor.feature_extractor,
            "sampling_rate",
            16000,
        )
        
        # 处理文本和音频输入
        inputs = self.processor(
            text=prompt,
            audios=[audio_array],
            return_tensors='pt',
            padding=True,
            sampling_rate=sampling_rate,
        )
        
        # 获取转录标签
        transcript_label = data['manual_transcript']

        # 在训练阶段打印当前样本的 Prompt、音频路径和 ground truth，方便检查训练数据是否正确
        if self.training:
            try:
                print(f"[Train sample {idx}] 音频路径: {data['path']}")
                print(f"[Train sample {idx}] Prompt:\n{prompt}")
                print(f"[Train sample {idx}] Ground truth: {transcript_label}")
                print("-" * 80)
            except Exception as e:
                # 打印失败时不要影响训练流程
                print(f"[Train sample {idx}] 打印 Prompt 时出错: {e}")
        
        
        # 构建答案，添加结束标记
        answer = f"{transcript_label}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids
        
        # 训练时，将输入和答案连接起来，标签只关注答案部分
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)   # [1, L_total]
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids

        # 注意：原始 inputs.attention_mask 只对应 prompt 部分；
        # 拼接了答案 token 之后，attention_mask 必须和 input_ids 同长，否则在 loss 里会报 shape 不匹配。
        # 这里直接为所有非 padding token 设置 1（当前没有在序列中额外 pad，所以全 1 即可），
        # 后续在 collate_fn 里再按 batch 维度统一 pad。
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        return {
            'input_ids': input_ids.squeeze(0),
            'labels': labels.squeeze(0),
            'audio_values': inputs.audio_values.squeeze(0) if hasattr(inputs, 'audio_values') else None,
            'attention_mask': attention_mask.squeeze(0),
        }


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    将序列列表填充到相同长度
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def pad_audio_sequences(sequences, padding_value=0.0):
    """
    将音频序列列表填充到相同长度
    """
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len), padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        output.data[i, :length] = seq
    return output


def catslu_collate_fn(batch):
    """
    CATSLU数据集的批处理函数，将多个样本组合为一个批次
    适配 Qwen2-Audio 的输入格式
    """
    input_ids_list = []
    labels_list = []
    audio_values_list = []
    attention_mask_list = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'])
        labels_list.append(inputs['labels'])
        if inputs['audio_values'] is not None:
            audio_values_list.append(inputs['audio_values'])
        if inputs['attention_mask'] is not None:
            attention_mask_list.append(inputs['attention_mask'])

    try:
        # 填充序列到相同长度
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=_IGNORE_INDEX)
        
        # 处理音频值
        if audio_values_list:
            audio_values = pad_audio_sequences(audio_values_list, padding_value=0.0)
        else:
            audio_values = None
            
        # 处理注意力掩码
        if attention_mask_list:
            attention_mask = pad_sequence(attention_mask_list, padding_side='left', padding_value=0)
        else:
            attention_mask = (input_ids != 0).long()
    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
        
    # 构建批次特征
    batch_feature = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }
    
    if audio_values is not None:
        batch_feature['audio_values'] = audio_values
    
    return BatchFeature(batch_feature)


def create_model(model_name_or_path, use_flash_attention=False):
    """
    创建 Qwen2-Audio 模型，可选择使用flash attention加速
    单显卡微调，先加载到CPU再移动到GPU以避免CUDA兼容性问题
    """
    gpu_id = GPU_IDS[0]
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请检查CUDA安装")
    
    print(f"使用GPU: cuda:{gpu_id}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"GPU设备名称: {torch.cuda.get_device_name(gpu_id)}")
    print(f"GPU计算能力: {torch.cuda.get_device_capability(gpu_id)}")

    # 按你的要求：使用 fp16 训练（FlashAttention 也支持 fp16），不再自动优先 bf16
    major, minor = torch.cuda.get_device_capability(gpu_id)
    supports_bf16 = major >= 8  # 保留信息备用，但当前策略固定使用 fp16
    torch_dtype = torch.float16

    # 可选：开启可扩展分段，减少显存碎片导致的 OOM
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # 注意力实现选择：
    # - 按你的要求：当 use_flash_attention=True 时，强制使用 flash_attention_2；
    #   不允许降级到 eager / sdpa。若 flash_attention_2 不可用则直接报错。
    #
    # 额外保险：关闭 PyTorch SDPA 的 flash/mem-efficient 路径，避免框架内部回退时走到 SDPA。
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            # 旧版本 torch 可能没有这些开关，忽略即可
            pass

    attn_candidates = ["eager"]
    if use_flash_attention:
        attn_candidates = ["flash_attention_2"]

    last_err = None
    model = None
    for attn_impl in attn_candidates:
        try:
            if USE_QLORA_8BIT:
                try:
                    from transformers import BitsAndBytesConfig
                except Exception as e:
                    raise RuntimeError(
                        "你已开启 USE_QLORA_8BIT=True，但当前环境缺少 bitsandbytes 支持。\n"
                        "请先安装：pip install -U bitsandbytes"
                    ) from e

                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                print(f"正在加载8bit( int8 ) LoRA模型到GPU... (attn_implementation={attn_impl}, dtype={torch_dtype})")
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                    trust_remote_code=True,
                    device_map={"": gpu_id},  # 量化加载建议直接上GPU
                )
            else:
                print(f"正在加载模型到CPU... (attn_implementation={attn_impl}, dtype={torch_dtype})")
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                    trust_remote_code=True,
                    device_map=None,  # 不使用device_map，手动移动
                )
            print(f"模型加载成功 (attn_implementation={attn_impl})")
            break
        except Exception as e:
            last_err = e
            print(f"加载失败 (attn_implementation={attn_impl})。错误: {e}")
            model = None

    if model is None:
        if use_flash_attention:
            raise RuntimeError(
                "已开启 USE_FLASH_ATTENTION=True，因此必须使用 flash_attention_2，但当前环境加载失败。\n"
                "请确认：\n"
                "1) transformers 版本支持 attn_implementation='flash_attention_2'\n"
                "2) 已安装 flash-attn（并与当前 CUDA / PyTorch / GPU 架构匹配）\n"
                f"最后一次错误: {last_err}"
            )
        raise RuntimeError(f"模型加载失败，最后一次错误: {last_err}")
    
    if not USE_QLORA_8BIT:
        # 将模型移动到指定GPU
        print(f"正在将模型移动到 cuda:{gpu_id}...")
        device = torch.device(f"cuda:{gpu_id}")
        model = model.to(device)

    # 训练建议关闭cache，省显存
    if hasattr(model, "config"):
        model.config.use_cache = False

    # LoRA：按你的要求开启 LoRA 训练。
    # 这里不再使用 8bit 量化，而是直接对 fp16 模型注入 LoRA 适配器。
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    
    return model


def main():
    """主函数，包含模型训练的完整流程"""
    
    # 设置CUDA设备
    gpu_id = GPU_IDS[0]
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f"已设置默认CUDA设备为: cuda:{gpu_id}")
    else:
        raise RuntimeError("CUDA不可用，无法进行训练")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
    )
    
    # 创建模型
    model = create_model(
        MODEL_NAME_OR_PATH,
        use_flash_attention=USE_FLASH_ATTENTION,
    )

    # 创建训练数据集
    train_dataset = CatsluKeywordsDataset(
        processor,
        data_path=CATSLU_DATA_PATH,
        keywords_dir=KEYWORDS_DIR,
        split="train",
        world_size=1
    )
    
    # 输出数据集统计信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Keywords directory: {KEYWORDS_DIR}")
    print(f"Use keywords: {USE_KEYWORDS}")
    if USE_KEYWORDS:
        if NUM_KEYWORDS > 0:
            print(f"Random keywords selection: enabled (selecting {NUM_KEYWORDS} keywords per sample)")
        else:
            print(f"Using all available keywords for each domain")

    # 单显卡训练
    num_gpus = 1
    print(f'training on {num_gpus} GPU')
    assert (
        BATCH_SIZE % BATCH_SIZE_PER_GPU == 0
    ), 'Batch size must be divisible by BATCH_SIZE_PER_GPU'
    gradient_accumulation_steps = BATCH_SIZE // BATCH_SIZE_PER_GPU

    # 混合精度：按你的要求固定使用 fp16 进行训练
    major, minor = torch.cuda.get_device_capability(GPU_IDS[0])
    supports_bf16 = major >= 8  # 仅作信息展示，不再优先 bf16
    bf16 = False
    fp16 = True

    # 设置训练参数
    training_args = TrainingArguments(
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=LEARNING_RATE,
        weight_decay=WD,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not TQDM_ENABLED,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,
    )

    # 创建输出目录
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 创建Trainer实例并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=catslu_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()

    processor.save_pretrained(training_args.output_dir)

    print('Training completed successfully!')


if __name__ == '__main__':
    main()
