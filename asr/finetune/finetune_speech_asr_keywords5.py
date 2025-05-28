"""
finetune Phi-4-multimodal-instruct on an ASR (Automatic Speech Recognition) task with domain-specific keywords
在ASR（自动语音识别）任务上微调 Phi-4-multimodal-instruct 模型，支持领域特定关键词

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
accelerate==1.3.0
"""

import json
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
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)

# 全局参数设置
GPU_IDS = [0,1]
MODEL_NAME_OR_PATH = 'microsoft/Phi-4-multimodal-instruct'
CATSLU_DATA_PATH = "data/catslu/train.csv"
KEYWORDS_DIR = "data/catslu"
USE_FLASH_ATTENTION = True
OUTPUT_DIR = 'asr/model/new'
BATCH_SIZE = 2
BATCH_SIZE_PER_GPU = 1
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 4.0e-5
WD = 0.01
TQDM_ENABLED = True
DEVICE_MAP_PATH = 'asr/finetune/device_map.json'

USE_KEYWORDS = True
INTENT_IN_PROMPT = True
# 关键词随机选择参数
NUM_KEYWORDS = 0  # 随机选择的关键词数量，设为0表示使用全部关键词
# 基础任务指令
BASE_INSTRUCTION = "Transcribe the audio clip into text."
# 带关键词的任务指令模板
KEYWORD_INSTRUCTION_TEMPLATE = "<{intent}> {keywords} </{intent}> Transcribe the audio clip into text."
# 答案后缀标记，用于标识生成结束
ANSWER_SUFFIX = "<|end|><|endoftext|>"
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
            if INTENT_IN_PROMPT:
                return KEYWORD_INSTRUCTION_TEMPLATE.format(intent=source, keywords=keywords_str)
            else:
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
        
        # 构建用户消息
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + instruction,
        }
        
        # 应用聊天模板，构建提示
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        
        # 处理文本和音频输入
        inputs = self.processor(
            text=prompt, 
            audios=[(audio_array, sampling_rate)], 
            return_tensors='pt'
        )
        
        # 获取转录标签
        transcript_label = data['manual_transcript']
        
        # 构建答案，添加结束标记
        answer = f"{transcript_label}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        
        # 训练时，将输入和答案连接起来，标签只关注答案部分
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids
        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
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


def cat_with_pad(tensors, dim, padding_value=0):
    """
    在指定维度上连接张量，同时为其他维度填充到最大值
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # 创建一个切片列表，除了连接维度外，所有维度都是完整切片
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # 更新连接维度的切片
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def catslu_collate_fn(batch):
    """
    CATSLU数据集的批处理函数，将多个样本组合为一个批次
    """
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )

    try:
        # 填充序列到相同长度
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
        
    attention_mask = (input_ids != 0).long()  # 创建注意力掩码
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)  # 连接音频嵌入
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)  # 连接音频嵌入大小

    # 返回包含所有必要字段的批次特征
    batch_feature = BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,  # speech mode 语音模式
        }
    )
        
    return batch_feature


def create_model(model_name_or_path, use_flash_attention=False):
    """
    创建因果语言模型，可选择使用flash attention加速
    """
    gpu_ids = GPU_IDS
    num_gpus = len(gpu_ids)
    print(f"使用GPU: {gpu_ids}")
    main_device = f"cuda:{gpu_ids[0]}"
    # 构建device_map
    if num_gpus > 1:
        # 读取设备映射配置文件
        with open(DEVICE_MAP_PATH, 'r') as f:
            device_map = json.load(f)
    else:
        # 单GPU情况
        device_map = main_device
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
        device_map=device_map
    )
    return model


def main():
    """主函数，包含模型训练的完整流程"""

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
    )
    # 创建模型
    model = create_model(
        MODEL_NAME_OR_PATH,
        use_flash_attention=USE_FLASH_ATTENTION,
    )

    # 设置模型使用语音适配器（在prepare之前）
    model.set_lora_adapter('speech')

    
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

    # 计算GPU数量并进行批处理大小断言
    gpu_ids = GPU_IDS
    num_gpus = len(gpu_ids)
    print(f'training on {num_gpus} GPUs')
    assert (
        BATCH_SIZE % (num_gpus * BATCH_SIZE_PER_GPU) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = BATCH_SIZE // (num_gpus * BATCH_SIZE_PER_GPU)

    # 根据是否使用flash attention选择精度
    if USE_FLASH_ATTENTION:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

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
        ddp_find_unused_parameters=True,  # 用于未使用的SigLIP层
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