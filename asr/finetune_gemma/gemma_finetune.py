"""
finetune Gemma-3N on an ASR (Automatic Speech Recognition) task with domain-specific keywords
在ASR（自动语音识别）任务上微调 Gemma-3N 模型，支持领域特定关键词

参考结构：asr/finetune/finetune_speech_asr_keywords4.py
参考实现：asr/gemma2.py（使用 messages 格式）
"""

import json
import os
import pandas as pd
from pathlib import Path
import random

import torch
import numpy as np
import soundfile as sf
import librosa
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Gemma3nForConditionalGeneration,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# 全局参数设置
GPU_IDS = [0]  # 单 GPU RTX 5090
MODEL_NAME_OR_PATH = 'google/gemma-3n-e2b-it'
CATSLU_DATA_PATH = "data/catslu/train.csv"
KEYWORDS_DIR = "data/catslu"
USE_FLASH_ATTENTION = False  # Gemma-3N 可能不支持 flash_attention_2
OUTPUT_DIR = 'asr/model/gemma_keywords2'
BATCH_SIZE = 1  # 单 GPU 时，batch size 可以设置小一点
BATCH_SIZE_PER_GPU = 1
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 4.0e-5
WD = 0.01
TQDM_ENABLED = True
DEVICE_MAP_PATH = 'asr/finetune/device_map.json'  # 单 GPU 时不会使用，但保留以兼容代码

# LoRA 配置
USE_LORA = True
LORA_R = 8  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 目标模块

USE_KEYWORDS = True
# 关键词随机选择参数
NUM_KEYWORDS = 0  # 随机选择的关键词数量，设为0表示使用全部关键词
# 基础任务指令（中文 ASR）
BASE_INSTRUCTION = "请将这段中文语音准确转写为中文文本，只输出转写结果本身，不要翻译、不需要解释。"
# 带关键词的任务指令模板
KEYWORD_INSTRUCTION_TEMPLATE = (
    "请将这段中文语音准确转写为中文文本，注意这些关键词：{keywords}。只输出转写结果本身，不要翻译、不需要解释。"
)
# 答案后缀标记，用于标识生成结束
ANSWER_SUFFIX = "<|end|><|endoftext|>"
# 标签忽略索引值，用于损失计算中忽略某些位置
_IGNORE_INDEX = -100


class CatsluKeywordsDataset(Dataset):
    """支持关键词的CATSLU数据集类，用于ASR任务（Gemma-3N版本）"""
    
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
            keywords_str = ' '.join(domain_keywords)
            return KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=keywords_str)
        else:
            return BASE_INSTRUCTION

    def __len__(self):
        return len(self.data)
    
    def load_audio(self, audio_path):
        """加载音频文件并重采样到16kHz"""
        try:
            audio_array, sampling_rate = sf.read(audio_path)
            # 确保音频是单声道
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            # 重采样到16kHz（Gemma-3N推荐）
            if sampling_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            # 确保是float32格式
            audio_array = audio_array.astype(np.float32)
            return audio_array, sampling_rate
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 返回一个非常短的静音音频
            return np.zeros(16000, dtype=np.float32), 16000

    def __getitem__(self, idx):
        """获取单个样本并处理为模型可接受的格式（使用Gemma-3N的messages格式）"""
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
        
        # 使用 Gemma-3N 的 messages 格式，在 content 中直接包含音频和文本
        # 这是 Gemma-3N 的正确用法（参考 gemma2.py）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        
        # 使用 apply_chat_template 处理 messages（训练时不需要 add_generation_prompt）
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,  # 训练时不需要生成提示
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # 获取转录标签
        transcript_label = data['manual_transcript']
        
        # 构建答案，添加结束标记
        answer = f"{transcript_label}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        
        # 训练时，将输入和答案连接起来，标签只关注答案部分
        input_ids = torch.cat([inputs['input_ids'], answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids
        
        # 返回所有必要的字段
        # Gemma-3N 的 processor 返回的字段名可能与 phi4 不同
        result = {
            'input_ids': input_ids,
            'labels': labels,
        }
        
        # 添加所有输入字段（apply_chat_template 返回的字段）
        # 打印第一个样本的字段名以便调试（仅第一次）
        if idx == 0:
            print(f"apply_chat_template returned keys: {list(inputs.keys())}")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 复制所有输入字段（apply_chat_template 会返回所有必要的字段）
        for key, value in inputs.items():
            if key != 'input_ids':  # input_ids 已经处理过了
                if isinstance(value, torch.Tensor):
                    # 移除可能的 batch 维度（如果存在）
                    if value.dim() > 0 and value.shape[0] == 1:
                        # 检查是否可以安全地移除 batch 维度
                        # 对于某些字段（如 input_features），可能需要保留 batch 维度
                        if key in ['input_features', 'input_features_mask']:
                            # 这些字段可能需要保留 batch 维度，先不处理
                            result[key] = value
                        else:
                            # 其他字段可以移除 batch 维度
                            result[key] = value.squeeze(0) if value.dim() > 1 else value
                    else:
                        result[key] = value
                else:
                    result[key] = value
        
        return result


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
    CATSLU数据集的批处理函数，将多个样本组合为一个批次（Gemma-3N版本）
    """
    input_ids_list = []
    labels_list = []
    
    # 收集音频相关字段
    input_features_list = []
    input_features_mask_list = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        
        # 收集音频特征（Gemma-3N 使用 input_features）
        if 'input_features' in inputs:
            # 移除 batch 维度（如果有）
            feat = inputs['input_features']
            if feat.dim() > 2:
                feat = feat.squeeze(0)  # 移除可能的 batch 维度
            input_features_list.append(feat)
        
        if 'input_features_mask' in inputs:
            mask = inputs['input_features_mask']
            if mask.dim() > 1:
                mask = mask.squeeze(0)
            input_features_mask_list.append(mask)
    
    # 填充序列到相同长度
    input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
    labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
    attention_mask = (input_ids != 0).long()  # 创建注意力掩码
    
    # 构建批次特征
    batch_feature = BatchFeature({
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    })
    
    # 处理音频特征
    if input_features_list:
        try:
            # 打印调试信息
            # print(f"Batching input_features: shapes={[f.shape for f in input_features_list]}")
            
            # 确保所有特征都是正确的维度
            # Gemma-3N 的 input_features 应该是 [T, D] 格式（2D）
            # 如果是 3D 或更高维，需要处理
            processed_features = []
            for feat in input_features_list:
                # 移除所有大小为 1 的维度
                while feat.dim() > 2 and 1 in feat.shape:
                    feat = feat.squeeze()
                # 如果仍然是 3D 或更高，取第一个（可能是 batch 维度）
                if feat.dim() > 2:
                    feat = feat[0] if feat.shape[0] == 1 else feat
                # 确保是 2D [T, D]
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                processed_features.append(feat)
            
            # 现在所有特征都应该是 [T, D] 格式
            # 在时间维度上 padding，然后在 batch 维度上 stack
            max_time = max(f.shape[0] for f in processed_features)
            padded_features = []
            for feat in processed_features:
                if feat.shape[0] < max_time:
                    padding = torch.zeros(max_time - feat.shape[0], feat.shape[1], 
                                        dtype=feat.dtype, device=feat.device)
                    feat = torch.cat([feat, padding], dim=0)
                padded_features.append(feat)
            
            # Stack 成 [B, T, D] 格式
            batch_feature['input_features'] = torch.stack(padded_features)
            # print(f"Batched input_features shape: {batch_feature['input_features'].shape}")
            
        except Exception as e:
            print(f"Error batching input_features: {e}")
            print(f"Original shapes: {[f.shape for f in input_features_list]}")
            import traceback
            traceback.print_exc()
            raise
    
    if input_features_mask_list:
        try:
            # 处理 mask
            if all(m.shape == input_features_mask_list[0].shape for m in input_features_mask_list):
                batch_feature['input_features_mask'] = torch.stack(input_features_mask_list)
            else:
                max_len = max(m.shape[0] for m in input_features_mask_list)
                padded_masks = []
                for mask in input_features_mask_list:
                    if mask.shape[0] < max_len:
                        padding = torch.zeros(max_len - mask.shape[0], dtype=mask.dtype, device=mask.device)
                        mask = torch.cat([mask, padding], dim=0)
                    padded_masks.append(mask)
                batch_feature['input_features_mask'] = torch.stack(padded_masks)
        except Exception as e:
            print(f"Error batching input_features_mask: {e}")
            # 如果失败，尝试创建默认 mask
            if input_features_list:
                batch_feature['input_features_mask'] = torch.ones(
                    batch_feature['input_features'].shape[:2], 
                    dtype=torch.bool, 
                    device=batch_feature['input_features'].device
                )
    
    return batch_feature


def create_model(model_name_or_path, use_flash_attention=False, use_lora=True):
    """
    创建Gemma-3N模型（单GPU版本），可选使用LoRA
    """
    gpu_ids = GPU_IDS
    num_gpus = len(gpu_ids)
    print(f"使用GPU: {gpu_ids} (单GPU模式)")
    main_device = f"cuda:{gpu_ids[0]}"
    
    # 单GPU情况：直接使用设备ID
    device_map = main_device
    
    # 加载Gemma-3N模型
    # 使用 bfloat16 以节省显存（RTX 5090 支持，且比 fp16 更稳定）
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16，节省显存且更稳定
        trust_remote_code=True,
        device_map=device_map
    )
    
    # 如果使用 LoRA，应用 LoRA 配置
    if use_lora:
        print("应用 LoRA 配置以节省显存...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def main():
    """主函数，包含模型训练的完整流程"""

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
    )
    
    # 创建模型（使用 LoRA 以节省显存）
    model = create_model(
        MODEL_NAME_OR_PATH,
        use_flash_attention=USE_FLASH_ATTENTION,
        use_lora=USE_LORA,
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

    # 计算GPU数量并进行批处理大小断言
    gpu_ids = GPU_IDS
    num_gpus = len(gpu_ids)
    print(f'training on {num_gpus} GPU(s) (RTX 5090)')
    
    # 单GPU情况下的批处理大小计算
    if num_gpus == 1:
        gradient_accumulation_steps = BATCH_SIZE // BATCH_SIZE_PER_GPU
        assert (
            BATCH_SIZE % BATCH_SIZE_PER_GPU == 0
        ), f'Batch size ({BATCH_SIZE}) must be divisible by batch size per GPU ({BATCH_SIZE_PER_GPU})'
    else:
        assert (
            BATCH_SIZE % (num_gpus * BATCH_SIZE_PER_GPU) == 0
        ), 'Batch size must be divisible by the number of GPUs'
        gradient_accumulation_steps = BATCH_SIZE // (num_gpus * BATCH_SIZE_PER_GPU)

    # 使用混合精度训练以节省显存
    # 注意：如果模型已经是 fp16，Trainer 的 fp16 可能会冲突
    # RTX 5090 支持 bfloat16，使用 bf16 更稳定
    fp16 = False
    bf16 = True   # 使用 bfloat16，更稳定且节省显存

    # 设置训练参数（优化显存使用）
    training_args = TrainingArguments(
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        gradient_checkpointing=True,  # 启用梯度检查点以节省显存
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
        dataloader_num_workers=2,  # 减少数据加载进程数以节省显存
        ddp_find_unused_parameters=False,  # 单 GPU 时不需要
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
    
    # 保存模型
    trainer.save_model()
    
    # 如果使用 LoRA，保存 LoRA 权重
    if USE_LORA:
        model.save_pretrained(training_args.output_dir)
        print(f"LoRA 权重已保存到 {training_args.output_dir}")

    processor.save_pretrained(training_args.output_dir)

    print('Training completed successfully!')


if __name__ == '__main__':
    main()
