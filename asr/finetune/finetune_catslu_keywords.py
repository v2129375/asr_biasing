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
import argparse

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

# 定义命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='微调ASR模型的参数')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1], help='使用的GPU ID列表')
    parser.add_argument('--model_name_or_path', type=str, default='asr/model/Saishellp1keywords', help='预训练模型路径或名称')
    parser.add_argument('--data_path', type=str, default="data/catslu/train.csv", help='训练数据集路径')
    parser.add_argument('--keywords_dir', type=str, default="data/catslu", help='关键词目录')
    parser.add_argument('--use_flash_attention', action='store_true', default=True, help='是否使用Flash Attention')
    parser.add_argument('--output_dir', type=str, default='asr/model/new', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=2, help='总批次大小')
    parser.add_argument('--batch_size_per_gpu', type=int, default=1, help='每个GPU的批次大小')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='学习率')
    parser.add_argument('--wd', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--tqdm_enabled', action='store_true', default=True, help='是否启用tqdm进度条')
    parser.add_argument('--device_map_path', type=str, default='asr/finetune/device_map.json', help='设备映射配置文件路径')
    parser.add_argument('--use_keywords', action='store_true', default=True, help='是否使用关键词')
    parser.add_argument('--num_keywords', type=int, default=0, help='随机选择的关键词数量，0表示使用全部关键词')
    parser.add_argument('--num_sentences', type=int, default=0, help='随机选择的语句数量，0表示使用全部语句')
    parser.add_argument('--randomize_domain', action='store_true', default=False, help='是否随机指定领域给训练资料')
    return parser.parse_args()

# 全局参数设置
args = parse_args()
GPU_IDS = args.gpu_ids
MODEL_NAME_OR_PATH = args.model_name_or_path
BASE_MODEL_NAME_OR_PATH = "microsoft/Phi-4-multimodal-instruct"
DATA_PATH = args.data_path
KEYWORDS_DIR = args.keywords_dir
USE_FLASH_ATTENTION = args.use_flash_attention
OUTPUT_DIR = args.output_dir
BATCH_SIZE = args.batch_size
BATCH_SIZE_PER_GPU = args.batch_size_per_gpu
NUM_TRAIN_EPOCHS = args.num_train_epochs
LEARNING_RATE = args.learning_rate
WD = args.wd
TQDM_ENABLED = args.tqdm_enabled
DEVICE_MAP_PATH = args.device_map_path


USE_KEYWORDS = args.use_keywords
# 关键词随机选择参数
NUM_KEYWORDS = args.num_keywords  # 随机选择的关键词数量，设为0表示使用全部关键词
# 随机选择语句参数
NUM_SENTENCES = args.num_sentences  # 随机选择的语句数量，设为0表示使用全部语句
# 随机指定领域参数
RANDOMIZE_DOMAIN = args.randomize_domain  # 设置为True时会随机指定领域给训练资料




# 基础任务指令
BASE_INSTRUCTION = "Transcribe the audio clip into text."
# 带关键词的任务指令模板
KEYWORD_INSTRUCTION_TEMPLATE = "Transcribe the audio clip into text. Pay attention to these keywords: {keywords}"
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
        
        # 检查是否存在source列
        self.has_source_column = 'source' in self.data.columns
        
        if self.has_source_column:
            # 检查source列的值
            unique_sources = self.data['source'].unique()
            print(f"数据集中发现的领域: {unique_sources}")
            
            # 验证source值是否在支持的范围内
            supported_sources = ['video', 'music', 'city']
            unsupported_sources = [s for s in unique_sources if s not in supported_sources]
            if unsupported_sources:
                print(f"警告: 发现不支持的领域: {unsupported_sources}")
                print(f"支持的领域: {supported_sources}")
        else:
            print("警告: 数据集中没有发现'source'列。在RANDOMIZE_DOMAIN=True时将随机分配领域。")
        
        # 加载各领域的关键词
        self.keywords_dict = self._load_keywords(keywords_dir)
        self.supported_domains = list(self.keywords_dict.keys())
        
        if not self.supported_domains:
            print("警告: 未找到任何领域的关键词文件。将使用基础指令。")
        else:
            print(f"可用的领域: {self.supported_domains}")
            
        # 随机选择指定数量的语句
        if NUM_SENTENCES > 0 and self.training:
            total_samples = len(self.data)
            if NUM_SENTENCES < total_samples:
                if self.has_source_column:
                    # 根据source类别进行平衡选择
                    print("根据source类别进行平衡选择训练语句")
                    source_counts = self.data['source'].value_counts()
                    print(f"各类别原始样本数量: {source_counts.to_dict()}")
                    
                    selected_indices = []
                    total_selected = 0
                    
                    # 计算每个类别应选择的样本数量（按比例分配）
                    for i, (source, count) in enumerate(source_counts.items()):
                        if i == len(source_counts) - 1:  # 最后一个类别，分配剩余的所有样本
                            samples_for_this_source = NUM_SENTENCES - total_selected
                        else:
                            # 按比例计算每个类别的样本数
                            proportion = count / total_samples
                            samples_for_this_source = int(NUM_SENTENCES * proportion)
                        
                        # 确保不超过该类别的总样本数
                        samples_for_this_source = min(samples_for_this_source, count)
                        
                        if samples_for_this_source > 0:
                            # 从该类别中随机选择样本
                            source_data_indices = self.data[self.data['source'] == source].index.tolist()
                            selected_source_indices = random.sample(source_data_indices, samples_for_this_source)
                            selected_indices.extend(selected_source_indices)
                            total_selected += samples_for_this_source
                            print(f"从 '{source}' 类别选择 {samples_for_this_source} 个样本")
                    
                    # 重新索引数据集
                    self.data = self.data.loc[selected_indices].reset_index(drop=True)
                    
                    # 输出最终的类别分布
                    final_source_counts = self.data['source'].value_counts()
                    print(f"选择后各类别样本数量: {final_source_counts.to_dict()}")
                    print(f"总共选择了 {len(self.data)} 条语句进行训练，原始数据集大小: {total_samples}")
                else:
                    # 没有source列，使用原有的随机选择方式
                    selected_indices = random.sample(range(total_samples), NUM_SENTENCES)
                    self.data = self.data.iloc[selected_indices].reset_index(drop=True)
                    print(f"随机选择了 {NUM_SENTENCES} 条语句进行训练，原始数据集大小: {total_samples}")
            else:
                print(f"NUM_SENTENCES ({NUM_SENTENCES}) 大于等于数据集总大小 ({total_samples})，使用全部语句")
        
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
        # 如果启用了随机领域，则从所有支持的领域中随机选择一个
        if RANDOMIZE_DOMAIN and self.training and self.supported_domains:
            # 只在第一次随机替换时打印一次提示
            if not hasattr(self, '_domain_randomization_logged'):
                self._domain_randomization_logged = True
                print(f"Domain randomization enabled. Randomly assigning domains from: {self.supported_domains}")
            
            # 如果source不在支持的领域中，则随机选择一个
            if source not in self.supported_domains:
                random_domain = random.choice(self.supported_domains)
                domain_keywords = self._get_domain_keywords(random_domain)
                actual_domain = random_domain
            else:
                # 使用传入的source，但在__getitem__中可能已经随机选择了
                domain_keywords = self._get_domain_keywords(source)
                actual_domain = source
        else:
            domain_keywords = self._get_domain_keywords(source)
            actual_domain = source
        
        if domain_keywords and USE_KEYWORDS:
            # 使用该领域的关键词（可能是随机选择的）
            keywords_str = ' '.join(domain_keywords)
            return KEYWORD_INSTRUCTION_TEMPLATE.format(intent=actual_domain, keywords=keywords_str)
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
        
        # 获取source字段，如果不存在则处理
        if self.has_source_column:
            source = data['source']
            if pd.isna(source) or source == '':
                # 如果为空值且RANDOMIZE_DOMAIN为True，或者没有source列
                if RANDOMIZE_DOMAIN and self.supported_domains:
                    source = random.choice(self.supported_domains)
                else:
                    # 默认使用'video'
                    source = 'video'
        else:
            # 如果没有source列且RANDOMIZE_DOMAIN为True
            if RANDOMIZE_DOMAIN and self.supported_domains:
                source = random.choice(self.supported_domains)
            else:
                # 默认使用'video'
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
        BASE_MODEL_NAME_OR_PATH,
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
        data_path=DATA_PATH,
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
    
    # 输出语句选择信息
    if NUM_SENTENCES > 0:
        print(f"Random sentences selection: enabled (training with {NUM_SENTENCES} sentences)")
        print(f"Class balanced selection: enabled (samples will be balanced across source categories)")
    else:
        print(f"Using all available sentences for training")
    
    # 输出领域随机化信息
    if RANDOMIZE_DOMAIN:
        print(f"Domain randomization: enabled (randomly assigning domains to training samples)")
    else:
        print(f"Using original domains from dataset")

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