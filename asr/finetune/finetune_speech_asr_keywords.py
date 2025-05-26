"""
finetune Phi-4-multimodal-instruct on an ASR (Automatic Speech Recognition) task with domain-specific keywords
在ASR（自动语音识别）任务上微调 Phi-4-multimodal-instruct 模型，支持领域特定关键词

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
accelerate==1.3.0
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
)


# 基础任务指令
BASE_INSTRUCTION = "Transcribe the audio clip into text."
# 带关键词的任务指令模板
KEYWORD_INSTRUCTION_TEMPLATE = "Transcribe the audio clip into text. Pay attention to these keywords: {keywords}"
# 答案后缀标记，用于标识生成结束
ANSWER_SUFFIX = "<|end|><|endoftext|>"
# 标签忽略索引值，用于损失计算中忽略某些位置
_IGNORE_INDEX = -100
# 训练和评估数据集大小限制
_TRAIN_SIZE = None  # 使用全部训练数据
_EVAL_SIZE = 200


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """
    停止条件类，能够处理多个停止标记和批处理输入。
    Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs.
    """

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """
        初始化多标记批处理停止条件。
        Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: 停止标记。
            batch_size: 批处理大小。
        """
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 只收集与停止标记兼容的最大数量的输入
        # 并检查生成的输入是否等于 `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # 为批处理中的每个输入标记停止标记产生的位置，
        # 但仅当相应的条目尚未设置时
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


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
            return self.keywords_dict[source]
        else:
            # 如果没有找到对应的关键词，打印警告并返回空列表
            print(f"Warning: No keywords found for source '{source}', using empty keyword list")
            return []

    def _build_instruction_with_keywords(self, source):
        """根据领域构建包含关键词的指令"""
        domain_keywords = self._get_domain_keywords(source)
        
        if domain_keywords:
            # 使用该领域的全部关键词
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
        
        # 根据是否为训练模式，构建不同的输入和标签
        if self.training:
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
        else:
            # 评估时，输入和标签分开，并返回关键词信息
            input_ids = inputs.input_ids
            labels = answer_ids
            keyword = data.get('keyword', '')  # 获取关键词，如果不存在则为空字符串
            return {
                'input_ids': input_ids,
                'labels': labels,
                'input_audio_embeds': inputs.input_audio_embeds,
                'audio_embed_sizes': inputs.audio_embed_sizes,
                'keyword': keyword,
                'manual_transcript': transcript_label,
                'source': source,
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
    keywords_list = []
    manual_transcripts_list = []
    sources_list = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )
        # 如果是评估模式，添加关键词和转录文本信息
        if 'keyword' in inputs:
            keywords_list.append(inputs['keyword'])
            manual_transcripts_list.append(inputs['manual_transcript'])
            sources_list.append(inputs.get('source', 'video'))
        

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
    
    # 如果是评估模式，添加关键词、转录文本和数据源信息
    if keywords_list:
        batch_feature['keywords'] = keywords_list
        batch_feature['manual_transcripts'] = manual_transcripts_list
        batch_feature['sources'] = sources_list
        
    return batch_feature


def create_model(model_name_or_path, use_flash_attention=False):
    """
    创建因果语言模型，可选择使用flash attention加速
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')

    return model


def cer(r: list, h: list):
    """
    使用Levenshtein距离计算字符错误率(CER)
    Calculation of CER with Levenshtein distance.
    """
    # 初始化
    import numpy
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # 计算
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / float(len(r))


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    """
    评估模型在ASR任务上的性能，计算CER和Keyword Error Rate，并按领域统计
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()  # 设置为评估模式
    all_generated_transcripts = []  # 存储生成的转录结果
    all_ground_truth = []  # 存储真实转录文本
    all_keywords = []  # 存储关键词
    all_sources = []  # 存储数据源信息

    # 创建评估数据加载器
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=catslu_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True,
    )
    
    # 定义停止标记
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f'cuda:{local_rank}')

    # 遍历评估数据集
    for inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='running eval'
    ):
        # 设置停止条件
        stopping_criteria=StoppingCriteriaList([MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=inputs.input_ids.size(0))])
        inputs_to_model = {k: v for k, v in inputs.items() if k not in ['keywords', 'manual_transcripts', 'sources']}
        inputs_to_model = {k: v.to(f'cuda:{local_rank}') if isinstance(v, torch.Tensor) else v for k, v in inputs_to_model.items()}
        
        # 生成ASR转录
        generated_ids = model.generate(
            **inputs_to_model, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            max_new_tokens=64,  # ASR任务需要更长的生成长度
            stopping_criteria=stopping_criteria,
        )

        # 处理停止标记位置
        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]

        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        
        # 解码生成的转录结果，仅保留模型生成部分
        generated_transcripts = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        
        # 移除生成文本中的ANSWER_SUFFIX
        generated_transcripts = [transcript.replace(ANSWER_SUFFIX, "").strip() for transcript in generated_transcripts]
        all_generated_transcripts.extend(generated_transcripts)
        
        # 解码真实转录文本
        ground_truth = [processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX) for _label_ids in inputs["labels"]]
        all_ground_truth.extend(ground_truth)
        
        # 获取关键词和数据源信息（如果有的话）
        if 'keywords' in inputs:
            all_keywords.extend(inputs['keywords'])
            all_sources.extend(inputs['sources'])

    # 在分布式环境中收集所有进程的结果
    all_generated_transcripts = gather_object(all_generated_transcripts)
    all_ground_truth = gather_object(all_ground_truth)
    all_keywords = gather_object(all_keywords)
    all_sources = gather_object(all_sources)
    
    # 只在主进程中计算评估指标
    if rank == 0:
        assert len(all_generated_transcripts) == len(all_ground_truth)
        
        # 计算总体CER
        total_cer = 0
        for i in range(len(all_generated_transcripts)):
            ground = all_ground_truth[i]
            asr_result = all_generated_transcripts[i]
            ground_chars = [x for x in ground]
            asr_chars = [x for x in asr_result]
            this_cer = cer(ground_chars, asr_chars)
            total_cer += this_cer
        
        average_cer = total_cer / len(all_generated_transcripts)
        
        # 按领域计算CER
        domain_stats = {}
        for domain in ['video', 'music', 'city']:
            domain_indices = [i for i, source in enumerate(all_sources) if source == domain]
            if domain_indices:
                domain_cer = 0
                for i in domain_indices:
                    ground = all_ground_truth[i]
                    asr_result = all_generated_transcripts[i]
                    ground_chars = [x for x in ground]
                    asr_chars = [x for x in asr_result]
                    domain_cer += cer(ground_chars, asr_chars)
                domain_stats[domain] = {
                    'count': len(domain_indices),
                    'cer': domain_cer / len(domain_indices)
                }
        
        # 计算总体Keyword Error Rate
        keyword_error_count = 0
        total_keyword_samples = 0
        if all_keywords:
            for i in range(len(all_generated_transcripts)):
                if all_keywords[i]:  # 只有当关键词存在时才计算
                    total_keyword_samples += 1
                    if all_keywords[i] not in all_generated_transcripts[i]:
                        keyword_error_count += 1
            
            keyword_error_rate = keyword_error_count / total_keyword_samples if total_keyword_samples > 0 else 0
        else:
            keyword_error_rate = None
        
        # 按领域计算Keyword Error Rate
        domain_keyword_stats = {}
        if all_keywords:
            for domain in ['video', 'music', 'city']:
                domain_indices = [i for i, source in enumerate(all_sources) if source == domain]
                domain_keyword_error = 0
                domain_keyword_total = 0
                for i in domain_indices:
                    if all_keywords[i]:
                        domain_keyword_total += 1
                        if all_keywords[i] not in all_generated_transcripts[i]:
                            domain_keyword_error += 1
                
                if domain_keyword_total > 0:
                    domain_keyword_stats[domain] = {
                        'keyword_error_count': domain_keyword_error,
                        'keyword_total': domain_keyword_total,
                        'keyword_error_rate': domain_keyword_error / domain_keyword_total
                    }
        
        # 打印结果
        print(f"Overall CER: {average_cer:.4f}")
        if keyword_error_rate is not None:
            print(f"Overall Keyword Error Rate: {keyword_error_rate:.4f}")
        
        print("\n按领域统计:")
        for domain, stats in domain_stats.items():
            print(f"{domain.upper()} - Count: {stats['count']}, CER: {stats['cer']:.4f}")
            if domain in domain_keyword_stats:
                kw_stats = domain_keyword_stats[domain]
                print(f"  Keyword Error Rate: {kw_stats['keyword_error_rate']:.4f} ({kw_stats['keyword_error_count']}/{kw_stats['keyword_total']})")
        
        # 打印一些错误识别的样本
        print("\n错误识别的样本:")
        error_count = 0
        for i in range(len(all_generated_transcripts)):
            if all_keywords and i < len(all_keywords) and all_keywords[i] and all_keywords[i] not in all_generated_transcripts[i]:
                print(f"领域: {all_sources[i] if i < len(all_sources) else 'unknown'}")
                print(f"原始文本: {all_ground_truth[i]}")
                print(f"识别结果: {all_generated_transcripts[i]}")
                if all_keywords[i]:
                    print(f"关键词: {all_keywords[i]}")
                print("---")
                error_count += 1
                if error_count >= 10:  # 只显示前10个错误样本
                    break
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                save_dict = {
                    'all_generated_transcripts': all_generated_transcripts,
                    'all_ground_truth': all_ground_truth,
                    'all_keywords': all_keywords,
                    'all_sources': all_sources,
                    'overall_cer': average_cer,
                    'overall_keyword_error_rate': keyword_error_rate,
                    'domain_stats': domain_stats,
                    'domain_keyword_stats': domain_keyword_stats,
                    'total_samples': len(all_generated_transcripts),
                    'keyword_error_count': keyword_error_count,
                    'total_keyword_samples': total_keyword_samples,
                }
                json.dump(save_dict, f, ensure_ascii=False, indent=2)

        return average_cer, keyword_error_rate
    return None, None


def main():
    """主函数，包含参数解析、模型训练和评估的完整流程"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        "--catslu_data_path",
        type=str,
        default="data/catslu/train.csv",
        help="Path to the CATSLU dataset CSV file",
    )
    parser.add_argument(
        "--keywords_dir",
        type=str,
        default="data/catslu",
        help="Directory containing keyword files (keyword_video.txt, keyword_music.txt, keyword_city.txt)",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=200,
        help="Number of samples to use for evaluation",
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='asr/model/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=1,
        help='Batch size per GPU (adjust this to fit in GPU memory)',
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    args = parser.parse_args()
    
    # 设置评估数据集大小
    global _EVAL_SIZE
    _EVAL_SIZE = args.eval_size

    # 初始化加速器，用于分布式训练
    accelerator = Accelerator()

    # 确保主进程先加载模型
    with accelerator.local_main_process_first():
        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        # 创建模型
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    # 设置模型使用语音适配器
    model.set_lora_adapter('speech')

    # 获取分布式训练的排名和总进程数
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 读取完整数据集
    full_dataset = pd.read_csv(args.catslu_data_path)
    
    # 划分训练集和评估集
    total_samples = len(full_dataset)
    if _EVAL_SIZE is not None and _EVAL_SIZE < total_samples:
        eval_indices = np.random.choice(total_samples, _EVAL_SIZE, replace=False)
        train_indices = np.array([i for i in range(total_samples) if i not in eval_indices])
        
        eval_data = full_dataset.iloc[eval_indices].reset_index(drop=True)
        train_data = full_dataset.iloc[train_indices].reset_index(drop=True)
        
        # 将划分后的数据保存为临时文件
        eval_path = os.path.join(os.path.dirname(args.catslu_data_path), "eval_temp.csv")
        train_path = os.path.join(os.path.dirname(args.catslu_data_path), "train_temp.csv")
        
        eval_data.to_csv(eval_path, index=False)
        train_data.to_csv(train_path, index=False)
    else:
        # 如果不需要划分或评估集大小过大，则使用相同的数据集
        eval_path = args.catslu_data_path
        train_path = args.catslu_data_path

    # 创建评估数据集
    eval_dataset = CatsluKeywordsDataset(
        processor,
        data_path=eval_path,
        keywords_dir=args.keywords_dir,
        split="eval",
        rank=rank,
        world_size=world_size
    )
    
    # 创建训练数据集
    train_dataset = CatsluKeywordsDataset(
        processor,
        data_path=train_path,
        keywords_dir=args.keywords_dir,
        split="train",
        rank=rank,
        world_size=1  # 训练集不做分片处理
    )
    
    # 输出数据集统计信息
    if accelerator.is_main_process:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        print(f"Keywords directory: {args.keywords_dir}")

    # 计算GPU数量并进行批处理大小断言
    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)

    # 根据是否使用flash attention选择精度
    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # 设置训练参数
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=1,
        ddp_find_unused_parameters=True,  # 用于未使用的SigLIP层
    )

    # 微调前先评估
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cer_before, keyword_error_rate_before = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'CER before finetuning: {cer_before}')
        if keyword_error_rate_before is not None:
            print(f'Keyword Error Rate before finetuning: {keyword_error_rate_before}')

    # 创建Trainer实例并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=catslu_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # 微调后评估（加载保存的检查点）
    # 首先尝试清理GPU内存
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    # 重新加载模型用于推理
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
    ).to('cuda')

    # 进行微调后的评估
    cer_after, keyword_error_rate_after = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'CER after finetuning: {cer_after}')
        if keyword_error_rate_after is not None:
            print(f'Keyword Error Rate after finetuning: {keyword_error_rate_after}')
    
    # 清理临时文件
    if _EVAL_SIZE is not None and _EVAL_SIZE < total_samples:
        if os.path.exists(eval_path) and eval_path != args.catslu_data_path:
            os.remove(eval_path)
        if os.path.exists(train_path) and train_path != args.catslu_data_path:
            os.remove(train_path)


if __name__ == '__main__':
    main() 