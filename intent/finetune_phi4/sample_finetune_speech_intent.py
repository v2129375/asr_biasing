"""
finetune Phi-4-multimodal-instruct on an speech intent classification task
在语音意图分类任务上微调 Phi-4-multimodal-instruct 模型

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import soundfile as sf
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

# Device map配置文件路径
DEVICE_MAP_CONFIG_PATH = "intent/finetune_phi4/device_map.json"

# 任务指令
INSTRUCTION = """
Task:
You are given an audio file in Mandarin Chinese containing a single spoken instruction. Your task is to classify the intent into one of three categories.
Classification process:
First, determine if the audio intent belongs to either video or music.
If it is not related to video or music, output city.
If it is related to video or music, pay extra attention to distinguishing between these two categories:
If the content could be both (such as a title that is both a song and a TV series), classify it according to common public perception.
For example, if the query is about a song, output music; if it is about a TV show or movie, output video.
Category definitions:
video: Related to movies, TV shows, cartoons, or any video content.
Examples: "我想看战狼" ("I want to watch Wolf Warrior"), "播放熊出没" ("Play Boonie Bears")
music: Related to songs, singers, or music.
Examples: "播放小星星" ("Play Twinkle Twinkle Little Star"), "我要听周杰伦的歌" ("I want to listen to Jay Chou's songs")
city: Related to weather or city information.
Examples: "北京今天天气怎么样" ("What's the weather like in Beijing today?"), "上海的温度" ("Temperature in Shanghai")
Instructions:
Only respond with one of these three labels: video, music, or city (in English, all lowercase).
Do not output any other explanation or language.
If the intent does not exactly fit video or music, always choose city by default.
When in doubt between video and music, use the category that matches common public understanding.
Input:
A Mandarin Chinese audio file with a single spoken instruction.
Output:
Only output one of: video, music, or city.
"""
# 答案后缀标记，用于标识生成结束
ANSWER_SUFFIX = "<|end|><|endoftext|>"
# 标签忽略索引值，用于损失计算中忽略某些位置
_IGNORE_INDEX = -100
# 训练和评估数据集大小限制
_TRAIN_SIZE = None  # 使用全部训练数据
_EVAL_SIZE = 200


def load_device_map(config_path):
    """
    从JSON文件加载device_map配置
    
    Args:
        config_path: JSON配置文件路径
        
    Returns:
        device_map: 设备映射配置，如果文件不存在则返回'auto'
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                device_map = json.load(f)
            print(f"从 {config_path} 加载device_map配置: {device_map}")
            return device_map
        else:
            print(f"Device map配置文件 {config_path} 不存在，使用默认配置 'auto'")
            return 'auto'
    except Exception as e:
        print(f"读取device_map配置文件时出错: {e}，使用默认配置 'auto'")
        return 'auto'


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


class CatsluDataset(Dataset):
    """CATSLU数据集类，用于语音意图分类任务"""
    def __init__(self, processor, data_path, split="train"):
        """
        初始化CATSLU数据集
        
        Args:
            processor: 模型处理器
            data_path: CSV数据文件路径
            split: 数据集划分，'train'或'eval'
        """
        # 读取CSV文件
        self.data = pd.read_csv(data_path)
        self.training = "train" in split
        self.processor = processor
        self.instruction = INSTRUCTION
        
        # 提取所有唯一的意图类别（source列）
        self.intent_categories = sorted(self.data['source'].unique().tolist())
        self.intent_to_id = {intent: idx for idx, intent in enumerate(self.intent_categories)}

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
        
        # 构建用户消息
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + self.instruction,
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
        
        # 获取意图标签
        intent_label = data['source']
        
        # 构建答案，添加结束标记
        answer = f"{intent_label}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        
        # 根据是否为训练模式，构建不同的输入和标签
        if self.training:
            # 训练时，将输入和答案连接起来，标签只关注答案部分
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            # 评估时，输入和标签分开
            input_ids = inputs.input_ids
            labels = answer_ids

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
    intent_list = []
    
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
    return BatchFeature(
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


def create_model(model_name_or_path, use_flash_attention=False, device_map='auto'):
    """
    创建因果语言模型，可选择使用flash attention加速和自定义device_map
    
    Args:
        model_name_or_path: 模型名称或路径
        use_flash_attention: 是否使用flash attention
        device_map: 设备映射配置
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
        device_map=device_map,
    )

    return model


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    """
    评估模型在意图分类任务上的性能
    """
    model.eval()  # 设置为评估模式
    all_generated_intents = []  # 存储生成的意图
    all_labels = []  # 存储实际标签

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
    # 获取模型的第一个参数设备作为目标设备
    model_device = next(model.parameters()).device
    stop_tokens_ids = stop_tokens_ids.to(model_device)

    # 遍历评估数据集
    for inputs in tqdm(eval_dataloader, disable=disable_tqdm, desc='running eval'):
        # 设置停止条件
        stopping_criteria=StoppingCriteriaList([MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=inputs.input_ids.size(0))])
        inputs_to_model = {k: v for k, v in inputs.items() if k != 'intent'}
        inputs_to_model = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs_to_model.items()}
        
        # 生成意图预测
        generated_ids = model.generate(
            **inputs_to_model, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            max_new_tokens=16,  # 意图分类只需要较短的生成长度
            stopping_criteria=stopping_criteria,
            do_sample=False,  # 使用贪婪解码
            num_logits_to_keep=1,  # 只保留最后一个token的logits
            pad_token_id=processor.tokenizer.eos_token_id,  # 设置pad_token_id
        )

        # 处理停止标记位置
        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]

        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        
        # 解码生成的意图，仅保留模型生成部分
        generated_intents = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        
        # 移除生成文本中的ANSWER_SUFFIX
        generated_intents = [intent.replace(ANSWER_SUFFIX, "").strip() for intent in generated_intents]
        all_generated_intents.extend(generated_intents)
        # 解码标签
        labels = [processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX) for _label_ids in inputs["labels"]]
        all_labels.extend(labels)

    # 计算分类指标
    accuracy = accuracy_score(y_true=all_labels, y_pred=all_generated_intents)
    f1 = f1_score(y_true=all_labels, y_pred=all_generated_intents, average='weighted')
    report = classification_report(y_true=all_labels, y_pred=all_generated_intents, output_dict=True, digits=4)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_true=all_labels, y_pred=all_generated_intents, digits=4))
    
    if save_path:
        with open(save_path, 'w') as f:
            save_dict = {
                'all_generated_intents': all_generated_intents,
                'all_labels': all_labels,
                'accuracy': accuracy,
                'f1': f1,
                'report': report,
            }
            json.dump(save_dict, f)

    return accuracy, f1


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
        default="tts/tts_data/sentences_audio.csv",
        help="Path to the CATSLU dataset CSV file",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=200,
        help="Number of samples to use for evaluation",
    )
    parser.add_argument('--use_flash_attention', action='store_true', default=True, help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='intent/model/new/', help='Output directory')
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

    # 加载device_map配置
    device_map = load_device_map(DEVICE_MAP_CONFIG_PATH)
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    # 创建模型
    model = create_model(
        args.model_name_or_path,
        use_flash_attention=args.use_flash_attention,
        device_map=device_map,
    )

    # 设置模型使用语音适配器
    model.set_lora_adapter('speech')
    
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
    eval_dataset = CatsluDataset(
        processor,
        data_path=eval_path,
        split="eval"
    )
    
    # 创建训练数据集
    train_dataset = CatsluDataset(
        processor,
        data_path=train_path,
        split="train"
    )
    
    # 输出数据集统计信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Intent categories: {train_dataset.intent_categories}")

    # 简化批处理大小计算
    gradient_accumulation_steps = args.batch_size // args.batch_size_per_gpu

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
    )

    # 微调前先评估
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    accuracy, f1 = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    print(f'Accuracy before finetuning: {accuracy}')
    print(f'F1 Score before finetuning: {f1}')

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
        device_map=device_map,
    )

    # 进行微调后的评估
    accuracy, f1 = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    print(f'Accuracy after finetuning: {accuracy}')
    print(f'F1 Score after finetuning: {f1}')
    
    # 清理临时文件
    if _EVAL_SIZE is not None and _EVAL_SIZE < total_samples:
        if os.path.exists(eval_path) and eval_path != args.catslu_data_path:
            os.remove(eval_path)
        if os.path.exists(train_path) and train_path != args.catslu_data_path:
            os.remove(train_path)


if __name__ == '__main__':
    main() 