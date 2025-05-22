"""
finetune Phi-4-multimodal-instruct on an speech task
在语音任务上微调 Phi-4-multimodal-instruct 模型

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
accelerate==1.3.0
"""

import argparse
import json
import os
from pathlib import Path

import torch
import sacrebleu
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
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


# 不同语言对的指令模板
INSTSRUCTION = {
    "en_zh-CN": "Translate the audio to Mandarin.",
    "en_id": "Translate the audio to Indonesian.",
    "en_sl": "Translate the audio to Slovenian.",
}
# 不同语言的分词器配置
TOKENIZER = {
    "en_zh-CN": "zh",
    "en_ja": "ja-mecab",
}
# 答案后缀标记，用于标识生成结束
ANSWER_SUFFIX = "<|end|><|endoftext|>"
# 标签忽略索引值，用于损失计算中忽略某些位置
_IGNORE_INDEX = -100
# 训练和评估数据集大小限制
_TRAIN_SIZE = 50000
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

class CoVoSTDataset(Dataset):
    """CoVoST数据集类，用于语音翻译任务"""
    def __init__(self, processor, data_dir, split, 
                 lang="en_zh-CN", rank=0, world_size=1):

        # 加载CoVoST2数据集，指定语言对和数据目录
        self.data = load_dataset("facebook/covost2", 
                           lang, 
                           data_dir=data_dir, 
                           split=split,
                           trust_remote_code=True
                           )
        self.training = "train" in split  # 判断是否为训练数据集
        self.processor = processor  # 用于处理输入的处理器
        self.instruction = INSTSRUCTION[lang]  # 获取对应语言的指令
        
        # 如果在分布式环境中，分片数据集
        if world_size > 1:
            self.data = self.data.shard(world_size, rank) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本并处理为模型可接受的格式
        {'client_id': '0013037a1d45cc33460806cc3f8ecee9d536c45639ba4cbbf1564f1c051f53ff3c9f89ef2f1bf04badf55b3a2e7654c086f903681a7b6299616cff6f67598eff',
        'file': '{data_dir}/clips/common_voice_en_699711.mp3',
        'audio': {'path': '{data_dir}/clips/common_voice_en_699711.mp3',
        'array': array([-1.28056854e-09, -1.74622983e-09, -1.16415322e-10, ...,
                3.92560651e-10,  6.62794264e-10, -3.89536581e-09]),
        'sampling_rate': 16000},
        'sentence': '"She\'ll be all right."',
        'translation': '她会没事的。',
        'id': 'common_voice_en_699711'}
        """
        data = self.data[idx]
        # 构建用户消息，包含音频标记和指令
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + self.instruction,
        }
        # 应用聊天模板，构建提示
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        # 处理文本和音频输入
        inputs = self.processor(text=prompt, audios=[(data["audio"]["array"], data["audio"]["sampling_rate"])], return_tensors='pt')
        
        # 构建答案，添加结束标记
        answer = f"{data['translation']}{ANSWER_SUFFIX}"
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


def covost_collate_fn(batch):
    """
    CoVoST数据集的批处理函数，将多个样本组合为一个批次
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


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    """
    评估模型性能
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()  # 设置为评估模式
    all_generated_texts = []  # 存储生成的文本
    all_labels = []  # 存储实际标签

    # 创建评估数据加载器
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=covost_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
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
        inputs = inputs.to(f'cuda:{local_rank}')
        # 生成翻译文本
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64,
            stopping_criteria=stopping_criteria,
        )

        # 处理停止标记位置
        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]

        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        # 解码生成的文本，仅保留模型生成部分
        generated_text = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        all_generated_texts.extend(generated_text)
        # 解码标签
        labels = [processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX) for _label_ids in inputs["labels"]]
        all_labels.extend(labels)

    # 在分布式环境中收集所有进程的结果
    all_generated_texts = gather_object(all_generated_texts)
    all_labels = gather_object(all_labels)
    
    # 只在主进程中计算BLEU分数
    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)
        bleu = sacrebleu.corpus_bleu(all_generated_texts, [all_labels])
        print(bleu)
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'all_generated_texts': all_generated_texts,
                    'all_labels': all_labels,
                    'score': bleu.score,
                }
                json.dump(save_dict, f)

        return bleu.score
    return None


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
        "--common_voice_dir",
        type=str,
        default="CommonVoice/EN",
        help="Unzipped Common Voice Audio dataset directory, refer to https://commonvoice.mozilla.org/en/datasets, version 4.0",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en_sl",
        help="Language pair for translation.",
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=32,
        help='Batch size per GPU (adjust this to fit in GPU memory)',
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    args = parser.parse_args()

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

    # 创建评估数据集
    eval_dataset = CoVoSTDataset(processor,
                                 data_dir=args.common_voice_dir,
                                 split=f'test[:{_EVAL_SIZE}]',
                                 lang=args.lang,
                                 rank=rank,
                                 world_size=world_size)
    
    # 创建训练数据集
    train_dataset = CoVoSTDataset(processor,
                                  data_dir=args.common_voice_dir,
                                  split=f'train[:{_TRAIN_SIZE}]',
                                  lang=args.lang)

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
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # 用于未使用的SigLIP层
    )

    # 微调前先评估
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'BLEU Score before finetuning: {score}')

    # 创建Trainer实例并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=covost_collate_fn,
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
    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'BLEU Score after finetuning: {score}')


if __name__ == '__main__':
    main()
