import json
import requests
import torch
import os
import io
import random
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen
import gc
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import re
import argparse
import math
from torch.cuda.amp import autocast
# 导入评分模块
from score import evaluate_asr

# 定义所有全局变量及其默认值
USE_KEYWORDS = True  # 是否使用关键词
NUM_KEYWORDS = 0  # 随机选择的关键词数量，设为0表示使用全部关键词
input_data_path = "intent/exp/phi4_intent_result.csv"  # 输入数据路径
model_path = "asr/model/new"  # 模型路径
output_data_path = f"asr/exp/{os.path.basename(model_path)}.csv"  # 输出数据路径
base_model_path = "microsoft/Phi-4-multimodal-instruct"  # 基础模型路径
DEVICE_MAP_PATH = 'asr/finetune/device_map.json'  # 设备映射文件路径
DEFAULT_GPUS = '0,1'  # 默认使用的GPU
DEFAULT_BATCH_SIZE = 1  # 默认批处理大小
BASE_INSTRUCTION = "Transcribe the audio clip into text."  # 基础任务指令（与微调代码保持一致）
KEYWORD_INSTRUCTION_TEMPLATE = "<{intent}> {keywords} </{intent}> Transcribe the audio clip into text."  # 带关键词的任务指令模板
KEYWORDS_DIR = "data/catslu"  # 关键词目录

def parse_args():
    parser = argparse.ArgumentParser(description='ASR with Phi-4 on multiple GPUs')
    parser.add_argument('--use_keywords', type=bool, default=USE_KEYWORDS, help='Whether to use keywords')
    parser.add_argument('--num_keywords', type=int, default=NUM_KEYWORDS, help='Number of keywords to randomly select (0 for all)')
    parser.add_argument('--input', type=str, default=input_data_path, help='Input data path')
    parser.add_argument('--model_path', type=str, default=model_path, help='Model path')
    parser.add_argument('--output', type=str, default=output_data_path, help='Output data path')
    parser.add_argument('--base_model', type=str, default=base_model_path, help='Base model path')
    parser.add_argument('--device_map', type=str, default=DEVICE_MAP_PATH, help='Device map path')
    parser.add_argument('--gpus', type=str, default=DEFAULT_GPUS, help='GPU IDs to use (comma-separated, e.g., "0,1,2")')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for processing')
    parser.add_argument('--keywords_dir', type=str, default=KEYWORDS_DIR, help='Keywords directory')
    parser.add_argument('--base_instruction', type=str, default=BASE_INSTRUCTION, help='Base instruction for ASR')
    parser.add_argument('--keyword_template', type=str, default=KEYWORD_INSTRUCTION_TEMPLATE, help='Keyword instruction template')
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# 更新全局变量
USE_KEYWORDS = args.use_keywords
NUM_KEYWORDS = args.num_keywords
input_data_path = args.input
model_path = args.model_path
output_data_path = args.output
base_model_path = args.base_model
DEVICE_MAP_PATH = args.device_map
DEFAULT_GPUS = args.gpus
DEFAULT_BATCH_SIZE = args.batch_size
KEYWORDS_DIR = args.keywords_dir
BASE_INSTRUCTION = args.base_instruction
KEYWORD_INSTRUCTION_TEMPLATE = args.keyword_template

class Phi4:
    def __init__(self, gpu_ids=None, batch_size=1):
        self.model = None
        self.processor = None
        self.device = None
        self.generation_config = None
        # 关键词字典
        self.keywords_dict = {}
        # GPU ID列表
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        # 批处理大小
        self.batch_size = batch_size
        # 设置主设备
        self.main_device = f"cuda:{self.gpu_ids[0]}"
        
    def load_phi4(self):
        # 获取可用的GPU数量
        num_gpus = len(self.gpu_ids)
        print(f"使用GPU: {self.gpu_ids}")
        
        # 构建device_map
        if num_gpus > 1:
            with open(DEVICE_MAP_PATH, 'r') as f:
                device_map = json.load(f)
        else:
            # 单GPU情况
            device_map = self.main_device

        # Load processor from base model to get necessary configurations
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

        # Load model with device mapping for multiple GPUs
        print(f"正在加载模型到{device_map}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device_map,
            torch_dtype="auto", 
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation='flash_attention_2'
        )

        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        
        # 打印模型分布情况
        if hasattr(self.model, "hf_device_map"):
            print("模型分布情况:")
            for key, device in self.model.hf_device_map.items():
                print(f"  {key}: {device}")
    
    def load_keywords(self, keywords_dir=KEYWORDS_DIR):
        """加载各领域的关键词文件"""
        keyword_files = {
            'video': os.path.join(keywords_dir, 'keyword_video.txt'),
            'music': os.path.join(keywords_dir, 'keyword_music.txt'), 
            'city': os.path.join(keywords_dir, 'keyword_city.txt')
        }
        
        for domain, file_path in keyword_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    keywords = [line.strip() for line in f.readlines() if line.strip()]
                self.keywords_dict[domain] = keywords
                print(f"Loaded {len(keywords)} keywords for {domain} domain")
            except FileNotFoundError:
                print(f"Warning: Keyword file not found: {file_path}")
                self.keywords_dict[domain] = []
    
    def get_domain_keywords(self, intent):
        """根据意图获取对应领域的关键词"""
        # 映射意图到领域
        if 'video' in intent.lower():
            domain = 'video'
        elif 'music' in intent.lower():
            domain = 'music'
        elif 'city' in intent.lower():
            domain = 'city'
        else:
            domain = 'video'  # 默认使用video领域
            
        keywords = self.keywords_dict.get(domain, [])
        
        # 如果设置了随机选择关键词数量且有足够的关键词
        if NUM_KEYWORDS > 0 and USE_KEYWORDS and keywords:
            # 确保选择的关键词数量不超过可用关键词总数
            num_to_select = min(len(keywords), NUM_KEYWORDS)
            # 随机选择指定数量的关键词
            return random.sample(keywords, num_to_select)
        else:
            return keywords
    
    def build_instruction_with_keywords(self, intent):
        """根据意图构建包含关键词的指令"""
        domain_keywords = self.get_domain_keywords(intent)
        
        if domain_keywords and USE_KEYWORDS:
            # 使用该领域的关键词（可能是随机选择的）
            keywords_str = ', '.join(domain_keywords)
            return KEYWORD_INSTRUCTION_TEMPLATE.format(intent=intent, keywords=keywords_str)
        else:
            return BASE_INSTRUCTION
    
    def release_phi4(self):
        del self.model,self.processor # 删除所有相关变量引用

        # 2. 强制触发垃圾回收
        gc.collect()

        # 3. 清空PyTorch的CUDA缓存
        if torch.cuda.is_available():
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(f"cuda:{gpu_id}"):
                    torch.cuda.empty_cache()

    def process_batch(self, batch_data):
        """批量处理音频数据"""
        results = []
        
        # 临时存储批量处理的输入
        batch_prompts = []
        batch_audios = []
        batch_indices = []
        
        for idx, (audio, samplerate, intent) in enumerate(batch_data):
            # 根据意图构建指令
            instruction = self.build_instruction_with_keywords(intent)
            
            # 构建用户消息
            user_message = {
                'role': 'user',
                'content': '<|audio_1|>\n' + instruction,
            }
            
            # 应用聊天模板
            prompt = self.processor.tokenizer.apply_chat_template(
                [user_message], tokenize=False, add_generation_prompt=True
            )
            
            batch_prompts.append(prompt)
            batch_audios.append((audio, samplerate))
            batch_indices.append(idx)
        
        # 批量处理
        with autocast(enabled=True):
            # 分批处理可能较大的批次
            inputs = self.processor(text=batch_prompts, audios=batch_audios, return_tensors='pt').to(self.main_device)
            
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                generation_config=self.generation_config,
            )
            
            # 提取生成的文本部分
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            
            # 解码生成的ID
            responses = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # 为每个索引分配相应的响应
            for i, response in enumerate(responses):
                # print(f'>>> Response {i+1}/{len(responses)}\n{response}')
                results.append(response)
        
        # 清理缓存
        del inputs, generate_ids
        if torch.cuda.is_available():
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(f"cuda:{gpu_id}"):
                    torch.cuda.empty_cache()
        
        return results
        
    def process_audio(self, audio, samplerate, intent):
        """
        处理单个音频，使用与微调代码一致的Prompt格式
        
        Args:
            audio: 音频数据
            samplerate: 采样率
            intent: 意图信息，用于选择相应的关键词
        """
        results = self.process_batch([(audio, samplerate, intent)])
        return results[0]

if __name__ == "__main__":
    
    
    # 输出关键词使用情况
    print(f"使用关键词: {USE_KEYWORDS}")
    if USE_KEYWORDS:
        if NUM_KEYWORDS > 0:
            print(f"随机选择关键词: 每个样本选择 {NUM_KEYWORDS} 个关键词")
        else:
            print(f"使用所有可用的关键词")
    
    # 解析GPU ID列表
    gpu_ids = [int(gpu_id.strip()) for gpu_id in DEFAULT_GPUS.split(',')]
    
    # 使用指定的GPU
    phi4 = Phi4(gpu_ids=gpu_ids, batch_size=DEFAULT_BATCH_SIZE)
    phi4.load_phi4()
    
    # 加载关键词
    phi4.load_keywords(KEYWORDS_DIR)
    
    # 读取数据文件
    df = pd.read_csv(input_data_path)
    
    # 添加asr列
    df['asr'] = ''
    
    # 准备批处理数据
    batch_data = []
    batch_indices = []
    
    # 使用tqdm创建进度条
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频"):
        # 读取音频文件
        audio, samplerate = sf.read(row['path'])
        # 读取预测的intents
        intents = row['predict']
        
        # 添加到批处理队列
        batch_data.append((audio, samplerate, intents))
        batch_indices.append(idx)
        
        # 当达到批处理大小或处理到最后一条数据时执行批处理
        if len(batch_data) >= phi4.batch_size or idx == len(df) - 1:
            # 批量处理音频
            responses = phi4.process_batch(batch_data)
            
            # 处理结果并保存
            for i, response in enumerate(responses):
                # 删除标点符号
                response = re.sub(r'[^\w\s]', '', response)
                
                # 将数字转换为中文字
                def num_to_chinese(matched):
                    num = matched.group(0)
                    chinese_nums = {'0':'零','1':'一','2':'二','3':'三','4':'四',
                                  '5':'五','6':'六','7':'七','8':'八','9':'九',
                                  '10':'十','11':'十一','12':'十二','13':'十三',
                                  '14':'十四','15':'十五','16':'十六','17':'十七',
                                  '18':'十八','19':'十九','20':'二十'}
                    return chinese_nums.get(num, num)
                
                response = re.sub(r'\d+', num_to_chinese, response)
                
                # 保存预测结果
                df.at[batch_indices[i], 'asr'] = response
            
            # 清空批处理队列
            batch_data = []
            batch_indices = []
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    # 保存结果到csv
    df.to_csv(output_data_path, index=False)
    
    # 释放资源
    phi4.release_phi4()
    
    # 调用评分函数进行ASR结果评估
    print("\n开始评估ASR结果...")

    # 从保存的结果文件中读取数据进行评估
    results_df = pd.read_csv(output_data_path)
    # 构建输出JSON文件路径，将CSV文件扩展名改为JSON
    output_json_path = output_data_path.replace('.csv', '.json')
    evaluation_results = evaluate_asr(results_df, cal_keyword_wer=True, print_errors=True, output_file=output_json_path)
    
    