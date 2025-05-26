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
# 导入评分模块
from score import evaluate_asr

# 基础任务指令（与微调代码保持一致）
BASE_INSTRUCTION = "Transcribe the audio clip into text."
# 带关键词的任务指令模板（与微调代码保持一致）
KEYWORD_INSTRUCTION_TEMPLATE = "Transcribe the audio clip into text. Pay attention to these keywords: {keywords}"
input_data_path = "intent/exp/phi4_intent_result.csv"
output_data_path = "asr/exp/phi4_keywords_asr_result_ori.csv"
model_path = "microsoft/Phi-4-multimodal-instruct"
base_model_path = "microsoft/Phi-4-multimodal-instruct"

class Phi4:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.generation_config = None
        # 关键词字典
        self.keywords_dict = {}
        
    def load_phi4(self):

        # Load processor from base model to get necessary configurations
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            # _attn_implementation='eager',
            _attn_implementation='flash_attention_2'
        )

        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(model_path)
    
    def load_keywords(self, keywords_dir="data/catslu"):
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
            
        return self.keywords_dict.get(domain, [])
    
    def build_instruction_with_keywords(self, intent):
        """根据意图构建包含关键词的指令"""
        domain_keywords = self.get_domain_keywords(intent)
        
        if domain_keywords:
            # 使用该领域的全部关键词
            keywords_str = ', '.join(domain_keywords)
            return KEYWORD_INSTRUCTION_TEMPLATE.format(keywords=keywords_str)
        else:
            return BASE_INSTRUCTION
    
    def release_phi4(self):
        del self.model,self.processor # 删除所有相关变量引用

        # 2. 强制触发垃圾回收
        gc.collect()

        # 3. 清空PyTorch的CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_audio(self, audio, samplerate, intent):
        """
        处理音频，使用与微调代码一致的Prompt格式
        
        Args:
            audio: 音频数据
            samplerate: 采样率
            intent: 意图信息，用于选择相应的关键词
        """
        # 根据意图构建指令
        instruction = self.build_instruction_with_keywords(intent)
        
        # 构建用户消息（与微调代码格式一致）
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + instruction,
        }
        
        # 应用聊天模板，构建提示
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        
        # Process with the model
        inputs = self.processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f'>>> Response\n{response}')
        return response

if __name__ == "__main__":
    phi4 = Phi4()
    phi4.load_phi4()
    
    # 加载关键词
    phi4.load_keywords("data/catslu")

    

    # 读取数据文件
    df = pd.read_csv(input_data_path)

    # 添加asr列
    df['asr'] = ''

    # 使用tqdm遍历处理每条数据
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 读取音频文件
        audio, samplerate = sf.read(row['path'])
        # 读取预测的intents
        intents = row['predict']
        
        # 获取预测结果（使用新的process_audio方法）
        response = phi4.process_audio(audio, samplerate, intents)
        
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
        df.at[idx, 'asr'] = response
        
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    # 保存结果到csv
    df.to_csv(output_data_path, index=False)
    
    phi4.release_phi4()

    # 调用评分函数进行ASR结果评估
    print("\n开始评估ASR结果...")
    try:
        # 从保存的结果文件中读取数据进行评估
        results_df = pd.read_csv(output_data_path)
        evaluation_results = evaluate_asr(results_df, cal_keyword_wer=True, print_errors=True)
        print(f"\n评估完成！")
        print(f"字符错误率 (CER): {evaluation_results['cer']:.4f}")
        if 'keyword_wer' in evaluation_results:
            print(f"关键词错误率 (Keyword WER): {evaluation_results['keyword_wer']:.4f}")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")

    
    