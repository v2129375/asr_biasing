import requests
import torch
import os
import io
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen
import gc
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import re

class Phi4:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.generation_config = None
    def load_phi4(self):

        # Define model path
        model_path = "microsoft/Phi-4-multimodal-instruct"

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
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
    
    def release_phi4(self):
        del self.model,self.processor # 删除所有相关变量引用

        # 2. 强制触发垃圾回收
        gc.collect()

        # 3. 清空PyTorch的CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_audio(self,speech_prompt,audio,samplerate,keywords,intents):
        keywords_str = ' '.join(keywords)
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        


        prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
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
    prompt = """Transcribe the audio clip into text."""

    input_data_path = "intent/exp/phi4_intent_result.csv"
    output_data_path = "asr/exp/phi4_keywords_asr_result.csv"
    video_keywords_path = "data/catslu/keyword_video.txt"
    music_keywords_path = "data/catslu/keyword_music.txt"
    city_keywords_path = "data/catslu/keyword_city.txt"

    # 读取数据文件
    df = pd.read_csv(input_data_path)
    video_keywords = open(video_keywords_path, 'r').read().splitlines()
    music_keywords = open(music_keywords_path, 'r').read().splitlines()
    city_keywords = open(city_keywords_path, 'r').read().splitlines()

    # 添加asr列
    df['asr'] = ''

    # 使用tqdm遍历处理每条数据
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 读取音频文件
        audio, samplerate = sf.read(row['path'])
        # 读取预测的intents
        intents = row['predict']
        if 'video' in intents:
            keywords = video_keywords
        elif 'music' in intents:
            keywords = music_keywords
        elif 'city' in intents:
            keywords = city_keywords
        # 获取预测结果
        response = phi4.process_audio(prompt, audio, samplerate, keywords, intents)
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