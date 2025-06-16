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

class Phi4:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.generation_config = None
    def load_phi4(self):

        # Define model path - 使用本地微调模型
        model_path = "intent/model/new"
        base_model_path = "microsoft/Phi-4-multimodal-instruct"

        # Load processor from base model to get necessary configurations
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        
        # Load fine-tuned model
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

    def process_audio(self,speech_prompt,audio,samplerate):
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'

        prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
        # Process with the model
        inputs = self.processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
            num_logits_to_keep=1,  # 只保留最后一个token的logits
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print(f'>>> Response\n{response}')
        return response

if __name__ == "__main__":
    phi4 = Phi4()
    phi4.load_phi4()
    prompt = """Task:
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

    input_data_path = "data/catslu"
    output_dir = "intent/exp"
    output_filename = "phi4_intent_result.csv"

    # 读取数据文件
    df = pd.read_csv(os.path.join(input_data_path, "test.csv"))

    # 添加predict列
    df['predict'] = ''

    # 使用tqdm遍历处理每条数据
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 读取音频文件
        audio, samplerate = sf.read(row['path'])
        # 获取预测结果
        response = phi4.process_audio(prompt, audio, samplerate)
        # 使用正则表达式提取response中的video、music或city
        import re
        pattern = r'(video|music|city)'
        match = re.search(pattern, response.lower())
        if match:
            response = match.group(1)
        else:
            response = 'city'  # 如果没有匹配到任何类别,默认为city
        # 保存预测结果
        df.at[idx, 'predict'] = response
        
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 保存结果到csv
    df.to_csv(os.path.join(output_dir, output_filename), index=False)


    phi4.release_phi4()
    