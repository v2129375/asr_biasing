import requests
import torch
import os
import io
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from urllib.request import urlopen
import gc
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import re

# 答案后缀标记，用于标识生成结束
ANSWER_SUFFIX = "<|end|><|endoftext|>"

class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """
    停止条件类，能够处理多个停止标记和批处理输入。
    """
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx)

class Phi4:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.generation_config = None
    def load_phi4(self):

        # Define model path - 使用本地微调模型
        model_path = "asr/model/"
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

    def process_audio(self, speech_prompt, audio, samplerate):
        # 构建用户消息，使用与finetune评估相同的格式
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + speech_prompt,
        }
        
        # 应用聊天模板，构建提示
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        
        # Process with the model
        inputs = self.processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
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
    prompt = "Transcribe the audio clip into text."

    input_data_path = "data/catslu/test.csv"
    output_data_path = "asr/exp/phi4_finetuned_asr_result.csv"

    # 读取数据文件
    df = pd.read_csv(input_data_path)

    # 添加predict列
    df['asr'] = ''

    # 使用tqdm遍历处理每条数据
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 读取音频文件
            audio, samplerate = sf.read(row['path'])
            # 获取预测结果
            response = phi4.process_audio(prompt, audio, samplerate)
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
        except Exception as e:
            print(f"处理文件 {row['path']} 时出错: {str(e)}")
            continue
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    # 保存结果到csv
    df.to_csv(output_data_path, index=False)


    phi4.release_phi4()