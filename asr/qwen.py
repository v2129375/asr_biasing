"""
Qwen2-Audio 批量 ASR。
- 默认使用 eager 注意力，避免 flash_attention_2 触发的 CUDA device-side assert。
- 若需加速且环境稳定，可：USE_FLASH_ATTENTION=1 python asr/qwen.py
- 调试 assert 时：CUDA_LAUNCH_BLOCKING=1 python asr/qwen.py
"""
import os
import re
import gc
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


class QwenAudio:
    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self, use_flash_attention=False):
        """加载 Qwen2-Audio 模型和处理器。
        use_flash_attention: 若为 True 使用 flash_attention_2（更快但部分环境会触发 CUDA assert），
                             默认 False 使用 eager 以规避 device-side assert。
        """
        model_path = "Qwen/Qwen2-Audio-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_path)
        attn = "flash_attention_2" if use_flash_attention else "eager"
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        )

    def release_model(self):
        """释放模型和显存"""
        del self.model, self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_audio(self, audio_path, prompt="Transcribe the audio clip into text."):
        """
        对单个音频文件进行识别。
        :param audio_path: 音频文件路径
        :param prompt: 提示文本，默认为转写任务
        :return: 识别结果文本
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # 使用处理器要求的采样率，并显式传入以避免静默错误
        sampling_rate = getattr(
            self.processor.feature_extractor,
            "sampling_rate",
            16000,
        )

        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_data, _ = librosa.load(
                            ele["audio_url"],
                            sr=sampling_rate,
                        )
                        if audio_data.size == 0:
                            return ""
                        audios.append(audio_data)

        inputs = self.processor(
            text=text,
            audios=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=sampling_rate,
        )
        inputs = inputs.to(self.model.device)

        # 使用 max_new_tokens 并设置 pad_token_id，避免 CUDA device-side assert
        pad_token_id = getattr(
            self.processor.tokenizer,
            "pad_token_id",
            self.processor.tokenizer.eos_token_id,
        )
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=pad_token_id,
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response


if __name__ == "__main__":
    # 设定输入 CSV 和输出 CSV 路径
    input_data_path = "data/catslu/test.csv"
    output_data_path = "asr/exp/qwen_audio_asr_result.csv"

    prompt = "Transcribe the audio clip into text."

    qwen = QwenAudio()
    # 默认用 eager 规避 CUDA assert；需要加速时可设环境变量 USE_FLASH_ATTENTION=1
    qwen.load_model(use_flash_attention=os.environ.get("USE_FLASH_ATTENTION", "") == "1")

    # 读取数据文件（CSV 需包含 path 列，表示音频文件路径）
    df = pd.read_csv(input_data_path)
    df["asr"] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio_path = row["path"]
            response = qwen.process_audio(audio_path, prompt)

            # 数字转中文
            def num_to_chinese(matched):
                num = matched.group(0)
                chinese_nums = {
                    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
                    "10": "十", "11": "十一", "12": "十二", "13": "十三",
                    "14": "十四", "15": "十五", "16": "十六", "17": "十七",
                    "18": "十八", "19": "十九", "20": "二十",
                }
                return chinese_nums.get(num, num)

            response = re.sub(r"\d+", num_to_chinese, response)
            # 只保留中文（去掉英文、标点、空格）
            response = re.sub(r"[^\u4e00-\u9fff]", "", response)
            df.at[idx, "asr"] = response

        except Exception as e:
            print(f"处理文件 {row['path']} 时出错: {str(e)}")
            continue

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_csv(output_data_path, index=False)

    qwen.release_model()
