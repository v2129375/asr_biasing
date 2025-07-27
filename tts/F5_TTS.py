#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F5-TTS 语音合成批处理程序
读取TXT文件中的文本或CSV文件中的文本，使用F5-TTS进行语音合成，并生成包含音频路径的新CSV文件
支持从CSV文件中读取参考音频列表
"""

import os
import pandas as pd
import subprocess
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import random

# ===================== 配置参数 =====================
# 输入文件路径 - 支持TXT或CSV格式
INPUT_FILE_PATH = "tts/tts_data/sentences_audio.csv"  # 可以是txt或csv文件

# 音频输出目录 - 可在此处修改输出目录
OUTPUT_AUDIO_DIR = "/home/v2129375/dataset/tts_audio_diff_ref"

# 输出CSV文件路径
OUTPUT_CSV_PATH = "tts/tts_data/sentences_audio_diff_ref.csv"

# F5-TTS模型配置
F5TTS_MODEL = "F5TTS_v1_Base"

# 参考音频文件路径（可以是单个音频文件或包含音频路径的CSV文件）
REF_AUDIO_PATH = "data/catslu/test.csv"  # CSV文件，包含path列

# 参考音频的文本内容（当使用单个参考音频时）
REF_TEXT = ""  # 请修改为参考音频的实际文本内容

# ===================== 主要功能函数 =====================

def check_dependencies():
    """检查F5-TTS是否已安装"""
    try:
        result = subprocess.run(['f5-tts_infer-cli', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("错误：F5-TTS未正确安装或无法访问f5-tts_infer-cli命令")
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("错误：找不到f5-tts_infer-cli命令，请确保F5-TTS已正确安装")
        return False

def create_output_directory(output_dir):
    """创建输出目录"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"输出目录已创建或已存在: {output_dir}")

def load_reference_audios(ref_audio_path):
    """
    从CSV文件中加载参考音频列表
    
    Args:
        ref_audio_path: CSV文件路径，包含path列
    
    Returns:
        list: 参考音频文件路径列表
    """
    try:
        if ref_audio_path.endswith('.csv'):
            # 从CSV文件读取参考音频
            df = pd.read_csv(ref_audio_path)
            if 'path' not in df.columns:
                print("错误：CSV文件中未找到'path'列")
                return []
            
            # 过滤掉空路径
            audio_paths = df['path'].dropna().tolist()
            # 过滤掉不存在的文件
            valid_paths = [path for path in audio_paths if os.path.exists(path)]
            
            if not valid_paths:
                print("错误：CSV文件中没有找到有效的音频文件路径")
                return []
            
            print(f"成功加载 {len(valid_paths)} 个参考音频文件")
            return valid_paths
        else:
            # 单个音频文件
            if os.path.exists(ref_audio_path):
                return [ref_audio_path]
            else:
                print(f"错误：参考音频文件不存在: {ref_audio_path}")
                return []
    except Exception as e:
        print(f"加载参考音频文件失败: {str(e)}")
        return []

def load_input_texts(input_file_path):
    """
    从TXT或CSV文件中加载输入文本和source信息
    
    Args:
        input_file_path: 输入文件路径（TXT或CSV）
    
    Returns:
        list: 包含字典的列表，每个字典包含'text'和'source'键
    """
    try:
        if input_file_path.endswith('.txt'):
            # 从TXT文件读取文本
            with open(input_file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            # TXT文件默认source为'txt'
            data = [{'text': text, 'source': 'txt'} for text in texts]
            print(f"成功从TXT文件加载 {len(data)} 行文本")
            return data
        elif input_file_path.endswith('.csv'):
            # 从CSV文件读取文本
            df = pd.read_csv(input_file_path)
            if 'manual_transcript' not in df.columns:
                print("错误：CSV文件中未找到'manual_transcript'列")
                print("可用的列:", list(df.columns))
                return []
            
            # 检查是否有source列，如果没有则使用默认值
            if 'source' not in df.columns:
                print("警告：CSV文件中未找到'source'列，将使用默认值'csv'")
                df['source'] = 'csv'
            
            # 过滤掉manual_transcript为空的行
            df = df.dropna(subset=['manual_transcript'])
            
            # 创建数据列表
            data = []
            for _, row in df.iterrows():
                data.append({
                    'text': str(row['manual_transcript']),
                    'source': str(row['source']) if pd.notna(row['source']) else 'csv'
                })
            
            print(f"成功从CSV文件加载 {len(data)} 行文本")
            return data
        else:
            print("错误：不支持的文件格式，请使用.txt或.csv文件")
            return []
    except Exception as e:
        print(f"加载输入文件失败: {str(e)}")
        return []

def generate_audio(text, output_path, ref_audio, ref_text, model):
    """
    使用F5-TTS生成音频
    
    Args:
        text: 要合成的文本
        output_path: 输出音频文件路径
        ref_audio: 参考音频路径
        ref_text: 参考音频的文本
        model: F5-TTS模型名称
    
    Returns:
        bool: 是否成功生成音频
    """
    try:
        cmd = [
            'f5-tts_infer-cli',
            '--model', model,
            '--ref_audio', ref_audio,
            '--ref_text', ref_text,
            '--gen_text', text,
            '-w', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        else:
            print(f"音频生成失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"音频生成超时: {text[:50]}...")
        return False
    except Exception as e:
        print(f"音频生成出错: {str(e)}")
        return False

def sanitize_filename(text, max_length=50):
    """
    清理文本以生成合法的文件名
    
    Args:
        text: 原始文本
        max_length: 文件名最大长度
    
    Returns:
        str: 清理后的文件名
    """
    # 移除或替换非法字符
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        text = text.replace(char, '_')
    
    # 限制长度
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()

def process_texts(input_file_path, output_audio_dir, output_csv_path, 
                 ref_audio_path, ref_text, model):
    """
    处理文本文件，生成音频并创建新的CSV文件
    
    Args:
        input_file_path: 输入文件路径（TXT或CSV）
        output_audio_dir: 音频输出目录
        output_csv_path: 输出CSV文件路径
        ref_audio_path: 参考音频文件路径（单个文件或CSV文件）
        ref_text: 参考音频文本（当使用单个参考音频时）
        model: F5-TTS模型
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误：输入文件不存在: {input_file_path}")
        return False
    
    # 加载输入文本和source信息
    text_data = load_input_texts(input_file_path)
    if not text_data:
        return False
    
    # 加载参考音频
    ref_audios = load_reference_audios(ref_audio_path)
    if not ref_audios:
        return False
    
    # 创建输出目录
    create_output_directory(output_audio_dir)
    
    # 准备输出数据
    output_data = []
    successful_count = 0
    failed_count = 0
    
    # 遍历每一行，生成音频
    for idx, item in tqdm(enumerate(text_data), total=len(text_data), desc="生成音频"):
        text = item['text']
        source = item['source']
        
        # 跳过空文本
        if not text or text.strip() == '':
            print(f"跳过空文本，行 {idx}")
            output_data.append({
                'manual_transcript': text,
                'path': '',
                'source': source
            })
            failed_count += 1
            continue
        
        # 生成音频文件名
        audio_filename = f"audio_{idx:04d}.wav"
        audio_path = os.path.join(output_audio_dir, audio_filename)
        
        # 随机选择一个参考音频
        ref_audio = random.choice(ref_audios)
        
        # 生成音频
        print(f"正在生成音频 {idx+1}/{len(text_data)}: {text[:50]}...")
        
        if generate_audio(text, audio_path, ref_audio, ref_text, model):
            output_data.append({
                'manual_transcript': text,
                'path': audio_path,
                'source': source
            })
            successful_count += 1
            print(f"成功生成: {audio_filename}")
        else:
            output_data.append({
                'manual_transcript': text,
                'path': '',
                'source': source
            })
            failed_count += 1
            print(f"生成失败，行 {idx}")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(output_data)
    
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"成功保存输出CSV文件: {output_csv_path}")
    except Exception as e:
        print(f"保存CSV文件失败: {str(e)}")
        return False
    
    # 输出统计信息
    print(f"\n=== 处理完成 ===")
    print(f"成功生成音频: {successful_count} 个")
    print(f"生成失败: {failed_count} 个")
    print(f"总计处理: {len(text_data)} 行")
    print(f"输出CSV文件: {output_csv_path}")
    print(f"音频文件目录: {output_audio_dir}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='F5-TTS 批量语音合成程序')
    parser.add_argument('--input_file', type=str, default=INPUT_FILE_PATH,
                       help='输入文件路径（TXT或CSV格式）')
    parser.add_argument('--output_audio_dir', type=str, default=OUTPUT_AUDIO_DIR,
                       help='音频输出目录')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV_PATH,
                       help='输出CSV文件路径')
    parser.add_argument('--ref_audio', type=str, default=REF_AUDIO_PATH,
                       help='参考音频文件路径（单个文件或包含path列的CSV文件）')
    parser.add_argument('--ref_text', type=str, default=REF_TEXT,
                       help='参考音频的文本内容（当使用单个参考音频时）')
    parser.add_argument('--model', type=str, default=F5TTS_MODEL,
                       help='F5-TTS模型名称')
    
    args = parser.parse_args()
    
    print("=== F5-TTS 批量语音合成程序 ===")
    print(f"输入文件: {args.input_file}")
    print(f"音频输出目录: {args.output_audio_dir}")
    print(f"输出CSV文件: {args.output_csv}")
    print(f"参考音频: {args.ref_audio}")
    print(f"参考文本: {args.ref_text}")
    print(f"使用模型: {args.model}")
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 处理文本文件
    success = process_texts(
        args.input_file,
        args.output_audio_dir, 
        args.output_csv,
        args.ref_audio,
        args.ref_text,
        args.model
    )
    
    if success:
        print("程序执行完成！")
    else:
        print("程序执行失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
