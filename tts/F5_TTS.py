#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F5-TTS 语音合成批处理程序
读取CSV文件中的文本，使用F5-TTS进行语音合成，并生成包含音频路径的新CSV文件
"""

import os
import pandas as pd
import subprocess
import sys
from pathlib import Path
import argparse
from tqdm import tqdm

# ===================== 配置参数 =====================
# CSV文件路径 - 可在此处修改输入文件路径
INPUT_CSV_PATH = "tts/tts_data/sentences.csv"

# 音频输出目录 - 可在此处修改输出目录
OUTPUT_AUDIO_DIR = "/home/v2129375/dataset/tts_audio"

# 输出CSV文件路径
OUTPUT_CSV_PATH = "tts/tts_data/sentences_audio.csv"

# F5-TTS模型配置
F5TTS_MODEL = "F5TTS_v1_Base"

# 参考音频文件路径（需要提供一个参考音频）
REF_AUDIO_PATH = "/home/v2129375/dataset/catslu_traindev/data/video/audios/1cfdff9db4788e7e8b0729b8236c6594_59cca4de3327930c51000687.wav"  # 请修改为实际的参考音频路径

# 参考音频的文本内容
REF_TEXT = "我想看白雪公主"  # 请修改为参考音频的实际文本内容

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

def process_csv(input_csv_path, output_audio_dir, output_csv_path, 
                ref_audio, ref_text, model):
    """
    处理CSV文件，生成音频并创建新的CSV文件
    
    Args:
        input_csv_path: 输入CSV文件路径
        output_audio_dir: 音频输出目录
        output_csv_path: 输出CSV文件路径
        ref_audio: 参考音频路径
        ref_text: 参考音频文本
        model: F5-TTS模型
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_csv_path):
        print(f"错误：输入CSV文件不存在: {input_csv_path}")
        return False
    
    # 检查参考音频文件是否存在
    if not os.path.exists(ref_audio):
        print(f"错误：参考音频文件不存在: {ref_audio}")
        return False
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_csv_path)
        print(f"成功读取CSV文件: {input_csv_path}")
        print(f"共有 {len(df)} 行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {str(e)}")
        return False
    
    # 检查是否存在manual_transcript列
    if 'manual_transcript' not in df.columns:
        print("错误：CSV文件中未找到'manual_transcript'列")
        print("可用的列:", list(df.columns))
        return False
    
    # 创建输出目录
    create_output_directory(output_audio_dir)
    
    # 添加path列
    audio_paths = []
    
    # 遍历每一行，生成音频
    successful_count = 0
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成音频"):
        text = str(row['manual_transcript'])
        
        # 跳过空文本
        if pd.isna(text) or text.strip() == '':
            print(f"跳过空文本，行 {idx}")
            audio_paths.append('')
            failed_count += 1
            continue
        
        # 生成音频文件名（只使用编号）
        audio_filename = f"audio_{idx:04d}.wav"
        audio_path = os.path.join(output_audio_dir, audio_filename)
        
        # 生成音频
        print(f"正在生成音频 {idx+1}/{len(df)}: {text[:50]}...")
        
        if generate_audio(text, audio_path, ref_audio, ref_text, model):
            audio_paths.append(audio_path)
            successful_count += 1
            print(f"成功生成: {audio_filename}")
        else:
            audio_paths.append('')  # 生成失败时记录空路径
            failed_count += 1
            print(f"生成失败，行 {idx}")
    
    # 添加path列到DataFrame
    df['path'] = audio_paths
    
    # 保存新的CSV文件
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
    print(f"总计处理: {len(df)} 行")
    print(f"输出CSV文件: {output_csv_path}")
    print(f"音频文件目录: {output_audio_dir}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='F5-TTS 批量语音合成程序')
    parser.add_argument('--input_csv', type=str, default=INPUT_CSV_PATH,
                       help='输入CSV文件路径')
    parser.add_argument('--output_audio_dir', type=str, default=OUTPUT_AUDIO_DIR,
                       help='音频输出目录')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV_PATH,
                       help='输出CSV文件路径')
    parser.add_argument('--ref_audio', type=str, default=REF_AUDIO_PATH,
                       help='参考音频文件路径')
    parser.add_argument('--ref_text', type=str, default=REF_TEXT,
                       help='参考音频的文本内容')
    parser.add_argument('--model', type=str, default=F5TTS_MODEL,
                       help='F5-TTS模型名称')
    
    args = parser.parse_args()
    
    print("=== F5-TTS 批量语音合成程序 ===")
    print(f"输入CSV文件: {args.input_csv}")
    print(f"音频输出目录: {args.output_audio_dir}")
    print(f"输出CSV文件: {args.output_csv}")
    print(f"参考音频: {args.ref_audio}")
    print(f"参考文本: {args.ref_text}")
    print(f"使用模型: {args.model}")
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 处理CSV文件
    success = process_csv(
        args.input_csv,
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
