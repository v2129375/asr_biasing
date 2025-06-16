#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键词随机采样程序
从多个关键词文件中随机选择指定比例的关键词并输出到新文件
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Set

# ============ 配置参数 ============
# 关键词文件路径列表
KEYWORD_FILES = [
    "keywords1.txt",
    "keywords2.txt", 
    "keywords3.txt"
]

# 随机截取比例 (0.0 - 1.0)
SAMPLING_RATIO = 0.3

# 输出文件夹
OUTPUT_DIR = "output"

# 输出文件名前缀
OUTPUT_PREFIX = "sampled_"

# ====================================


def read_keywords_from_file(file_path: str) -> Set[str]:
    """
    从文件中读取关键词
    
    Args:
        file_path: 关键词文件路径
        
    Returns:
        关键词集合
    """
    keywords = set()
    
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过")
        return keywords
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                keyword = line.strip()
                if keyword:  # 忽略空行
                    keywords.add(keyword)
        print(f"从 {file_path} 读取了 {len(keywords)} 个关键词")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return keywords


def read_all_keywords(file_paths: List[str]) -> Set[str]:
    """
    从多个文件中读取所有关键词
    
    Args:
        file_paths: 关键词文件路径列表
        
    Returns:
        所有关键词的集合
    """
    all_keywords = set()
    
    for file_path in file_paths:
        keywords = read_keywords_from_file(file_path)
        all_keywords.update(keywords)
    
    print(f"总共读取了 {len(all_keywords)} 个唯一关键词")
    return all_keywords


def sample_keywords(keywords: Set[str], ratio: float) -> List[str]:
    """
    随机采样指定比例的关键词
    
    Args:
        keywords: 关键词集合
        ratio: 采样比例 (0.0 - 1.0)
        
    Returns:
        采样后的关键词列表
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("采样比例必须在 0.0 到 1.0 之间")
    
    keyword_list = list(keywords)
    sample_size = int(len(keyword_list) * ratio)
    
    if sample_size == 0:
        print("警告: 采样数量为0，将至少采样1个关键词")
        sample_size = min(1, len(keyword_list))
    
    sampled = random.sample(keyword_list, sample_size)
    print(f"从 {len(keyword_list)} 个关键词中采样了 {len(sampled)} 个")
    
    return sampled


def save_keywords(keywords: List[str], output_path: str) -> None:
    """
    保存关键词到文件
    
    Args:
        keywords: 关键词列表
        output_path: 输出文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for keyword in keywords:
                f.write(keyword + '\n')
        
        print(f"关键词已保存到: {output_path}")
    except Exception as e:
        print(f"保存文件 {output_path} 时出错: {e}")


def generate_output_path(input_file: str, output_dir: str, prefix: str) -> str:
    """
    生成输出文件路径
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        prefix: 文件名前缀
        
    Returns:
        输出文件路径
    """
    file_name = os.path.basename(input_file)
    output_file = prefix + file_name
    return os.path.join(output_dir, output_file)


def process_keywords(file_paths: List[str], sampling_ratio: float, 
                    output_dir: str, output_prefix: str) -> None:
    """
    处理关键词：读取、采样、输出
    
    Args:
        file_paths: 输入文件路径列表
        sampling_ratio: 采样比例
        output_dir: 输出目录
        output_prefix: 输出文件前缀
    """
    print("=" * 50)
    print("关键词随机采样程序")
    print("=" * 50)
    
    # 读取所有关键词
    all_keywords = read_all_keywords(file_paths)
    
    if not all_keywords:
        print("没有读取到任何关键词，程序退出")
        return
    
    # 随机采样
    sampled_keywords = sample_keywords(all_keywords, sampling_ratio)
    
    # 为每个输入文件生成对应的输出文件
    if len(file_paths) == 1:
        # 只有一个输入文件时，直接输出
        output_path = generate_output_path(file_paths[0], output_dir, output_prefix)
        save_keywords(sampled_keywords, output_path)
    else:
        # 多个输入文件时，创建一个合并的输出文件
        output_file = f"{output_prefix}merged_keywords.txt"
        output_path = os.path.join(output_dir, output_file)
        save_keywords(sampled_keywords, output_path)
    
    print("=" * 50)
    print("处理完成！")
    print("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='关键词随机采样程序')
    parser.add_argument('--files', nargs='+', default=KEYWORD_FILES,
                       help='关键词文件路径列表')
    parser.add_argument('--ratio', type=float, default=SAMPLING_RATIO,
                       help='采样比例 (0.0-1.0)')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                       help='输出目录')
    parser.add_argument('--prefix', default=OUTPUT_PREFIX,
                       help='输出文件名前缀')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子，用于结果复现')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    # 验证参数
    if not 0.0 <= args.ratio <= 1.0:
        print("错误: 采样比例必须在 0.0 到 1.0 之间")
        return
    
    if not args.files:
        print("错误: 必须指定至少一个关键词文件")
        return
    
    # 处理关键词
    process_keywords(args.files, args.ratio, args.output_dir, args.prefix)


if __name__ == "__main__":
    # 可以直接修改配置参数运行，或使用命令行参数
    if len(os.sys.argv) == 1:
        # 没有命令行参数时，使用配置的默认值
        print("使用默认配置运行程序...")
        process_keywords(KEYWORD_FILES, SAMPLING_RATIO, OUTPUT_DIR, OUTPUT_PREFIX)
    else:
        # 有命令行参数时，解析并运行
        main()
