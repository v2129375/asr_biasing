#!/usr/bin/env python3
import pandas as pd
import os

def copy_source_column():
    """
    将sentences_audio.csv中的source栏位复制到sentences_audio_diff_ref.csv中
    """
    
    # 定义文件路径
    source_file = "tts/tts_data/sentences_audio.csv"
    target_file = "tts/tts_data/sentences_audio_diff_ref.csv"
    
    # 检查文件是否存在
    if not os.path.exists(source_file):
        print(f"错误: 源文件 {source_file} 不存在")
        return False
        
    if not os.path.exists(target_file):
        print(f"错误: 目标文件 {target_file} 不存在")
        return False
    
    try:
        # 读取源文件和目标文件
        print("正在读取源文件...")
        source_df = pd.read_csv(source_file)
        
        print("正在读取目标文件...")
        target_df = pd.read_csv(target_file)
        
        # 检查两个文件的行数是否一致
        if len(source_df) != len(target_df):
            print(f"警告: 源文件有 {len(source_df)} 行，目标文件有 {len(target_df)} 行")
            print("将按较小的行数进行复制...")
        
        # 检查source栏位是否存在
        if 'source' not in source_df.columns:
            print("错误: 源文件中没有找到'source'栏位")
            return False
            
        if 'source' not in target_df.columns:
            print("错误: 目标文件中没有找到'source'栏位")
            return False
        
        # 复制source栏位
        min_rows = min(len(source_df), len(target_df))
        target_df.loc[:min_rows-1, 'source'] = source_df.loc[:min_rows-1, 'source']
        
        # 备份原文件
        backup_file = target_file + ".backup"
        print(f"正在备份原文件到 {backup_file}...")
        target_df_original = pd.read_csv(target_file)
        target_df_original.to_csv(backup_file, index=False)
        
        # 保存更新后的文件
        print("正在保存更新后的文件...")
        target_df.to_csv(target_file, index=False)
        
        print(f"成功! 已将 {min_rows} 行的source栏位从源文件复制到目标文件")
        print(f"原文件已备份为: {backup_file}")
        
        # 显示一些统计信息
        print("\n源文件中source栏位的值分布:")
        print(source_df['source'].value_counts())
        
        print("\n更新后目标文件中source栏位的值分布:")
        print(target_df['source'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始复制source栏位...")
    success = copy_source_column()
    
    if success:
        print("\n操作完成!")
    else:
        print("\n操作失败!")
