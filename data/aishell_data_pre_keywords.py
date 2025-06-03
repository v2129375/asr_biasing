import os
import random

import pandas
from tqdm import tqdm
import re
from opencc import OpenCC
from pathlib import Path

# 全局变量设置
local_path = "/home/v2129375/dataset/aishell1/data_aishell"
dir_path = "data/aishell_keywords"
train_path = f"{dir_path}/train.csv"
train_samples = 1000  # 设置训练集语句数量，如果为None则使用全部数据
random_seed = 42  # 随机种子，保证可重复性
s2t = False # 简体转繁体
transcript_path = f"{local_path}/transcript/aishell_transcript_v0.8.txt"

# 讀取train和dev檔案路徑
train_files_path = []
dev_files_path = []
test_files_path = []


def recur_train(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            train_files_path.append(os.path.join(root, file))
        for dir in dirs:
            recur_train(dir)


def recur_dev(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            dev_files_path.append(os.path.join(root, file))
        for dir in dirs:
            recur_dev(dir)


def recur_test(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            test_files_path.append(os.path.join(root, file))
        for dir in dirs:
            recur_test(dir)


recur_train(f"{local_path}/wav/train")
recur_dev(f"{local_path}/wav/dev")
recur_test(f"{local_path}/wav/test")

# 讀取文本檔案轉成字典
cc = OpenCC('s2t') # 簡體轉繁體
_d = {}
with open(transcript_path, encoding='utf-8') as f:
    data = f.readlines()
    for i in tqdm(data):
        k, v = re.split('\s+', i, 1)
        if s2t:
            _d[k.strip()] = cc.convert(v.replace('\n', '').replace('\t', '').replace(' ', ''))
        else:
            _d[k.strip()] = v.replace('\n', '').replace('\t', '').replace(' ', '')
# print(_d)

# 整合資訊並輸出成csv檔案
res_train = []
for file in tqdm(train_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_train.append((file, _d[file_name]))
res_dev = []
for file in tqdm(dev_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_dev.append((file, _d[file_name]))
res_test = []
for file in tqdm(test_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_test.append((file, _d[file_name]))

# 随机采样指定数量的训练数据
random.seed(random_seed)
if train_samples is not None and train_samples < len(res_train):
    res_train = random.sample(res_train, train_samples)
    print(f"随机采样 {train_samples} 条训练数据（总共 {len(res_train)} 条）")
else:
    print(f"使用全部 {len(res_train)} 条训练数据")

# 创建输出目录并保存训练集
Path.mkdir(Path(dir_path), exist_ok=True)
pandas.DataFrame(res_train, columns=["path", "manual_transcript"]).to_csv(train_path, index=False)
print(f"训练集已保存至 {train_path}")