import json
import os
from shutil import copyfile
import pandas
import re
import warnings
warnings.simplefilter("ignore")
# 需要手动设置的内容
train_data_root = "/Data/dataset/catslu_traindev"
test_data_root = "/Data/dataset/catslu_test"
local_data_path = "video_test"
# 根据手动设置的路径自动设置的路径
test_csv_path = f"{local_data_path}/test.csv"

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def isChinese(word):
    for ch in word:
        if not '\u4e00' <= ch <= '\u9fff':
            return False
    return True

# 执行档案的准备
if not os.path.exists(local_data_path):
    os.mkdir(local_data_path)

# copyfile(f"{train_data_root}/data/video/lexicon/video_name.txt", f"{local_data_path}/keyword.txt")
keywords = open(f"{train_data_root}/data/video/lexicon/video_name.txt").readlines()
save_keywords_path = f"{local_data_path}/keyword.txt"
f = open(save_keywords_path,"w",encoding="utf-8")
for i in keywords:
    if isChinese(i.strip("\n")):
        f.write(i)
f.close()


data = json.load(open(f"{train_data_root}/data/video/train.json"))
df = pandas.DataFrame(columns=["path","manual_transcript","keyword"])
# 原先训练集中的内容
for i in data:
    for j in i["utterances"]:
        manual_transcript = find_chinese(j["manual_transcript"]) # 去除非中文的手动转录
        path = f"{train_data_root}/data/video/audios/{j['wav_id']}.wav"
        have_keyword = False
        keyword = ""
        for k in j["semantic"]:
            if k[1] == "片名":
                have_keyword = True
                keyword = k[2]
        if have_keyword:
            df = df.append({"manual_transcript":manual_transcript,"path":path,"keyword":keyword},ignore_index=True)
# 原先测试集中的内容
data = json.load(open(f"{test_data_root}/data/video/test.json"))
for i in data:
    for j in i["utterances"]:
        manual_transcript = find_chinese(j["manual_transcript"]) # 去除非中文的手动转录
        path = f"{test_data_root}/data/video/audios/{j['wav_id']}.wav"
        have_keyword = False
        keyword = ""
        for k in j["semantic"]:
            if k[1] == "片名" and len(k)==3:
                keyword = k[2]
                if not isChinese(keyword): # 不要含有英文的keyword
                    continue
                have_keyword = True
        if have_keyword:
            df = df.append({"manual_transcript":manual_transcript,"path":path,"keyword":keyword},ignore_index=True)
df.to_csv(test_csv_path,index=False)

