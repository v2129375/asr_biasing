import json
import jieba
from tqdm import tqdm
import pandas
import re
import sys
from pathlib import Path


csv_file = "data/aishell2/data.csv"
asr_text = "data/aishell2/text"
asp_data_dir = "data/asp_data"
asp_csv_file = f"{asp_data_dir}/data_aishell2.csv"
Path.mkdir(Path(asp_data_dir),exist_ok=True)

def find_chinese(file): # 用来去除非中文字符
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

class Node:
    value = 0
    action = ""
    last = ""
    def __init__(self,v) -> None:
        self.value = v

def min_distance(word1, word2):
 
    row = len(word1) + 1
    column = len(word2) + 1
    cache = []
    for i in range(row):
        cache.append([])
        for j in range(column):
            cache[i].append(Node(0))


    for i in range(row):
        for j in range(column):
 
            if i ==0 and j ==0:
                cache[i][j].value = 0
            elif i == 0 and j!=0:
                cache[i][j].value = j
            elif j == 0 and i!=0:
                cache[i][j].value = i
            else:
                if word1[i-1] == word2[j-1]:
                    cache[i][j].action = "correct"
                    cache[i][j].value = cache[i-1][j-1].value
                    cache[i][j].last = cache[i-1][j-1]
                else:
                    replace = cache[i-1][j-1].value + 1
                    insert = cache[i][j-1].value + 1
                    remove = cache[i-1][j].value + 1

                    this_min = sys.maxsize
                    this_action = ""
                    dic = {"replace":replace,"insert":insert,"remove":remove}
                    for k in dic.keys():
                        if dic[k] < this_min:
                            this_min = dic[k] 
                            this_action = k
                    if this_action == "replace":
                        cache[i][j].last = cache[i-1][j-1]
                    elif this_action == "insert":
                        cache[i][j].last = cache[i][j-1]
                    elif this_action == "remove":
                        cache[i][j].last = cache[i-1][j]

                    cache[i][j].action = this_action
                    cache[i][j].value = min(replace, insert, remove)
    # 回溯看看做了什么action
    action = []
    a = cache[row-1][column-1]
    while a.action != "":
        action.insert(0,a.action)
        a = a.last
    # print(action)
    wrong_index = []

    for i in range(len(action)):
        if action[i] == "insert":
            word1.insert(i," ")
        elif action[i] == "remove":
            word2.insert(i," ")
        elif action[i] == "replace":
            wrong_index.append(i)

    return word1,word2,wrong_index    


df = pandas.read_csv(csv_file)
print(df)
asr_df = pandas.read_table(asr_text,sep=" ",header=None,names=["id","asr"])
print(asr_df)
df = pandas.merge(df,asr_df,on="id")
print(df)
asp_df = pandas.DataFrame(columns=["text","wrong"])
new_data = []
for i in tqdm(range(len(df))):
    asr = str(df["asr"][i])
    ground = df["text"][i]
    asr = list(jieba.cut(asr))
    ground = list(jieba.cut(ground))
    asr,ground,index = min_distance(asr,ground)
    if len(index) > 0:
        # print(asr)
        # print(ground)
        for j in index:
            wrong = asr[j]
            text = ground[j]
            # print({"wrong":wrong,"text":text})
            new_data.append({"wrong":wrong,"text":text})
asp_df = asp_df.append(new_data,ignore_index=True)    
asp_df.to_csv(asp_csv_file,index=False)        
            
    
