import pandas
from tqdm import tqdm
asp_table = "asp/exp/asp_table_jieba.csv"
result_file = "wenet_decode/exp/result_catslu_biasing.csv"
output_file = "wenet_decode/exp/result_catslu_biasing_asp.csv"
keyword_file = "data/video_test/keyword.txt"
df_asp = pandas.read_csv(asp_table)
df_result = pandas.read_csv(result_file)
keyword = open(keyword_file,"r").readlines()

# 去掉keyword末尾的換行符號
for i in range(len(keyword)):
    keyword[i] = keyword[i].strip("\n")

# 將asp table轉為字典
biasing_dic = {}
for i in range(len(df_asp)):
    text = df_asp["text"][i]
    asp = eval(df_asp["asp"][i])
    for j in asp:
        biasing_dic[j] = text

def check_keyword(asr):
    for i in keyword:
        if i in asr:
            return True
    return False

def biasing(asr,num):
    for i in range(0,len(asr)-num+1):
        word = asr[i:i+num]
        if word in biasing_dic.keys():
            asr = asr.replace(word,biasing_dic[word])
            print(f"replace{num}!")
            return asr
    return ""



for i in tqdm(range(len(df_result))):
    asr = df_result["asr"][i]
    if check_keyword(asr): # 如果已經有辨識對的關鍵詞就不執行偏移
        continue

    a = biasing(asr,4)
    if a != "":
        df_result.loc[i,"asr"] = a
        continue

    a = biasing(asr,3)
    if a != "":
        df_result.loc[i,"asr"] = a
        continue

    a = biasing(asr,2)
    if a != "":
        df_result.loc[i,"asr"] = a
        continue

df_result.to_csv(output_file,index=False)
    
    
    