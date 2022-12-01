from pathlib import Path
import pandas
from tqdm import tqdm

def isChinese(word): # 检测str是否为全中文
    for ch in word:
        if not '\u4e00' <= ch <= '\u9fff':
            return False
    return True

local_path = "/Data/dataset/ASR_dataset/AISHELL-2"
dir_path = "data/aishell2"
data_path = f"{dir_path}/data.csv"
drop_english = True

data_dir = f"{local_path}/iOS/data"
transcript_path = f"{data_dir}/trans.txt"
wav_scp_path = f"{data_dir}/wav.scp"


df_path = pandas.read_table(wav_scp_path,sep="\t",header=None,names=["id","path"])
df_text = pandas.read_table(transcript_path,sep="\t",header=None,names=["id","text"])
df = pandas.merge(df_path,df_text,on="id")
for i in tqdm(range(len(df))): # 修改路径为绝对路径
    df.loc[i,"path"] = data_dir + "/" + df.loc[i,"path"]

if drop_english:
    drop_index = []
    for i in tqdm(range(len(df))): # 删除非中文的句子
        if not isChinese(df["text"][i]):
            drop_index.append(i)
    print(df.loc[drop_index,"text"])
    df.drop(index=drop_index,inplace=True)

# df.drop(columns=["id"],inplace=True)
print(df)
Path.mkdir(Path(dir_path),exist_ok=True)
df.to_csv(data_path,index=False)