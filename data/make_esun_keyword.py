import pandas
from tqdm import tqdm
data_csv = "data/e_sun/train.csv"
keyword_txt = "data/e_sun/keyword.txt"

df = pandas.read_csv(data_csv)
count_dic = {}
for i in tqdm(range(len(df))):
    word = df["asr_word"][i].split(" ")
    for j in word:
        if j in count_dic:
            count_dic[j] +=1
        else:
            count_dic[j] =1

f = open(keyword_txt,"w")
for i in count_dic:
    if 50<count_dic[i]<1000 and len(i)>=2: # 设定偏移关键词的条件，且长度要大于2，作为偏移词
        f.write(i+"\n")
f.close()
