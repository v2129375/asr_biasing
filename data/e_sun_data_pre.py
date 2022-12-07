import pandas
from tqdm import tqdm
json_path = "data/e_sun/train_all.json"
dev_set_ratio = 0.1
train_csv = "data/e_sun/train.csv"
dev_csv = "data/e_sun/dev.csv"

df = pandas.read_json(json_path)
df = df.drop(columns=["phoneme_sequence_list","id"])
df = df.rename(columns={"sentence_list":"asr","ground_truth_sentence":"text"})

dev_df = df.sample(frac=dev_set_ratio,random_state=0,axis=0)
train_df = df[~df.index.isin(dev_df.index)]

# 重置index
dev_df=dev_df.reset_index().drop(columns=["index"]) 
train_df=train_df.reset_index().drop(columns=["index"])

# 确定val set的原始nbest1语句和ground truth
dev_df = dev_df.rename(columns={"text":"manual_transcript"})
for i in range(len(dev_df)):
    dev_df["asr"][i] = dev_df["asr"][i][0].replace(" ","")

# 将训练集的语句从nbest1-10拆分成多笔ASP训练资料
text_list = []
asr_list = []
asr_word_list = []
for i in tqdm(range(len(train_df))):
    text = train_df["text"][i]
    asr_all = train_df["asr"][i]
    for j in asr_all:
        asr_word_list.append(j)
        asr = j.replace(" ","")
        text_list.append(text)
        asr_list.append(asr)

train_df = pandas.DataFrame()
train_df["asr"] = asr_list
train_df["text"] = text_list
train_df["asr_word"] = asr_word_list


train_df.to_csv(train_csv,index=False)
dev_df.to_csv(dev_csv,index=False)


