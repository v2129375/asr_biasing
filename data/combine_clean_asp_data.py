# 用来整合并清理训练ASP模型data
import pandas
from tqdm import tqdm
data_dir = "data/asp_data"
csv_file = ["data_aishell_train.csv","data_aishell2.csv"]

df = pandas.DataFrame(columns=["text","wrong"])
for i in csv_file:
    this_df = pandas.read_csv(f"{data_dir}/{i}")
    df = df.append(this_df,ignore_index=True)

# 设定清理资料的规则
drop_index = []
for i in tqdm(range(len(df))):
    if len(str(df["text"][i])) != len(str(df["wrong"][i])): # 删除字数不相同的
        drop_index.append(i)
df.drop(drop_index,inplace=True)

# 分割资料集
df_val = df.sample(frac=0.1)
df_train = df[~df.index.isin(df_val.index)]   

df.to_csv(f"{data_dir}/data.csv",index=False)
df_val.to_csv(f"{data_dir}/val.csv",index=False)
df_train.to_csv(f"{data_dir}/train.csv",index=False)

