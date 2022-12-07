# 将我的csv格式档案转成wenet要求的格式

import pandas
from tqdm import tqdm
import json
csv_path = "data/aishell2/data.csv"
data_list_path = csv_path.rstrip(csv_path.split("/")[-1]) + "data.list"


df = pandas.read_csv(csv_path)
df.rename(columns={"id":"key","path":"wav","text":"txt"},inplace=True)

df.to_json(data_list_path,orient="records",lines=True,force_ascii=False)