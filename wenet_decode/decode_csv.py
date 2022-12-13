import json
from tqdm import tqdm

import wenetruntime as wenet
import pandas
import re
csv_file = "data/city/test.csv"
keyword_file = "data/city/keyword.txt"
model_dir = "/Data/models/20220506_u2pp_conformer_libtorch"
result_csv = "wenet_decode/exp/result_city_biasing.csv"
biasing = True

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese
if biasing:
    keywords = open(keyword_file).readlines()
    for i in range(len(keywords)):
        keywords[i] = keywords[i].strip("\n")
    decoder = wenet.Decoder(lang='chs',
                        model_dir=model_dir,
                        context=keywords)
else:
    decoder = wenet.Decoder(lang='chs',
                        model_dir=model_dir)
df = pandas.read_csv(csv_file)

df["asr"] = ""
for i in tqdm(range(len(df))):
    ans = decoder.decode_wav(df["path"][i])
    ans = json.loads(ans)
    # print(find_chinese(ans["nbest"][0]["sentence"]))
    df.loc[i,"asr"] = find_chinese(ans["nbest"][0]["sentence"])

df.to_csv(result_csv,index=False)