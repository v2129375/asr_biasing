import os
import pandas

input_data_path = ["data/video","data/city","data/music"]
output_data_path = "data/catslu"

# 创建输出目录
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)

# 读取并合并所有test.csv
df = pandas.DataFrame()
for path in input_data_path:
    source = path.split("/")[-1]  # 获取来源名称
    csv_path = f"{path}/test.csv"
    if os.path.exists(csv_path):
        temp_df = pandas.read_csv(csv_path)
        temp_df["source"] = source  # 添加来源列
        df = pandas.concat([df, temp_df], ignore_index=True)

# 保存合并后的csv
df.to_csv(f"{output_data_path}/data.csv", index=False)

# 复制并重命名keyword文件
for path in input_data_path:
    source = path.split("/")[-1]
    keyword_src = f"{path}/keyword.txt"
    keyword_dst = f"{output_data_path}/keyword_{source}.txt"
    if os.path.exists(keyword_src):
        with open(keyword_src, 'r', encoding='utf-8') as src, open(keyword_dst, 'w', encoding='utf-8') as dst:
            dst.write(src.read())

# 按类别分层抽样分割训练集和测试集
from sklearn.model_selection import train_test_split

# 获取每个类别的数据
video_df = df[df['source'] == 'video']
music_df = df[df['source'] == 'music'] 
city_df = df[df['source'] == 'city']

# 分别对每个类别进行分割
video_train, video_test = train_test_split(video_df, test_size=0.2, random_state=42)
music_train, music_test = train_test_split(music_df, test_size=0.2, random_state=42)
city_train, city_test = train_test_split(city_df, test_size=0.2, random_state=42)

# 合并训练集和测试集
train_df = pandas.concat([video_train, music_train, city_train], ignore_index=True)
test_df = pandas.concat([video_test, music_test, city_test], ignore_index=True)

# 保存训练集和测试集
train_df.to_csv(f"{output_data_path}/train.csv", index=False)
test_df.to_csv(f"{output_data_path}/test.csv", index=False)

