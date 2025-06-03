import pandas as pd
import jieba
from collections import Counter
import os

# 读取训练集
dir_path = "data/aishell_keywords"
train_path = f"{dir_path}/train.csv"
keywords_path = f"{dir_path}/keywords.txt"

# 确保文件存在
if not os.path.exists(train_path):
    print(f"训练集文件 {train_path} 不存在！")
    exit(1)

# 读取CSV文件
try:
    df = pd.read_csv(train_path)
    print(f"成功读取训练集，共 {len(df)} 条数据")
except Exception as e:
    print(f"读取训练集失败: {e}")
    exit(1)

# 分词并统计词频
all_words = []
for text in df["manual_transcript"]:
    # 使用jieba进行分词
    words = jieba.cut(text)
    all_words.extend(words)

# 统计词频
word_counts = Counter(all_words)

# 过滤掉1个字的词、词频为1的词和词频大于5的词
filtered_words = {word: count for word, count in word_counts.items() if len(word) >= 2 and 1 < count <= 5}

# 按词频排序
sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

# 打印结果
print("\n两个字及以上、词频在2-5之间的词频统计（按频率降序）:")
print("-" * 40)
print(f"{'词语':<10} {'频率':<8}")
print("-" * 40)

for word, count in sorted_words:
    print(f"{word:<10} {count:<8}")

print(f"\n共有 {len(sorted_words)} 个符合条件的词语")

# 将结果输出到keywords.txt，只输出词语不输出词频
try:
    with open(keywords_path, 'w', encoding='utf-8') as f:
        for word, _ in sorted_words:
            f.write(f"{word}\n")
    print(f"\n已将词语写入到 {keywords_path}（不含词频）")
except Exception as e:
    print(f"写入文件失败: {e}") 