
import pandas as pd


input_data_path = "intent/exp/phi4_intent_result.csv"
# 读取CSV文件
df = pd.read_csv(input_data_path)

# 计算每个类别的正确率
categories = ['video', 'music', 'city']
for category in categories:
    # 获取该类别的样本总数
    total = len(df[df['source'] == category])
    # 获取预测正确的样本数
    correct = len(df[(df['source'] == category) & (df['source'] == df['predict'])])
    # 计算正确率
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"{category}类别的正确率: {accuracy:.2f}%")

# 计算总体正确率
total_samples = len(df)
total_correct = len(df[df['source'] == df['predict']])
total_accuracy = total_correct / total_samples * 100

print(f"\n总体正确率: {total_accuracy:.2f}%")
