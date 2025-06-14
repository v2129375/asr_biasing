import torch
import random
import os
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ========== 配置设置 ==========
# 不同领域的关键词文件路径
KEYWORD_FILES = {
    "video": "data/catslu/keyword_video.txt",
    "music": "data/catslu/keyword_music.txt",
    "city": "data/catslu/keyword_city.txt"
}

# 输出文件路径（CSV格式）
OUTPUT_FILE = "tts/tts_data/sentences.csv"

# 模型设置
MODEL_PATH = "microsoft/Phi-4-mini-instruct"
# ================================
 
torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 定义系统消息
system_message = """我会给你一些专有名词和领域，请根据这些专有名词和它的领域进行造句，句子是在进行询问天气、点播音乐、点播视频或者电视剧
其中video领域是点播视频或者电视剧，music领域是点播音乐，city领域是询问天气
如果是播放音乐，则造句和音乐播放相关的点播命令
如果是询问天气则造句应该和天气询问相关的疑问句
注意点播视频或者音乐应该是命令的句式不应该有疑问语气，而询问天气是疑问句
"""

# 创建pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
generation_args = {
    "max_new_tokens": 20,
    "return_full_text": False,
    "do_sample": False,
}

def read_keywords_by_domain(keyword_files):
    """读取各个领域的关键词文件并返回带领域标记的词列表"""
    keywords_with_domain = []
    
    for domain, file_path in keyword_files.items():
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue
            
        print(f"正在读取{domain}领域文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_keywords = [line.strip() for line in f if line.strip()]
                # 为每个关键词添加领域信息
                domain_keywords = [(keyword, domain) for keyword in file_keywords]
                keywords_with_domain.extend(domain_keywords)
                print(f"从 {file_path} 读取了 {len(file_keywords)} 个{domain}领域关键词")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    
    return keywords_with_domain

def generate_sentence(keyword, domain):
    """为给定关键词和领域生成句子"""
    keyword_with_domain = f"{domain}:{keyword}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "video:何以笙箫默"},
        {"role": "assistant", "content": "我想看电视剧何以笙箫默"},
        {"role": "user", "content": "music:温岚"},
        {"role": "assistant", "content": "我要听温岚的囚鸟"},
        {"role": "user", "content": "city:深圳"},
        {"role": "assistant", "content": "深圳天气怎么样"},
        {"role": "user", "content": keyword_with_domain},
    ]
    
    try:
        output = pipe(messages, **generation_args)
        return output[0]['generated_text'].strip()
    except Exception as e:
        print(f"生成句子时出错，关键词: {keyword_with_domain}, 错误: {e}")
        return None

def save_to_csv(sentences_data, output_file):
    """保存句子数据到CSV文件"""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['manual_transcript', 'source'])
        # 写入数据
        for sentence, domain in sentences_data:
            writer.writerow([sentence, domain])

def main():
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取所有关键词（带领域信息）
    print("正在读取关键词文件...")
    keywords_with_domain = read_keywords_by_domain(KEYWORD_FILES)
    print(f"总共读取到 {len(keywords_with_domain)} 个关键词")
    
    if not keywords_with_domain:
        print("未读取到任何关键词，程序退出")
        return
    
    # 随机打乱关键词顺序
    random.shuffle(keywords_with_domain)
    
    # 生成句子并保存
    sentences_data = []  # 存储 (句子, 领域) 对
    print("开始生成句子...")
    
    for i, (keyword, domain) in enumerate(keywords_with_domain):
        print(f"正在处理关键词 {i+1}/{len(keywords_with_domain)}: {domain}:{keyword}")
        sentence = generate_sentence(keyword, domain)
        
        if sentence:
            sentences_data.append((sentence, domain))
            print(f"生成句子: {sentence} (领域: {domain})")
        
        # 每处理50个关键词保存一次，防止意外中断丢失数据
        if (i + 1) % 50 == 0:
            print(f"已处理 {i+1} 个关键词，保存中间结果...")
            save_to_csv(sentences_data, OUTPUT_FILE)
    
    # 最终保存所有句子
    print(f"保存 {len(sentences_data)} 个句子到 {OUTPUT_FILE}")
    save_to_csv(sentences_data, OUTPUT_FILE)
    
    # 输出统计信息
    domain_counts = {}
    for _, domain in sentences_data:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("各领域句子生成统计:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} 个句子")
    
    print("句子生成完成！")

if __name__ == "__main__":
    main()