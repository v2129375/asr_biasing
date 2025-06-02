import pandas
import numpy as np

def cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
    # 如果h为空，CER为100%
    if len(h) == 0:
        return 1.0 if len(r) > 0 else 0.0
    
    # 如果r为空，CER为h的长度除以1（避免除零）
    if len(r) == 0:
        return float(len(h))
    
    # initialisation
    import numpy
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / float(len(r))


def evaluate_asr(df, cal_keyword_wer=True, print_errors=True):
    """
    评估ASR结果的函数
    
    参数:
    df: pandas.DataFrame - 包含 'manual_transcript', 'asr', 'keyword', 'source' 列的数据框
    cal_keyword_wer: bool - 是否计算关键词错误率，默认为True
    print_errors: bool - 是否打印错误识别的样本，默认为True
    
    返回:
    dict - 包含总体和各类别的 'cer' 和 'keyword_wer'(如果计算) 的字典
    """
    
    # 打印错误的样本
    if print_errors:
        print("\n错误识别的样本:")
        for i in range(len(df)):
            # 检查ASR结果是否为空值
            asr_result = df["asr"][i]
            keyword = df["keyword"][i]
            
            # 处理空值情况
            if pandas.isna(asr_result) or asr_result == "" or asr_result is None:
                print(f"原始文本: {df['manual_transcript'][i]}")
                print(f"识别结果: [空值]")
                print(f"关键词: {keyword}")
                print("---")
            elif not str(keyword) in str(asr_result):
                print(f"原始文本: {df['manual_transcript'][i]}")
                print(f"识别结果: {asr_result}")
                print(f"关键词: {keyword}")
                print("---")

    # 获取所有类别
    categories = df['source'].unique()
    
    # 创建结果字典，包含总体结果和各类别结果
    results = {"overall": {}}
    for category in categories:
        results[category] = {}
    
    # 计算总体CER
    total_cer = 0
    keyword_wer = 0
    
    # 为每个类别创建计数器
    category_counts = {category: 0 for category in categories}
    category_cer = {category: 0 for category in categories}
    category_keyword_wer = {category: 0 for category in categories}
    
    for i in range(len(df)):
        ground = df["manual_transcript"][i]
        asr_result = df["asr"][i]
        category = df["source"][i]
        
        # 处理ground为空值的情况
        if pandas.isna(ground) or ground == "" or ground is None:
            ground = ""
        
        # 处理ASR结果为空值的情况
        if pandas.isna(asr_result) or asr_result == "" or asr_result is None:
            asr_result = ""
        
        ground = [x for x in str(ground)]
        asr = [x for x in str(asr_result)]
        this_cer = cer(ground, asr)
        
        # 累加总体CER
        total_cer += this_cer
        
        # 累加类别CER
        category_cer[category] += this_cer
        category_counts[category] += 1

        if cal_keyword_wer:
            keyword = df["keyword"][i]
            # 检查关键词错误率：如果ASR结果为空或关键词不在ASR结果中
            if (pandas.isna(asr_result) or asr_result == "" or asr_result is None or 
                not str(keyword) in str(asr_result)):
                keyword_wer += 1
                category_keyword_wer[category] += 1

    # 计算总体平均CER
    avg_cer = total_cer / len(df)
    results["overall"]["cer"] = avg_cer
    
    # 计算各类别平均CER
    for category in categories:
        if category_counts[category] > 0:
            results[category]["cer"] = category_cer[category] / category_counts[category]
        else:
            results[category]["cer"] = 0
    
    # 如果需要计算关键词错误率
    if cal_keyword_wer:
        # 计算总体关键词错误率
        avg_keyword_wer = keyword_wer / len(df)
        results["overall"]["keyword_wer"] = avg_keyword_wer
        
        # 计算各类别关键词错误率
        for category in categories:
            if category_counts[category] > 0:
                results[category]["keyword_wer"] = category_keyword_wer[category] / category_counts[category]
            else:
                results[category]["keyword_wer"] = 0
    
    # 打印总体结果
    print("\n总体结果:")
    print(f"CER: {results['overall']['cer']:.4f}")
    if cal_keyword_wer:
        print(f"Keyword WER: {results['overall']['keyword_wer']:.4f}")
    
    # 打印各类别结果
    print("\n各类别结果:")
    for category in categories:
        print(f"\n类别: {category}")
        print(f"样本数: {category_counts[category]}")
        print(f"CER: {results[category]['cer']:.4f}")
        if cal_keyword_wer:
            print(f"Keyword WER: {results[category]['keyword_wer']:.4f}")
    
    return results


def main():
    """主函数，用于直接执行脚本时调用"""
    df = pandas.read_csv("asr/exp/phi4_keywords_asr_result_ori.csv")
    return evaluate_asr(df, cal_keyword_wer=True)


if __name__ == "__main__":
    main()



