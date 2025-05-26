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
    df: pandas.DataFrame - 包含 'manual_transcript', 'asr', 'keyword' 列的数据框
    cal_keyword_wer: bool - 是否计算关键词错误率，默认为True
    print_errors: bool - 是否打印错误识别的样本，默认为True
    
    返回:
    dict - 包含 'cer' 和 'keyword_wer'(如果计算) 的字典
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

    total_cer = 0
    keyword_wer = 0
    
    for i in range(len(df)):
        ground = df["manual_transcript"][i]
        asr_result = df["asr"][i]
        
        # 处理ground为空值的情况
        if pandas.isna(ground) or ground == "" or ground is None:
            ground = ""
        
        # 处理ASR结果为空值的情况
        if pandas.isna(asr_result) or asr_result == "" or asr_result is None:
            asr_result = ""
        
        ground = [x for x in str(ground)]
        asr = [x for x in str(asr_result)]
        this_cer = cer(ground, asr)
        total_cer += this_cer

        if cal_keyword_wer:
            keyword = df["keyword"][i]
            # 检查关键词错误率：如果ASR结果为空或关键词不在ASR结果中
            if (pandas.isna(asr_result) or asr_result == "" or asr_result is None or 
                not str(keyword) in str(asr_result)):
                keyword_wer += 1

    # 计算平均CER
    avg_cer = total_cer / len(df)
    
    # 准备返回结果
    results = {"cer": avg_cer}
    
    if cal_keyword_wer:
        avg_keyword_wer = keyword_wer / len(df)
        results["keyword_wer"] = avg_keyword_wer
    
    # 打印结果
    print("cer:", avg_cer)
    if cal_keyword_wer:
        print("keyword_wer:", results["keyword_wer"])
    
    return results


def main():
    """主函数，用于直接执行脚本时调用"""
    df = pandas.read_csv("asr/exp/phi4_keywords_asr_result.csv")
    return evaluate_asr(df, cal_keyword_wer=True)


if __name__ == "__main__":
    main()



