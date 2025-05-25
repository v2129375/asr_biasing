import pandas

df = pandas.read_csv("asr/exp/phi4_finetuned_asr_result.csv")
cal_keyword_wer = True

# 打印错误的样本
print("\n错误识别的样本:")
for i in range(len(df)):
    if not df["keyword"][i] in df["asr"][i]:
        print(f"原始文本: {df['manual_transcript'][i]}")
        print(f"识别结果: {df['asr'][i]}")
        print(f"关键词: {df['keyword'][i]}")
        print("---")

def cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
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


total_cer = 0
keyword_wer = 0
for i in range(len(df)):
    ground = df["manual_transcript"][i]
    asr = df["asr"][i]
    ground = [x for x in ground]
    asr = [x for x in asr]
    this_cer = cer(ground, asr)
    total_cer += this_cer

    if cal_keyword_wer:
        if not df["keyword"][i] in df["asr"][i]:
            keyword_wer += 1

print("cer:",total_cer/len(df))
if cal_keyword_wer:
    print("keyword_wer:",keyword_wer/len(df))



