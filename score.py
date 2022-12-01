import pandas

df = pandas.read_csv("wenet_decode/exp/result_biasing.csv")


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

    if not df["keyword"][i] in df["asr"][i]:
        keyword_wer += 1

print("cer:",total_cer/len(df))
print("keyword_wer:",keyword_wer/len(df))
