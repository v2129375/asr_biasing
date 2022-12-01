import pandas

keyword_file = "video_test/keyword.txt"
csv_file = "video_test/test.csv"
f = open(keyword_file)
keywords = f.readlines()
for i in range(len(keywords)):
    keywords[i] = keywords[i].strip("\n")
keywords1 = []
df = pandas.read_csv(csv_file)
for i in range(len(df)):
    keywords1.append(df["keyword"][i])
not_show = []
# print(keywords)
for i in keywords:
    if not i in keywords1:
        not_show.append(i)
print(not_show)
print(len(not_show))
print(len(keywords))
print(len(keywords1))