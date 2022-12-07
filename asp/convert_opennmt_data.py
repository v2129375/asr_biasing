import pandas
from tqdm import tqdm
import jieba
def convert(csv_path,out_path):
    f_src = open(out_path.rstrip(out_path.split("/")[-1]) + "src-" + out_path.split("/")[-1] ,"w",encoding="utf-8")
    f_tgt = open(out_path.rstrip(out_path.split("/")[-1]) + "tgt-" + out_path.split("/")[-1] ,"w",encoding="utf-8")
    df = pandas.read_csv(csv_path)
    for i in tqdm(range(len(df))):
        for j in range(len(str(df["text"][i]))):
            f_src.write(str(df["text"][i])[j]+" ")
            f_tgt.write(str(df["wrong"][i])[j]+" ")
        f_src.write("\n")
        f_tgt.write("\n")

# 将keyword词表加上空格，符合opennmt的资料格式要求
def convert_txt(input,output): 
    f = open(input,"r")
    input_lines = f.readlines()
    f = open(output,"w",encoding="utf-8")
    for i in input_lines:
        for j in i:
            if j != "\n":
                f.write(j+" ")
            else:
                f.write(j)

# 先用jieba对关键词断词，然后将断词后的词建立ASP table的生成资料
def convert_txt_jieba(input,output): 
    keyword = open(input,"r").read().splitlines()
    keyword_jieba = []
    in_nomal_word = []
    for i in keyword:
        keyword_jieba+=list(jieba.cut(i))
    for i in keyword_jieba: # 删除常见词
        if check_nomal_word(i,"asp/exp/nomal_word.txt"):
            in_nomal_word.append(i)
    for i in in_nomal_word:
        keyword_jieba.remove(i)
    pandas.DataFrame(keyword_jieba).to_csv(output,sep="\t",header=None,index=None)
    convert_txt(output,output+"_space")

def make_nomal_word_table(data_csv,nomal_num,out_txt): # 制作常见词表
    df = pandas.read_csv(data_csv)
    counter_dict={}
    for i in tqdm(range(len(df))):
        text = df["text"][i]
        for j in range(len(text)):
            word = text[j]
            if word in counter_dict:
                counter_dict[word] += 1
            else:
                counter_dict[word] = 1
            
            if j+2 <= len(text):
                word = text[j:j+2]
                if word in counter_dict:
                    counter_dict[word] += 1
                else:
                    counter_dict[word] = 1
            
            if j+3 <= len(text):
                word = text[j:j+3]
                if word in counter_dict:
                    counter_dict[word] += 1
                else:
                    counter_dict[word] = 1
            
            if j+4 <= len(text):
                word = text[j:j+4]
                if word in counter_dict:
                    counter_dict[word] += 1
                else:
                    counter_dict[word] = 1

    nomal_word = []
    for i in counter_dict.keys():
        if counter_dict[i] >= nomal_num:
            nomal_word.append(i)
    
    pandas.DataFrame(nomal_word).to_csv(out_txt,sep="\t",header=None,index=None)

def check_nomal_word(word,nomal_word): # 检查目标词是否为常见词
    nomal_word = open(nomal_word,"r").read().splitlines()
    if word in nomal_word:
        return True
    else:
        return False

def convert_2_asp_table(asp_out,ori_keyword,out_csv,nbest): # 将asp模型的输出转换为asp_table同时进行基于规则的资料清理
    df_ori = pandas.read_table(ori_keyword,header=None,names=["text"])
    df_asp = pandas.read_table(asp_out,sep="\t",header=None,names=["asp","score"])
    asp_word = []
    for i in tqdm(range(0,len(df_asp),nbest)):
        asp_word.append([])
        for j in range(nbest):
            str = df_asp["asp"][i+j].replace(" ","")
            # 删除有oov的词和信度过低的词，并删除异常长度太长的词,删除常见词
            if "<unk>" not in str and \
            df_asp["score"][i+j] > -1 and \
            len(str) < 15 and \
            not check_nomal_word(str,"asp/exp/nomal_word.txt"):
                asp_word[i//nbest].append(str)
    df_ori["asp"] = asp_word
    # 删去与原来相同的词
    for i in range(len(df_ori)):
        if df_ori["text"][i] in df_ori["asp"][i]:
            df_ori["asp"][i].remove(df_ori["text"][i])
    df_ori.to_csv(out_csv,index=False)



if __name__ == "__main__":
    # convert("data/asp_data/train.csv","data/asp_data/train_opennmt.txt")
    # convert("data/asp_data/val.csv","data/asp_data/val_opennmt.txt")
    # convert("data/asp_data/data.csv","data/asp_data/data_opennmt.txt")
    # convert_txt("data/video_test/keyword.txt","data/video_test/keyword_space.txt")
    # convert_txt_jieba("data/video_test/keyword.txt","data/video_test/keyword_jieba.txt")
    # make_nomal_word_table("data/aishell2/data.csv",500,"asp/exp/nomal_word.txt")
    convert_2_asp_table("asp/exp/opennmt_out.txt","data/video_test/keyword_jieba.txt","asp/exp/asp_table_jieba.csv",5)
    # print(check_nomal_word("你好","asp/exp/nomal_word.txt"))
    