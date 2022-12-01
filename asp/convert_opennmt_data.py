import pandas
from tqdm import tqdm
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

if __name__ == "__main__":
    convert("data/asp_data/train.csv","data/asp_data/train_opennmt.txt")
    convert("data/asp_data/val.csv","data/asp_data/val_opennmt.txt")
    convert("data/asp_data/data.csv","data/asp_data/data_opennmt.txt")
    