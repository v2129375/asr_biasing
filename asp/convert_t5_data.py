import pandas
def convert(csv_path,out_path): 
    df = pandas.read_csv(csv_path)
    df.rename(columns={"text":"input_text","wrong":"target_text"},inplace=True)
    df["prefix"] = "MT"

    df.to_csv(out_path,index=False)

if __name__ == "__main__":
    convert("data/asp_data/train.csv","data/asp_data/train_t5.csv")
    convert("data/asp_data/val.csv","data/asp_data/val_t5.csv")
    convert("data/asp_data/data.csv","data/asp_data/data_t5.csv")