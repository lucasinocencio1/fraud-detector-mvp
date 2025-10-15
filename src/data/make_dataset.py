
import pandas as pd
from pathlib import Path
from src.utils.time_split import time_split

if __name__ == "__main__":
    df = pd.read_csv("data/transactions_synth.csv")
    train, val, test = time_split(df, time_col="Time", train_ratio=0.7, val_ratio=0.15)
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    train.to_parquet("artifacts/train.parquet")
    val.to_parquet("artifacts/val.parquet")
    test.to_parquet("artifacts/test.parquet")
    print("Splits salvos em artifacts/*.parquet")
