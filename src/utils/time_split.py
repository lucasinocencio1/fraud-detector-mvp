import pandas as pd


def time_split(df: pd.DataFrame, time_col="Time", train_ratio=0.7, val_ratio=0.15):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
