
import pandas as pd
import numpy as np
from pathlib import Path

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount_log1p"] = np.log1p(df["Amount"])
    for col in ["Amount","Amount_log1p"]:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
    return df

if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    for split in ["train","val","test"]:
        df = pd.read_parquet(f"artifacts/{split}.parquet")
        feat = build_features(df)
        feat.to_parquet(f"artifacts/{split}_feat.parquet")
    print("Features geradas em artifacts/*_feat.parquet")
