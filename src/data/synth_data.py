
import numpy as np, pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

def gen_transactions(n=120_000, fraud_rate=0.02):
    time = np.arange(n)  # índice temporal
    amount = RNG.gamma(shape=2.0, scale=60.0, size=n)  # assimétrico
    V = RNG.normal(0, 1, size=(n, 28))                 # 28 componentes simuladas
    # prob fraude ↑ com amount e extremos em V
    base = fraud_rate + 0.00005*amount
    base = np.clip(base, 0, 0.2)
    is_fraud = (RNG.random(n) < base).astype(int)

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1,29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = is_fraud
    return df

if __name__ == "__main__":
    df = gen_transactions()
    Path("data").mkdir(exist_ok=True, parents=True)
    df.to_csv("data/transactions_synth.csv", index=False)
    print("Gerado: data/transactions_synth.csv", len(df), "linhas")
