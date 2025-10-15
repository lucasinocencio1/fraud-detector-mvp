from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

REGIONS = ["EU", "US", "BR", "ASIA"]
DEVICE_TYPES = ["mobile", "desktop"]
MERCHANT_CATS = [
    "electronics",
    "groceries",
    "fashion",
    "gaming",
    "travel",
    "restaurants",
]


def gen_transactions(n=250_000, fraud_rate=0.012, start_time=0):
    """
    Gera n transações com variáveis categóricas e numéricas:
    - region, device_type, merchant_category
    - transaction_hour [0..23]
    - Amount assimétrico
    - V1..V28 componentes "latentes"
    Fraude com dependências reais (mobile, madrugada, categorias de alto ticket, amount alto).
    """
    # tempo "unix-like" simples para split temporal
    time = np.arange(start_time, start_time + n)

    # variáveis categóricas
    region = RNG.choice(REGIONS, size=n, p=[0.45, 0.25, 0.15, 0.15])
    device_type = RNG.choice(DEVICE_TYPES, size=n, p=[0.65, 0.35])
    merchant_category = RNG.choice(
        MERCHANT_CATS, size=n, p=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15]
    )

    # hora da transação (madrugada terá maior chance de fraude)
    transaction_hour = RNG.integers(0, 24, size=n)

    # amount com cauda longa
    amount = RNG.gamma(shape=2.2, scale=70.0, size=n)  # ~ ticket médio ~150

    # latentes
    V = RNG.normal(0, 1, size=(n, 28))

    # probabilidade base de fraude
    base = np.full(n, fraud_rate, dtype=float)

    # padrões de risco:
    # amounts muito altos
    base += 0.00008 * np.clip(amount - 200, 0, None)  # acima de 200 aumenta risco
    # madrugada (0-5)
    base += 0.015 * ((transaction_hour >= 0) & (transaction_hour <= 5))
    # mobile um pouco mais arriscado
    base += 0.01 * (device_type == "mobile")
    # categorias de alto risco
    base += 0.02 * np.isin(merchant_category, ["electronics", "travel", "gaming"])
    # regiões com maior chargeback histórico (exemplo)
    base += 0.01 * np.isin(region, ["US", "ASIA"])
    # extremos nos latentes
    base += 0.02 * (np.abs(V[:, 0]) > 2.0)
    base += 0.02 * (V[:, 1] < -2.0)

    base = np.clip(base, 0, 0.8)
    is_fraud = (RNG.random(n) < base).astype(int)

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = is_fraud
    df["region"] = region
    df["device_type"] = device_type
    df["merchant_category"] = merchant_category
    df["transaction_hour"] = transaction_hour

    return df


if __name__ == "__main__":
    df = gen_transactions()
    Path("data").mkdir(exist_ok=True, parents=True)
    df.to_csv("data/transactions.csv", index=False)
    print("Gerado: data/transactions.csv", len(df), "linhas")
