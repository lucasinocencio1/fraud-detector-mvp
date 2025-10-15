"""
src/data/synth_data.py

Geração de dataset sintético com variáveis realistas para detecção de fraudes.
Substitui os componentes V1..V28 por features interpretáveis:
- Comportamento de gasto
- Tempo da transação
- Histórico do cliente
- Categoria e dispositivo
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Semente global para reprodutibilidade
RNG = np.random.default_rng(42)

REGIONS = ["EU", "US", "BR", "ASIA"]
DEVICE_TYPES = ["mobile", "desktop"]
MERCHANT_CATS = [
    "electronics", "groceries", "fashion",
    "gaming", "travel", "restaurants"
]


def gen_transactions(n=250_000, fraud_rate=0.04, start_time=0):
    """
    Gera n transações sintéticas com variáveis explicáveis.
    Atribui risco com base em combinações realistas de fatores.
    """

    # Sequência temporal
    time = np.arange(start_time, start_time + n)

    # Variáveis categóricas
    region = RNG.choice(REGIONS, size=n, p=[0.45, 0.25, 0.15, 0.15])
    device_type = RNG.choice(DEVICE_TYPES, size=n, p=[0.65, 0.35])
    merchant_category = RNG.choice(MERCHANT_CATS, size=n, p=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15])

    # Tempo e comportamento
    transaction_hour = RNG.integers(0, 24, size=n)
    is_night = (transaction_hour >= 0) & (transaction_hour <= 5)
    is_weekend = RNG.choice([0, 1], size=n, p=[0.7, 0.3])

    # Valor e perfil de gasto
    amount = RNG.gamma(shape=2.2, scale=70.0, size=n)
    avg_amount_user = RNG.gamma(shape=2.0, scale=80.0, size=n)
    amount_to_avg_ratio = amount / (avg_amount_user + 1e-6)
    tx_last_24h = RNG.poisson(2, n)
    tx_last_7d = tx_last_24h + RNG.poisson(4, n)

    # Probabilidade base de fraude
    base = np.full(n, fraud_rate, dtype=float)

    # Aumenta risco com padrões típicos
    base += 0.00008 * np.clip(amount - 200, 0, None)
    base += 0.015 * is_night
    base += 0.01 * (device_type == "mobile")
    base += 0.02 * np.isin(merchant_category, ["electronics", "travel", "gaming"])
    base += 0.01 * np.isin(region, ["US", "ASIA"])
    base += 0.01 * (tx_last_24h > 10)
    base += 0.02 * (amount_to_avg_ratio > 2.5)

    # Normaliza e aplica corte
    base = np.clip(base, 0, 0.8)
    is_fraud = (RNG.random(n) < base).astype(int)

    # Monta o DataFrame
    df = pd.DataFrame({
        "time": time,
        "region": region,
        "device_type": device_type,
        "merchant_category": merchant_category,
        "transaction_hour": transaction_hour,
        "is_night": is_night.astype(int),
        "is_weekend": is_weekend,
        "amount": amount,
        "avg_amount_user": avg_amount_user,
        "amount_to_avg_ratio": amount_to_avg_ratio,
        "tx_last_24h": tx_last_24h,
        "tx_last_7d": tx_last_7d,
        "class": is_fraud
    })

    return df


if __name__ == "__main__":
    df = gen_transactions()
    Path("data").mkdir(exist_ok=True, parents=True)
    df.to_csv("data/transactions.csv", index=False)

    fraud_rate = df["class"].mean() * 100
    print(f"✅ Dataset gerado: data/transactions.csv ({len(df)} linhas)")
    print(f"🎯 Taxa de fraude: {fraud_rate:.3f}%")
    print("🔹 Colunas:", ", ".join(df.columns))
