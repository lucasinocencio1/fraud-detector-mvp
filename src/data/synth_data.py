"""
src/data/synth_data.py

Geração de dataset sintético com variáveis realistas para detecção de fraudes.
Inclui vinculação automática com customers.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import uuid
import os

# ============================================================
# CONFIGURAÇÃO
# ============================================================

RNG = np.random.default_rng(42)

REGIONS = ["EU", "US", "BR", "ASIA"]
DEVICE_TYPES = ["mobile", "desktop"]
MERCHANT_CATS = ["electronics", "groceries", "fashion", "gaming", "travel", "restaurants"]

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def gen_transactions(n=250_000, fraud_rate=0.07, start_time=0):
    """Gera transações sintéticas explicáveis."""
    time = np.arange(start_time, start_time + n)
    region = RNG.choice(REGIONS, size=n, p=[0.45, 0.25, 0.15, 0.15])
    device_type = RNG.choice(DEVICE_TYPES, size=n, p=[0.65, 0.35])
    merchant_category = RNG.choice(MERCHANT_CATS, size=n)

    transaction_hour = RNG.integers(0, 24, size=n)
    is_night = (transaction_hour >= 0) & (transaction_hour <= 5)
    is_weekend = RNG.choice([0, 1], size=n, p=[0.7, 0.3])

    amount = RNG.gamma(shape=2.2, scale=70.0, size=n)
    avg_amount_user = RNG.gamma(shape=2.0, scale=80.0, size=n)
    amount_to_avg_ratio = amount / (avg_amount_user + 1e-6)
    tx_last_24h = RNG.poisson(2, n)
    tx_last_7d = tx_last_24h + RNG.poisson(4, n)

    base = np.full(n, fraud_rate, dtype=float)
    base += 0.00008 * np.clip(amount - 200, 0, None)
    base += 0.015 * is_night
    base += 0.01 * (device_type == "mobile")
    base += 0.02 * np.isin(merchant_category, ["electronics", "travel", "gaming"])
    base += 0.01 * np.isin(region, ["US", "ASIA"])
    base += 0.01 * (tx_last_24h > 10)
    base += 0.02 * (amount_to_avg_ratio > 2.5)
    base = np.clip(base, 0, 0.8)

    is_fraud = (RNG.random(n) < base).astype(int)

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


# ============================================================
# VINCULAR COM CUSTOMERS.CSV
# ============================================================

def attach_customers(df):
    """Adiciona customer_id às transações com base em customers.csv."""
    customers_path = "data/customers.csv"
    if os.path.exists(customers_path):
        customers = pd.read_csv(customers_path)
        if "customer_id" not in customers.columns:
            print("⚠️ customers.csv sem customer_id — criando...")
            customers["customer_id"] = [str(uuid.uuid4()) for _ in range(len(customers))]
            customers.to_csv(customers_path, index=False)
        df["customer_id"] = np.random.choice(customers["customer_id"], size=len(df))
    else:
        print("⚠️ Nenhum customers.csv encontrado — gerando IDs aleatórios.")
        df["customer_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    return df


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    df = gen_transactions()
    df = attach_customers(df)
    Path("data").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)

    print(f"✅ Dataset gerado: data/transactions.csv ({len(df)} linhas)")
    print(f"🎯 Taxa de fraude: {df['class'].mean() * 100:.3f}%")
    print("🔹 Colunas:", ", ".join(df.columns))
