import os
import time
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import requests
import uuid
import numpy as np

# ============================================================
# Script: push_to_supabase.py
# Sincroniza o schema do Supabase com o CSV e envia os dados.
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://unoabldfhxojxlcvsalr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVub2FibGRmaHhvanhsY3ZzYWxyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA0NzU4MTYsImV4cCI6MjA3NjA1MTgxNn0.2Ihoh_iKVOQ7xg3FHRDIn7EJFdqOL2W0TFTqHZ1zH08"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# SCHEMA SYNC
# ============================================================

def sync_schema_with_csv(table_name: str, df: pd.DataFrame):
    print(f"\n🔍 Verificando schema da tabela '{table_name}'...")

    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/{table_name}?limit=1", headers=headers)
    if resp.status_code != 200:
        print(f"⚠️ Erro ao acessar '{table_name}': {resp.text}")
        return

    remote_cols = resp.json()[0].keys() if resp.json() else []
    local_cols = df.columns
    missing = [c for c in local_cols if c not in remote_cols]
    extra = [c for c in remote_cols if c not in local_cols]

    if missing:
        print(f"➕ Faltando no Supabase: {missing}")
    if extra:
        print(f"➖ Extras no Supabase: {extra}")

    for col in missing:
        try:
            sql = f'ALTER TABLE {table_name} ADD COLUMN "{col}" text;'
            supabase.postgrest.rpc("exec_sql", {"sql": sql}).execute()
            print(f"🛠️ Coluna '{col}' criada.")
        except Exception as e:
            if "already exists" in str(e):
                print(f"↳ '{col}' já existe, pulando.")
            else:
                print(f"⚠️ Erro ao criar '{col}': {e}")

# ============================================================
# UPLOAD EM BATCHES
# ============================================================

def upload_table(df, table_name, conflict_field=None):
    batch_size = 5000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size].to_dict(orient="records")
        start = time.time()
        try:
            if conflict_field:
                supabase.table(table_name).upsert(batch, on_conflict=conflict_field).execute()
            else:
                supabase.table(table_name).insert(batch).execute()
            print(f"✅ Batch {i // batch_size + 1} enviado ({len(batch)} registros)")
        except Exception as e:
            print(f"❌ Erro no batch {i // batch_size + 1}: {e}")
            break
        print(f"⏱️ Tempo: {round(time.time() - start, 2)}s")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    customers_path = "data/customers.csv"
    transactions_path = "data/transactions.csv"

    if not os.path.exists(customers_path):
        raise FileNotFoundError("❌ data/customers.csv ausente.")
    if not os.path.exists(transactions_path):
        raise FileNotFoundError("❌ data/transactions.csv ausente.")

    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path)

    # Garantir UUIDs válidos para customers
    if "customer_id" not in customers.columns:
        customers["customer_id"] = [str(uuid.uuid4()) for _ in range(len(customers))]
    else:
        nulls = customers["customer_id"].isna().sum()
        if nulls > 0:
            print(f"⚠️ {nulls} customer_id nulos — regenerando.")
            customers["customer_id"] = [str(uuid.uuid4()) for _ in range(len(customers))]

    # Garantir vínculo em transactions
    if "customer_id" not in transactions.columns:
        transactions["customer_id"] = np.random.choice(customers["customer_id"], size=len(transactions))

    customers.to_csv(customers_path, index=False)
    transactions.to_csv(transactions_path, index=False)
    print("💾 CSVs atualizados localmente.")

    sync_schema_with_csv("customers", customers)
    sync_schema_with_csv("transactions", transactions)

    print("\n🧹 Limpando 'customers' antes de subir...")
    try:
        supabase.table("customers").delete().neq("customer_id", "0").execute()
        print("✅ Limpeza concluída.")
    except Exception as e:
        print(f"⚠️ Falha ao limpar: {e}")

    if "is_night" in transactions.columns:
        transactions["is_night"] = transactions["is_night"].astype(bool)

    upload_table(customers, "customers", conflict_field="customer_id")
    upload_table(transactions, "transactions")

    print("\n🎯 Upload completo com sucesso!")
