import os
import time
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

# ------------------------------------------------------------
# Script: push_to_supabase.py
# Envia as tabelas customers e transactions para o Supabase
# com controle de batches e compatibilidade total com schema.
# ------------------------------------------------------------

# Carregar variáveis de ambiente
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://unoabldfhxojxlcvsalr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVub2FibGRmaHhvanhsY3ZzYWxyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA0NzU4MTYsImV4cCI6MjA3NjA1MTgxNn0.2Ihoh_iKVOQ7xg3FHRDIn7EJFdqOL2W0TFTqHZ1zH08"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------------------------
# Função genérica para enviar DataFrame em batches
# ------------------------------------------------------------

def upload_table(df, table_name, conflict_field=None):
    batch_size = 5000
    print(f"Enviando {len(df)} registros para tabela '{table_name}'...")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size].to_dict(orient="records")
        start = time.time()

        if conflict_field:
            res = supabase.table(table_name).upsert(batch, on_conflict=conflict_field).execute()
        else:
            res = supabase.table(table_name).insert(batch).execute()

        end = time.time()
        elapsed = round(end - start, 2)

        if res.data is not None:
            print(f"Batch {i // batch_size + 1} enviado ({len(batch)} registros) em {elapsed}s")
        else:
            print(f"Falha no batch {i // batch_size + 1}: {res}")

# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------

if __name__ == "__main__":
    customers_path = "data/customers.csv"
    transactions_path = "data/transactions.csv"

    if not os.path.exists(customers_path):
        raise FileNotFoundError("Arquivo data/customers.csv não encontrado.")
    if not os.path.exists(transactions_path):
        raise FileNotFoundError("Arquivo data/transactions.csv não encontrado.")

    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path)

    customers.columns = [c.lower() for c in customers.columns]
    transactions.columns = [c.lower() for c in transactions.columns]

    # --------------------------------------------------------
    # Limpeza segura da tabela customers via função RPC
    # --------------------------------------------------------
    print("Limpando tabela 'customers' antes de subir novos registros...")

    try:
        # Usa SQL direto via RPC (função interna do Supabase)
        supabase.postgrest.rpc("exec_sql", {"sql": "DELETE FROM customers;"}).execute()
    except Exception as e:
        print(f"Aviso: limpeza direta via RPC falhou ({e}), prosseguindo com upsert.")

    # Enviar dados
    upload_table(customers, "customers", conflict_field="customer_id")
    upload_table(transactions, "transactions")

    print("Upload completo.")
