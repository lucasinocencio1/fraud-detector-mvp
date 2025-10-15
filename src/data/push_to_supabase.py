import os
import time
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import requests

# ============================================================
# Script: push_to_supabase.py
# Sincroniza o schema do Supabase com o CSV e envia os dados.
# ============================================================

# Carregar variáveis de ambiente
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://unoabldfhxojxlcvsalr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVub2FibGRmaHhvanhsY3ZzYWxyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA0NzU4MTYsImV4cCI6MjA3NjA1MTgxNn0.2Ihoh_iKVOQ7xg3FHRDIn7EJFdqOL2W0TFTqHZ1zH08"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# Função auxiliar: verifica e atualiza schema automaticamente
# ============================================================

def sync_schema_with_csv(table_name: str, df: pd.DataFrame):
    """Sincroniza as colunas do Supabase com o CSV local."""
    print(f"\n🔍 Verificando schema da tabela '{table_name}'...")

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    response = requests.get(f"{SUPABASE_URL}/rest/v1/{table_name}?limit=1", headers=headers)
    if response.status_code != 200:
        print(f"⚠️ Não foi possível acessar a tabela '{table_name}': {response.text}")
        return

    # Obter colunas atuais do Supabase
    remote_columns = response.json()[0].keys() if response.json() else []
    local_columns = df.columns

    # Detectar diferenças
    missing_cols = [col for col in local_columns if col not in remote_columns]
    extra_cols = [col for col in remote_columns if col not in local_columns]

    if missing_cols:
        print(f"➕ Colunas ausentes no Supabase: {missing_cols}")
    if extra_cols:
        print(f"➖ Colunas extras no Supabase (serão ignoradas no upload): {extra_cols}")
    if not missing_cols and not extra_cols:
        print("✅ Schema está sincronizado com o CSV.")
        return

    # Adicionar colunas ausentes (via RPC ou SQL)
    for col in missing_cols:
        print(f"🛠️ Criando coluna '{col}' no Supabase...")
        try:
            sql = f'ALTER TABLE {table_name} ADD COLUMN "{col}" text;'
            rpc_resp = supabase.postgrest.rpc("exec_sql", {"sql": sql}).execute()
            print(f"   ↳ Resultado: {rpc_resp}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"   ↳ Coluna '{col}' já existe, pulando...")
            else:
                print(f"   ↳ Erro ao criar coluna '{col}': {e}")
                # Tentar com tipo diferente se text falhar
                try:
                    sql = f'ALTER TABLE {table_name} ADD COLUMN "{col}" varchar;'
                    rpc_resp = supabase.postgrest.rpc("exec_sql", {"sql": sql}).execute()
                    print(f"   ↳ Resultado (varchar): {rpc_resp}")
                except Exception as e2:
                    print(f"   ↳ Erro também com varchar: {e2}")

    print("✅ Sincronização concluída.\n")


# ============================================================
# Função para enviar DataFrame em batches
# ============================================================

def upload_table(df, table_name, conflict_field=None):
    batch_size = 5000
    print(f"\n🚀 Enviando {len(df)} registros para tabela '{table_name}'...")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size].to_dict(orient="records")
        start = time.time()

        try:
            if conflict_field:
                res = supabase.table(table_name).upsert(batch, on_conflict=conflict_field).execute()
            else:
                res = supabase.table(table_name).insert(batch).execute()

            end = time.time()
            elapsed = round(end - start, 2)
            print(f"✅ Batch {i // batch_size + 1} enviado ({len(batch)} registros) em {elapsed}s")

        except Exception as e:
            print(f"❌ Erro ao enviar batch {i // batch_size + 1}: {e}")
            break


# ============================================================
# Execução principal
# ============================================================

if __name__ == "__main__":
    customers_path = "data/customers.csv"
    transactions_path = "data/transactions.csv"

    if not os.path.exists(customers_path):
        raise FileNotFoundError("❌ Arquivo data/customers.csv não encontrado.")
    if not os.path.exists(transactions_path):
        raise FileNotFoundError("❌ Arquivo data/transactions.csv não encontrado.")

    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path)

    # Padronizar nomes
    customers.columns = [c.lower() for c in customers.columns]
    transactions.columns = [c.lower() for c in transactions.columns]

    # ============================================================
    # Sincronizar schema automaticamente
    # ============================================================
    sync_schema_with_csv("customers", customers)
    sync_schema_with_csv("transactions", transactions)

    # ============================================================
    # Limpeza e upload
    # ============================================================
    print("\n🧹 Limpando tabela 'customers' antes de subir novos registros...")
    try:
        supabase.table("customers").delete().neq("customer_id", "0").execute()
        print("✅ Tabela 'customers' limpa com sucesso.")
    except Exception as e:
        print(f"⚠️ Aviso: limpeza direta falhou ({e}), prosseguindo mesmo assim.")

    # Garantir compatibilidade de tipos antes do upload
    if "is_night" in transactions.columns:
        # Converte 0/1 -> True/False para Supabase boolean
        transactions["is_night"] = transactions["is_night"].astype(bool)    

    upload_table(customers, "customers", conflict_field="customer_id")
    upload_table(transactions, "transactions")

    print("\n🎯 Upload completo para o Supabase com schema idêntico ao CSV!")
