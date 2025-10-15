import os
from supabase import create_client, Client
from dotenv import load_dotenv

# ============================================================
# Script: update_schema_supabase.py
# Remove colunas antigas e adiciona apenas as novas colunas
# necessárias no Supabase (tabela "transactions").
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://unoabldfhxojxlcvsalr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVub2FibGRmaHhvanhsY3ZzYWxyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA0NzU4MTYsImV4cCI6MjA3NjA1MTgxNn0.2Ihoh_iKVOQ7xg3FHRDIn7EJFdqOL2W0TFTqHZ1zH08"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# Colunas antigas e novas
# ============================================================

old_cols = [f"v{i}" for i in range(1, 29)]
new_cols_sql = """
ALTER TABLE transactions
ADD COLUMN IF NOT EXISTS avg_amount_user FLOAT,
ADD COLUMN IF NOT EXISTS amount_to_avg_ratio FLOAT,
ADD COLUMN IF NOT EXISTS tx_last_24h FLOAT,
ADD COLUMN IF NOT EXISTS tx_last_7d FLOAT,
ADD COLUMN IF NOT EXISTS amount_log FLOAT,
ADD COLUMN IF NOT EXISTS high_amount_flag BOOLEAN,
ADD COLUMN IF NOT EXISTS hour_sin FLOAT,
ADD COLUMN IF NOT EXISTS hour_cos FLOAT,
ADD COLUMN IF NOT EXISTS region_risk INT,
ADD COLUMN IF NOT EXISTS v_mean FLOAT,
ADD COLUMN IF NOT EXISTS v_std FLOAT,
ADD COLUMN IF NOT EXISTS is_night BOOLEAN;
"""

# ============================================================
# Execução principal
# ============================================================

print("🚀 Atualizando schema da tabela 'transactions' no Supabase...")

# 1️⃣ Remove colunas antigas (v1–v28)
drop_sql = "ALTER TABLE transactions " + ", ".join([f"DROP COLUMN IF EXISTS {c}" for c in old_cols]) + ";"

try:
    supabase.postgrest.rpc("exec_sql", {"sql": drop_sql}).execute()
    print("✅ Colunas antigas (V1–V28) removidas com sucesso.")
except Exception as e:
    print(f"⚠️ Falha ao remover colunas antigas: {e}")

# 2️⃣ Adiciona colunas novas
try:
    supabase.postgrest.rpc("exec_sql", {"sql": new_cols_sql}).execute()
    print("✅ Novas colunas adicionadas com sucesso.")
except Exception as e:
    print(f"⚠️ Falha ao adicionar novas colunas: {e}")

print("🎯 Schema atualizado com sucesso!")
