import os
import pandas as pd
import requests
from supabase import create_client
from dotenv import load_dotenv

# ------------------------------------------------------------
# Configurações
# ------------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://unoabldfhxojxlcvsalr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ULTRAMSG_INSTANCE = os.getenv("ULTRAMSG_INSTANCE") or "instance128916"
ULTRAMSG_TOKEN = os.getenv("ULTRAMSG_TOKEN") or "gtkqpv7chlvb9sc2"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------------------------
# Função: enviar mensagem no WhatsApp
# ------------------------------------------------------------
def send_whatsapp_message(phone, message):
    """Envia mensagem via UltraMsg API"""
    url = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE}/messages/chat"
    payload = {"token": ULTRAMSG_TOKEN, "to": phone, "body": message}
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print(f"Mensagem enviada para {phone}")
    else:
        print(f"Erro ao enviar para {phone}: {response.text}")

# ------------------------------------------------------------
# Função: buscar fraudes no Supabase
# ------------------------------------------------------------
def get_fraud_transactions(limit=10):
    """Busca últimas transações com fraude detectada"""
    data = (
        supabase.table("transactions")
        .select("id, customer_id, amount, region, class")
        .eq("class", 1)  # 1 = fraude
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    return data.data if data.data else []

# ------------------------------------------------------------
# Função: mapear cliente e enviar alerta
# ------------------------------------------------------------
def notify_fraudulent_transactions():
    frauds = get_fraud_transactions()

    if not frauds:
        print("Nenhuma fraude detectada.")
        return

    customers = (
        supabase.table("customers")
        .select("customer_id, name, phone, region")
        .execute()
    ).data

    cust_map = {c["customer_id"]: c for c in customers}

    for f in frauds:
        cust = cust_map.get(f["customer_id"])
        if not cust:
            continue

        msg = (
            f"⚠️ Alerta de possível fraude detectada!\n\n"
            f"Cliente: {cust['name']}\n"
            f"Região: {f['region']}\n"
            f"Valor: €{f['amount']:.2f}\n"
            f"Ação: Revisar transação no painel de risco."
        )
        send_whatsapp_message(cust["phone"], msg)

# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------
if __name__ == "__main__":
    notify_fraudulent_transactions()
