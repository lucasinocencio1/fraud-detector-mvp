# import os
# import requests
# from supabase import create_client
# from dotenv import load_dotenv

# # Configuration
# load_dotenv()

# SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://unoabldfhxojxlcvsalr.supabase.co"
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# ULTRAMSG_INSTANCE = os.getenv("ULTRAMSG_INSTANCE") or "instance128916"
# ULTRAMSG_TOKEN = os.getenv("ULTRAMSG_TOKEN") or "gtkqpv7chlvb9sc2"

# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# # WhatsApp messaging helper
# def send_whatsapp_message(phone, message):
#     """Send a WhatsApp message through the UltraMsg API."""
#     url = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE}/messages/chat"
#     payload = {"token": ULTRAMSG_TOKEN, "to": phone, "body": message}
#     response = requests.post(url, data=payload)
#     if response.status_code == 200:
#         print(f"Message sent to {phone}")
#     else:
#         print(f"Failed to send message to {phone}: {response.text}")

# # ------------------------------------------------------------
# # Supabase lookups
# # ------------------------------------------------------------
# def get_fraud_transactions(limit=10):
#     """Fetch the latest confirmed fraud transactions."""
#     data = (
#         supabase.table("transactions")
#         .select("id, customer_id, amount, region, class")
#         .eq("class", 1)  # 1 = fraud
#         .order("id", desc=True)
#         .limit(limit)
#         .execute()
#     )
#     return data.data if data.data else []

# # ------------------------------------------------------------
# # Customer mapping and alert dispatch
# # ------------------------------------------------------------
# def notify_fraudulent_transactions():
#     frauds = get_fraud_transactions()

#     if not frauds:
#         print("No fraudulent transactions detected.")
#         return

#     customers = (
#         supabase.table("customers")
#         .select("customer_id, name, phone, region")
#         .execute()
#     ).data

#     cust_map = {c["customer_id"]: c for c in customers}

#     for f in frauds:
#         cust = cust_map.get(f["customer_id"])
#         if not cust:
#             continue

#         msg = (
#             "Fraud alert detected.\n\n"
#             f"Customer: {cust['name']}\n"
#             f"Region: {f['region']}\n"
#             f"Amount: €{f['amount']:.2f}\n"
#             "Action: Review the transaction in the risk console."
#         )
#         send_whatsapp_message(cust["phone"], msg)

# # Entrypoint
# if __name__ == "__main__":
#     notify_fraudulent_transactions()
