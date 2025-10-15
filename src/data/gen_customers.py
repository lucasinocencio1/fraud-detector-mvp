# src/data/gen_customers.py
import uuid
import pandas as pd
from faker import Faker
from pathlib import Path
R = Faker("pt_BR")
R.seed_instance(42)

REGIONS = ["EU","US","BR","ASIA"]
NAMES = 10000

def gen_customers(n=NAMES):
    rows = []
    for _ in range(n):
        cid = str(uuid.uuid4())
        name = R.name()
        phone = R.phone_number()  # gerará números no formato BR; podem ser "fictícios"
        email = R.email()
        region = R.random.choice(REGIONS)
        rows.append({"customer_id": cid, "name": name, "phone": phone, "email": email, "region": region})
    df = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True, parents=True)
    df.to_csv("data/customers.csv", index=False)
    print("Gerado: data/customers.csv", len(df))
    return df

if __name__ == "__main__":
    gen_customers()
