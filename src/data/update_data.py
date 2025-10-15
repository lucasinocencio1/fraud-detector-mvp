import pandas as pd
from pathlib import Path
from src.data.synth_data import gen_transactions

def update_transactions(batch=50_000):
    """
    Anexa um novo lote de transações ao CSV mestre, simulando atualização diária.
    """
    Path("data").mkdir(exist_ok=True, parents=True)
    master = Path("data/transactions.csv")
    if master.exists():
        df_old = pd.read_csv(master)
        start_time = df_old["Time"].max() + 1
        df_new = gen_transactions(n=batch, start_time=start_time)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(master, index=False)
        print(f"Atualizado: +{len(df_new)} linhas -> total {len(df_all)}")
    else:
        df_new = gen_transactions(n=batch, start_time=0)
        df_new.to_csv(master, index=False)
        print(f"Criado: data/transactions.csv com {len(df_new)} linhas")

if __name__ == "__main__":
    update_transactions()
