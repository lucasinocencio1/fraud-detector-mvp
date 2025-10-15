
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

schema = DataFrameSchema({
    "Time": Column(int, Check.ge(0)),
    "Amount": Column(float, Check.ge(0)),
    "Class": Column(int, Check.isin([0,1])),
})

def test_input_schema():
    df = pd.read_csv("data/transactions_synth.csv").head(1000)
    schema.validate(df, lazy=True)
