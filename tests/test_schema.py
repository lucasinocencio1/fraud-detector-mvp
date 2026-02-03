import pandas as pd
from pandera import Check, Column, DataFrameSchema

schema = DataFrameSchema(
    {
        "time": Column(int, Check.ge(0)),
        "amount": Column(float, Check.ge(0)),
        "class": Column(int, Check.isin([0, 1])),
    }
)


def test_input_schema():
    df = pd.read_csv("data/sample_transactions.csv")
    schema.validate(df, lazy=True)
