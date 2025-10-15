
import pandas as pd
from src.utils.time_split import time_split

def test_time_split_is_temporal_and_disjoint():
    df = pd.DataFrame({"Time": range(100)})
    tr, va, te = time_split(df, "Time", 0.7, 0.2)
    assert tr["Time"].is_monotonic_increasing
    assert va["Time"].is_monotonic_increasing
    assert te["Time"].is_monotonic_increasing
    assert tr["Time"].max() < va["Time"].min() <= te["Time"].min()
