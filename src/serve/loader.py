import joblib

ART_SUP = "artifacts/supervised_xgb.joblib"


def load_supervised():
    obj = joblib.load(ART_SUP)
    return obj["model"], obj["features"]
