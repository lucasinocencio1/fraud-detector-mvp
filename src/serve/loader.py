import joblib

ART_XGB = "artifacts/supervised_xgb.joblib"
ART_LGBM = "artifacts/supervised_lgbm.joblib"
ART_THRESHOLD = "artifacts/optimal_threshold.joblib"
ART_FEATURES = "artifacts/feature_names.joblib"


def load_supervised():
    """Carrega modelos ensemble (XGBoost + LightGBM) e threshold ótimo"""
    xgb_model = joblib.load(ART_XGB)
    lgbm_model = joblib.load(ART_LGBM)
    optimal_threshold = joblib.load(ART_THRESHOLD)
    feature_names = joblib.load(ART_FEATURES)
    
    # Retornar como um dicionário para compatibilidade com API
    ensemble = {
        "xgb": xgb_model,
        "lgbm": lgbm_model,
        "threshold": optimal_threshold,
        "type": "ensemble"
    }
    
    return ensemble, feature_names
