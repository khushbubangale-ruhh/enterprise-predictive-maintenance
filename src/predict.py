import joblib
from src.common import MODEL, engineer

def predict_one(record):
    if not MODEL.exists():
        raise FileNotFoundError('Model not trained. Run python -m src.train')
    import pandas as pd
    bundle=joblib.load(MODEL)  # Load only trusted artifacts created locally.
    x=engineer(pd.DataFrame([record]))
    probability=float(bundle['pipeline'].predict_proba(x)[0,1])
    return {'failure_probability':round(probability,4),'predicted_failure':int(probability>=bundle['threshold']),'decision_threshold':round(float(bundle['threshold']),4),'model':bundle['model_name']}
