from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'/'ai4i2020.csv'
MODEL = ROOT/'models'/'failure_model.joblib'
FEATURES = ['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
TARGET = 'Machine failure'

def engineer(df):
    x = df[FEATURES].copy()
    x['Temperature difference [K]'] = x['Process temperature [K]']-x['Air temperature [K]']
    x['Estimated power [W]'] = x['Torque [Nm]']*x['Rotational speed [rpm]']*2*np.pi/60
    x['Wear torque interaction'] = x['Tool wear [min]']*x['Torque [Nm]']
    return x

def load_data():
    if not DATA.exists():
        raise FileNotFoundError('Dataset missing. Run python download_data.py first.')
    df = pd.read_csv(DATA)
    required = set(FEATURES+[TARGET])
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required-set(df.columns)}')
    return df
