import pandas as pd
from src.common import engineer
from src.train import build_pipeline
from sklearn.linear_model import LogisticRegression

def test_engineer():
    x=engineer(pd.DataFrame([{'Type':'L','Air temperature [K]':300,'Process temperature [K]':310,'Rotational speed [rpm]':1500,'Torque [Nm]':40,'Tool wear [min]':100}]))
    assert x.iloc[0]['Temperature difference [K]']==10
    assert x.iloc[0]['Wear torque interaction']==4000

def test_pipeline_fit():
    rows=[]
    for i in range(12):
        rows.append({'Type':['L','M','H'][i%3],'Air temperature [K]':300+i,'Process temperature [K]':310+i,'Rotational speed [rpm]':1400+i*20,'Torque [Nm]':30+i,'Tool wear [min]':i*10})
    x=engineer(pd.DataFrame(rows));y=[0,1]*6
    p=build_pipeline(LogisticRegression(max_iter=500)).fit(x,y)
    assert p.predict_proba(x).shape==(12,2)
