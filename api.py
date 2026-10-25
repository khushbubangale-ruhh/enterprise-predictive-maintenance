from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from src.predict import predict_one
app=FastAPI(title='Predictive Maintenance API',version='1.0')
class MachineInput(BaseModel):
    type:Literal['L','M','H']
    air_temperature:float=Field(ge=250,le=400)
    process_temperature:float=Field(ge=250,le=420)
    rotational_speed:int=Field(ge=0,le=10000)
    torque:float=Field(ge=0,le=500)
    tool_wear:int=Field(ge=0,le=1000)
@app.get('/health')
def health():return {'status':'ok'}
@app.post('/predict')
def predict(data:MachineInput):
    record={'Type':data.type,'Air temperature [K]':data.air_temperature,'Process temperature [K]':data.process_temperature,'Rotational speed [rpm]':data.rotational_speed,'Torque [Nm]':data.torque,'Tool wear [min]':data.tool_wear}
    try:return predict_one(record)
    except FileNotFoundError as exc:raise HTTPException(status_code=503,detail=str(exc)) from exc
