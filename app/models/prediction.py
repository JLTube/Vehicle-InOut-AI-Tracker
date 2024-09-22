from pydantic import BaseModel

class PredictionRequest(BaseModel):
    data: list

class PredictionResponse(BaseModel):
    prediction: dict