from fastapi import APIRouter
from app.models.prediction import PredictionRequest, PredictionResponse
from app.services.ai_service import get_prediction

router = APIRouter()

@router.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    prediction = await get_prediction(request)
    return PredictionResponse(prediction=prediction)