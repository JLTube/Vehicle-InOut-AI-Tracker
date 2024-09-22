# Entry point for the FastAPI application

from fastapi import FastAPI
from app.api.endpoints import predictions, health

app = FastAPI()

app.include_router(predictions.router)
app.include_router(health.router)