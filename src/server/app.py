import logging

from fastapi import FastAPI

from src.server.api import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

app = FastAPI(title="Fraud Detector API", version="1.0.0")
app.include_router(router)
