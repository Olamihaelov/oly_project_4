"""
FastAPI application with user, auth, and ML endpoints.
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import logging

import dal_users
from router_users import router as users_router
from router_predictions import router as predictions_router
from router_auth import router as auth_router

# Set up app logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('app')

# Initialize FastAPI app
app = FastAPI(title="🏃‍♂️ Running Time Prediction API")

# Create database tables on startup
@app.on_event("startup")
def startup():
    dal_users.create_table_users()
    logger.info("✅ Database initialized - System ready")
    for handler in logging.getLogger().handlers:
        handler.flush()

# Home page endpoint
@app.get("/")
def root():
    if os.path.exists("users.html"):
        return FileResponse("users.html", media_type="text/html")
    return {"message": "API Ready"}

# ML page endpoint
@app.get("/ml")
def ml_page():
    if os.path.exists("predictions.html"):
        return FileResponse("predictions.html", media_type="text/html")
    return {"message": "ML Predictions Interface"}

# Include all routers
app.include_router(users_router)
app.include_router(predictions_router)
app.include_router(auth_router)