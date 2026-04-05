import logging
import os
from typing import List, Optional

import joblib
import numpy as np
from sklearn.metrics import r2_score

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import get_current_user
from main import predict_from_model, train_and_save_model
import dal_users


logger = logging.getLogger('app')
router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    X: List[float]
    Y: List[float]
    degree: int = Field(default=2, ge=1, le=5, description="Polynomial degree (1-5)")


class PredictResponse(BaseModel):
    predicted_running_time: float
    predictions_remaining: Optional[int] = None


class PurchaseRequest(BaseModel):
    card_number: str = Field(min_length=13, max_length=19)
    expiry: str
    cvv: str


def get_model_filename(username: str) -> str:
    return f"{username}.joblib"


@router.post("/train")
def train_model(request: TrainRequest, current_user=Depends(get_current_user)):
    username = current_user['user_name']
    logger.info(f"📈 POST /train - Training polynomial regression for {username} (degree={request.degree}, points={len(request.X)})")
    try:
        user = dal_users.get_user_by_username(username)
        if not user:
            logger.warning(f"⚠️ POST /train - Account not found: {username}")
            raise HTTPException(status_code=404, detail="User not found. Account may have been deleted.")
        
        model_filename = get_model_filename(username)
        train_and_save_model(request.X, request.Y, model_filename, degree=request.degree)
        logger.info(f"✔️ POST /train - Model successfully trained for {username}")
        return {
            "message": f"Model trained and saved for user {username}",
            "model_name": model_filename,
            "degree": request.degree,
            "data_points": len(request.X)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"⚠️ POST /train - Training exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@router.delete("/model")
def delete_model(current_user=Depends(get_current_user)):
    username = current_user['user_name']
    model_filename = get_model_filename(username)
    
    if os.path.exists(model_filename):
        try:
            os.remove(model_filename)
            logger.info(f"� DELETE /model - Model removed for {username}")
            return {"message": "Model deleted"}
        except Exception as e:
            logger.error(f"⚠️ DELETE /model - Deletion failed for {username}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")
    else:
        logger.info(f"ℹ️ DELETE /model - No model found for {username}")
        return {"message": "No model to delete"}


@router.get("/predict/{hours}")
def predict_running_time(hours: float, current_user=Depends(get_current_user)):
    username = current_user['user_name']
    logger.info(f"🎯 GET /predict/{hours} - Making prediction for {username}")
    user = dal_users.get_user_by_username(username)
    if not user:
        logger.warning(f"⚠️ GET /predict/{hours} - Account not found: {username}")
        raise HTTPException(status_code=404, detail="User not found. Account may have been deleted.")
    
    model_filename = get_model_filename(username)
    if not os.path.exists(model_filename):
        logger.warning(f"⚠️ GET /predict/{hours} - Model not trained for {username}")
        raise HTTPException(status_code=404, detail="Model not found. Please train a model first.")
    
    predictions_left = user.get("predictions_remaining")
    if predictions_left is not None and predictions_left <= 0:
        logger.warning(f"⚠️ GET /predict/{hours} - Prediction credits exhausted for {username}")
        raise HTTPException(status_code=403, detail="No predictions remaining. Please purchase more credits.")
    
    try:
        prediction = predict_from_model(model_filename, hours)
    except Exception as e:
        logger.error(f"⚠️ GET /predict/{hours} - Prediction generation failed for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    dal_users.deduct_prediction(username)
    predictions_remaining = dal_users.get_predictions_remaining(username)
    rounded_prediction = round(prediction, 2)
    logger.info(f"✔️ GET /predict/{hours} - Prediction result: {rounded_prediction} hours for {username}")
    return {
        "predicted_running_time": rounded_prediction,
        "predictions_remaining": predictions_remaining
    }


@router.post("/purchase")
def purchase_predictions(request: PurchaseRequest, current_user=Depends(get_current_user)):
    username = current_user['user_name']
    logger.info(f"💳 POST /purchase - Processing credits purchase from {username}")
    user = dal_users.get_user_by_username(username)
    if not user:
        logger.warning(f"⚠️ POST /purchase - Account not found: {username}")
        raise HTTPException(status_code=404, detail="User not found. Account may have been deleted.")
    
    if not request.card_number or not request.expiry or not request.cvv:
        logger.warning(f"⚠️ POST /purchase - Invalid payment details from {username}")
        raise HTTPException(status_code=400, detail="Invalid card details")
    card_number_str = str(request.card_number)
    if len(card_number_str) < 13 or len(card_number_str) > 19:
        raise HTTPException(status_code=400, detail="Invalid card number format (13-19 digits)")
    if not "/" in request.expiry or len(request.expiry) != 5:
        raise HTTPException(status_code=400, detail="Invalid expiry format (use MM/YY)")
    if len(request.cvv) != 3 or not request.cvv.isdigit():
        raise HTTPException(status_code=400, detail="Invalid CVV (must be exactly 3 digits)")
    
    new_count = dal_users.add_predictions(username, 10)
    logger.info(f"✔️ POST /purchase - Credit purchase successful: 10 credits added for {username} (total: {new_count})")
    return {"message": "Payment successful. 10 predictions added.", "predictions_remaining": new_count}


@router.get("/accuracy")
def get_accuracy(current_user=Depends(get_current_user)):
    username = current_user['user_name']
    logger.info(f"📊 GET /accuracy - Calculating model accuracy for {username}")
    model_filename = get_model_filename(username)
    
    if not os.path.exists(model_filename):
        logger.warning(f"⚠️ GET /accuracy - Model not available for {username}")
        raise HTTPException(status_code=404, detail="Model not found. Please train a model first.")
    
    try:
        model = joblib.load(model_filename)
        X_test = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        y_test = np.array([2, 4, 6, 8, 10, 12, 14, 16])
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        r2_rounded = round(r2, 2)
        logger.info(f"✔️ GET /accuracy - Model performance score: {r2_rounded} for {username}")
        return {"message": f"{r2_rounded}", "note": "R² Score (0.0-1.0, closer to 1.0 is better)"}
    except Exception as e:
        logger.error(f"⚠️ GET /accuracy - Accuracy calculation failed for {username}: {str(e)}")
        return {"message": "Error", "note": f"Could not calculate score: {str(e)}"}
