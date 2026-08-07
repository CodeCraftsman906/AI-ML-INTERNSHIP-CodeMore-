from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# --------------------------------------------------
# Load Model
# --------------------------------------------------

try:
    model = joblib.load("app/model.joblib")
except Exception as e:
    raise RuntimeError(f"Unable to load model: {e}")

# --------------------------------------------------
# Create FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Credit Card Prediction API",
    description="API for Credit Card Classification",
    version="1.0"
)

# --------------------------------------------------
# Input Schema
# Replace feature names if your dataset differs.
# --------------------------------------------------

class CreditCardInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Credit Card Prediction API is Running"
    }

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: CreditCardInput):

    try:
        import pandas as pd
        features = pd.DataFrame([[
            data.Time,
            data.V1,
            data.V2,
            data.V3,
            data.V4,
            data.V5,
            data.V6,
            data.V7,
            data.V8,
            data.V9,
            data.V10,
            data.V11,
            data.V12,
            data.V13,
            data.V14,
            data.V15,
            data.V16,
            data.V17,
            data.V18,
            data.V19,
            data.V20,
            data.V21,
            data.V22,
            data.V23,
            data.V24,
            data.V25,
            data.V26,
            data.V27,
            data.V28,
            data.Amount
        ]])

        prediction = model.predict(features)

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
