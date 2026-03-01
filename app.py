from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    contract_type: int

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API Running"}

@app.post("/predict")
def predict(data: CustomerData):

    input_data = np.array([[data.tenure,
                            data.monthly_charges,
                            data.contract_type]])

    prediction = model.predict_proba(input_data)[0][1]

    if prediction > 0.7:
        risk = "High Risk"
    elif prediction > 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "churn_probability": float(prediction),
        "prediction": risk
    }