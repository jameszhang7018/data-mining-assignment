from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel

# 全局客户数据模型
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.9,
                "TotalCharges": 2157.6
            }
        }

# 加载模型
model = joblib.load("churn_rf_model.joblib")
encoder = joblib.load("encoder.joblib")
scaler = joblib.load("scaler.joblib")

cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
num_cols = ["tenure","MonthlyCharges","TotalCharges"]

app = FastAPI(title="Telecom Customer Churn Prediction API")

@app.get("/health")
def health():
    return {"status":"service running normally, model loaded successfully"}

@app.post("/predict")
def predict(customer_data: CustomerData):
    try:
        data = customer_data.dict()
        df_input = pd.DataFrame([data])
        # Encode categorical features
        cat_data = encoder.transform(df_input[cat_cols])
        cat_df = pd.DataFrame(cat_data, columns=encoder.get_feature_names_out(cat_cols))
        # Scale numerical features
        num_data = scaler.transform(df_input[num_cols])
        num_df = pd.DataFrame(num_data, columns=num_cols)
        # Combine features
        input_x = pd.concat([num_df, cat_df], axis=1)
        # Predict result
        pred = model.predict(input_x)[0]
        pred_proba = model.predict_proba(input_x)[0][1]
        # Risk tier
        if pred_proba > 0.7:
            risk = "High Risk"
        elif pred_proba > 0.3:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"
        return {
            "churn_prediction": int(pred),
            "churn_probability": round(pred_proba,2),
            "risk_level": risk
        }
    except Exception as err:
        return {"error_detail": str(err)}, 500
