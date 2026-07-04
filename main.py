from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

# Initialise API
app = FastAPI(title="Telecom Customer Churn Prediction API")

# Load saved model & preprocessing tools
model = joblib.load("churn_rf_model.joblib")
encoder = joblib.load("encoder.joblib")
scaler = joblib.load("scaler.joblib")
cat_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
num_cols = ['tenure','MonthlyCharges','TotalCharges']

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "service running normally, model loaded successfully"}

# Single customer prediction endpoint
@app.post("/predict")
def predict(customer_data: dict):
    try:
        # 下面全部是原来的业务代码，原样保留，整体缩进一层
        df_input = pd.DataFrame([customer_data])
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
        # 报错时返回详细错误信息，不再只显示 Internal Server Error
        return {"error_detail": str(err)}, 500
