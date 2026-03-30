from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Smart Hybrid Ensemble API")

# Setup CORS to communicate with React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectParams(BaseModel):
    extInputs: float
    extOutputs: float
    extInquiries: float
    intLogFiles: float
    extInterfaces: float

# Attempt to load the trained Stacking Regressor models globally
try:
    imputer = joblib.load('saved_models/imputer.joblib')
    smart_hybrid_model = joblib.load('saved_models/smart_hybrid_model.joblib')
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Models could not be loaded. Please run train_real_model.py first. Error: {e}")
    MODELS_LOADED = False


def calculate_cocomo(inputs, outputs, inquiries, files, interfaces):
    # Calculate Unadjusted Function Points (UFP) using standard average weights
    ufp = (inputs * 4) + (outputs * 5) + (inquiries * 4) + (files * 10) + (interfaces * 7)
    
    # Convert UFP to KLOC (Assuming modern language like Java/C# where 1 FP = 50 LOC)
    loc = ufp * 50
    kloc = loc / 1000
    
    # Calculate COCOMO Person-Months (Organic Model)
    person_months = 2.4 * (kloc ** 1.05)
    
    # Convert Person-Months to Hours (Assuming 152 working hours per month)
    cocomo_hours = person_months * 152
    
    return cocomo_hours


@app.post("/predict")
def predict_effort(params: ProjectParams):
    cocomo_est = calculate_cocomo(
        params.extInputs, 
        params.extOutputs, 
        params.extInquiries, 
        params.intLogFiles, 
        params.extInterfaces
    )

    if not MODELS_LOADED:
        return {
            "cocomo_estimate": round(cocomo_est),
            "ai_estimate": 0,
            "error": "Models not trained yet. Run train_real_model.py to enable AI Hybrid Prediction."
        }

    # Prepare input vector
    input_features = np.array([[
        params.extInputs, 
        params.extOutputs, 
        params.extInquiries, 
        params.intLogFiles, 
        params.extInterfaces
    ]])
    
    input_imputed = imputer.transform(input_features)

    # 1. Smart Stacking Prediction
    # The StackingRegressor handles passing the inputs internally to RF, GB, and MLP, then into Ridge!
    hybrid_pred = smart_hybrid_model.predict(input_imputed)[0]

    return {
        "cocomo_estimate": round(cocomo_est),
        "ai_estimate": round(hybrid_pred)
    }

# Run with: uvicorn api:app --reload
