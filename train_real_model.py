import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import the new Scikit-Learn models for the Stacking Regressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

def train_and_save():
    os.makedirs('saved_models', exist_ok=True)
    
    csv_file = 'china_clean.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found! Please run download_china_data.py first.")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Feature engineering: matching FPA inputs
    features = ['Input', 'Output', 'Enquiry', 'File', 'Interface']
    target = 'Effort'

    print("Preprocessing data...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df[features])
    y = df[target].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Define Base Estimators
    # We use make_pipeline for MLP so that inputs are scaled naturally within the stacking framework
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('mlp', make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)))
    ]

    # 2. Define Final Meta-Estimator
    final_estimator = Ridge()

    # 3. Construct the Stacking Regressor
    stack_reg = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5
    )

    print("Training Stacking Regressor (Smart Hybrid) on 5-Fold Cross Validation...")
    # This automatically fits all base estimators and the meta-estimator!
    stack_reg.fit(X_train, y_train)

    # Note: We save the 'imputer' so the API can still handle raw incoming inputs if they happen to have NaNs (though our frontend forces required)
    joblib.dump(imputer, 'saved_models/imputer.joblib')
    
    # Save the huge meta-model
    joblib.dump(stack_reg, 'saved_models/smart_hybrid_model.joblib')

    print("✅ Training complete! Smart Hybrid Stacking Regressor saved perfectly as 'smart_hybrid_model.joblib'.")

if __name__ == "__main__":
    train_and_save()
