import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data
from ml_model_training import train_and_evaluate, cocomo_organic
from deep_learning_model import build_and_train_mlp

def run_hybrid_ensemble(file_path):
    # 1. Get metrics and predictions from RF (and COCOMO)
    print("--- Gathering Random Forest and COCOMO results ---")
    rf_model, rf_mae, rf_rmse = train_and_evaluate(file_path)
    
    # 2. Get metrics and predictions from Deep Learning
    print("\n--- Gathering Deep Learning results ---")
    dl_model, dl_mae, dl_rmse = build_and_train_mlp(file_path)

    # 3. To get the actual predictions for the ensemble, we need the test set
    X, y = preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RF Predictions
    rf_preds = rf_model.predict(X_test)

    # DL Predictions (need scaling)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    dl_preds = dl_model.predict(X_test_scaled).flatten()

    # 4. Hybrid Ensemble Logic
    print("\n--- Computing Hybrid Ensemble (0.95 RF + 0.05 MLP) ---")
    hybrid_preds = (0.95 * rf_preds) + (0.05 * dl_preds)

    hybrid_mae = mean_absolute_error(y_test, hybrid_preds)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_preds))

    # 5. Traditional COCOMO (for leaderboard)
    # Re-calculate to ensure consistency
    np.random.seed(42)
    scaling_factors = np.random.uniform(2, 10, size=len(X_test))
    kloc_simulated = X_test['Complexity_Score'] * scaling_factors
    cocomo_pred_pm = np.array([cocomo_organic(k) for k in kloc_simulated])
    cocomo_pred_hours = cocomo_pred_pm * 160
    cocomo_mae = mean_absolute_error(y_test, cocomo_pred_hours)
    cocomo_rmse = np.sqrt(mean_squared_error(y_test, cocomo_pred_hours))

    # 6. Final Leaderboard
    models = [
        {"Model": "Traditional COCOMO", "MAE": cocomo_mae, "RMSE": cocomo_rmse},
        {"Model": "Deep Learning (alone)", "MAE": dl_mae, "RMSE": dl_rmse},
        {"Model": "Random Forest (alone)", "MAE": rf_mae, "RMSE": rf_rmse},
        {"Model": "Hybrid Ensemble", "MAE": hybrid_mae, "RMSE": hybrid_rmse}
    ]

    leaderboard = pd.DataFrame(models).sort_values(by="MAE")
    
    print("\n" + "="*60)
    print("FINAL MODEL LEADERBOARD")
    print("="*60)
    print(leaderboard.to_string(index=False))
    print("="*60)

    return leaderboard

if __name__ == "__main__":
    data_file = "software_projects_data.csv"
    try:
        run_hybrid_ensemble(data_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
