from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from data_preprocessing import preprocess_data

def cocomo_organic(kloc):
    """Calculates effort using the Organic COCOMO formula: Effort = 2.4 * (KLOC) ** 1.05"""
    return 2.4 * (kloc ** 1.05)

def train_and_evaluate(file_path):
    # 1. Load and preprocess data
    X, y = preprocess_data(file_path)

    # 2. Split into training and testing sets (80% train, 20% test)
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size: {X_test.shape[0]} rows")

    # 3. Initialize and train RandomForestRegressor
    print("\n--- Training Random Forest Model ---")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Random Forest training complete.")

    # 4. Make RF Predictions
    print("\n--- Evaluating Models ---")
    rf_pred = rf_model.predict(X_test)

    # Calculate RF metrics
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    # 5. Traditional COCOMO Implementation
    # Simulate KLOC using Complexity_Score * random scaling factor (2 to 10)
    # We use a fixed seed for reproducibility in simulation
    np.random.seed(42)
    scaling_factors = np.random.uniform(2, 10, size=len(X_test))
    kloc_simulated = X_test['Complexity_Score'] * scaling_factors
    
    # Generate COCOMO predictions (convert effort to hours assuming person-months to hours conversion if needed)
    # The formula usually gives Person-Months. Assuming 1 PM = 152 hours (standard)
    # Actually, the user asked for simple COCOMO output compared to Target_Effort_Hours.
    # Let's see if we need to scale it to hours or if Target_Effort_Hours is comparable to COCOMO PM.
    # Given Target_Effort_Hours values like ~2000, and KLOC ~ 30, COCOMO PM would be ~90.
    # 90 * 152 = 13,680. That's a bit high. 
    # Let's assume Target_Effort_Hours is what the user wants to compare against directly.
    # Wait, the user said "represent lines of code" and "calculate effort".
    # I'll stick to the formula and see current results.
    
    cocomo_pred_pm = np.array([cocomo_organic(k) for k in kloc_simulated])
    # Multiplier to reach realistic 'Hours' if needed, but I'll stick to the base formula first.
    # However, Target_Effort_Hours is in hundreds/thousands.
    # Let's use a standard conversion (1 PM = 160 hours) to make it comparable.
    cocomo_pred_hours = cocomo_pred_pm * 160 

    cocomo_mae = mean_absolute_error(y_test, cocomo_pred_hours)
    cocomo_rmse = np.sqrt(mean_squared_error(y_test, cocomo_pred_hours))

    # 6. Print Comparison Results
    print("\n" + "="*50)
    print(f"{'Metric':<10} | {'Random Forest':<15} | {'COCOMO (Organic)':<15}")
    print("-" * 50)
    print(f"{'MAE':<10} | {rf_mae:>15.2f} | {cocomo_mae:>15.2f}")
    print(f"{'RMSE':<10} | {rf_rmse:>15.2f} | {cocomo_rmse:>15.2f}")
    print("="*50)

    return rf_model, rf_mae, rf_rmse

if __name__ == "__main__":
    data_file = "software_projects_data.csv"
    try:
        train_and_evaluate(data_file)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
