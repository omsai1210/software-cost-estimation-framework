import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split

def build_and_train_mlp(file_path):
    # 1. Load and preprocess data
    X, y = preprocess_data(file_path)

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scale the data (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Build the MLP Model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Output layer for regression
    ])

    # 5. Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 6. Train the model
    print("\n--- Training Deep Learning Model (MLP) ---")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=8,
        verbose=0 # Keep terminal clean
    )
    print("Training complete.")

    # 7. Evaluate on test data
    print("\n--- Evaluating Deep Learning Model ---")
    y_pred = model.predict(X_test_scaled).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Deep Learning MAE: {mae:.2f}")
    print(f"Deep Learning RMSE: {rmse:.2f}")

    return model, mae, rmse

if __name__ == "__main__":
    data_file = "software_projects_data.csv"
    try:
        build_and_train_mlp(data_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
