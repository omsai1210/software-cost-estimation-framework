import pandas as pd
import numpy as np

def preprocess_data(file_path):
    # 1. Load the dataset
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # 2. Print first 5 rows and dataset info
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Dataset Info ---")
    df.info()

    # 3. Handle missing values
    print("\n--- Missing Values Count ---")
    missing_count = df.isnull().sum()
    print(missing_count)
    
    print("\nDropping rows with missing values...")
    df_cleaned = df.dropna()
    print(f"Rows after dropping NaCs: {len(df_cleaned)}")

    # 4. Handle duplicates
    print("\n--- Duplicate Rows ---")
    duplicate_count = df_cleaned.duplicated().sum()
    print(f"Number of duplicate rows found: {duplicate_count}")
    
    print("Removing duplicate rows...")
    df_final = df_cleaned.drop_duplicates()
    print(f"Final dataset row count: {len(df_final)}")

    # 5. Separate into Features (X) and Target (y)
    # Target: 'Target_Effort_Hours'
    # Features: Everything else (excluding Project_ID as it's just an identifier)
    print("\n--- Splitting Features and Target ---")
    target_col = 'Target_Effort_Hours'
    
    X = df_final.drop(columns=[target_col, 'Project_ID'])
    y = df_final[target_col]

    # 6. Print final shapes
    print(f"Final shape of Features (X): {X.shape}")
    print(f"Final shape of Target (y): {y.shape}")
    
    return X, y

if __name__ == "__main__":
    file_name = "software_projects_data.csv"
    try:
        X, y = preprocess_data(file_name)
    except FileNotFoundError:
        print(f"Error: {file_name} not found. Please run generate_sample_data.py first.")
