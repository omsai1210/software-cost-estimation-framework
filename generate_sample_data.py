import pandas as pd
import numpy as np
import random

def generate_data(n_rows=50):
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Realistic ranges for software projects
    project_ids = [f"PRJ_{1000 + i}" for i in range(n_rows)]
    team_exp = np.random.randint(2, 15, size=n_rows)
    mgr_exp = np.random.randint(5, 25, size=n_rows)
    proj_length = np.random.randint(3, 24, size=n_rows)
    complexity = np.random.randint(1, 11, size=n_rows)
    
    # Calculate Target Effort Hours with a realistic formula + noise
    # Base effort: 100 hours per month * Complexity
    # Modified by team/manager experience (more exp = less effort)
    base_effort = proj_length * 160 # 160 hours per month
    exp_factor = (15 - team_exp) * 10 + (25 - mgr_exp) * 5
    complexity_factor = complexity * 50
    
    target_effort = base_effort + exp_factor + complexity_factor
    # Add random noise (+/- 10%)
    noise = np.random.uniform(0.9, 1.1, size=n_rows)
    target_effort = (target_effort * noise).astype(int)

    data = {
        'Project_ID': project_ids,
        'Team_Experience_Years': team_exp,
        'Manager_Experience_Years': mgr_exp,
        'Project_Length_Months': proj_length,
        'Complexity_Score': complexity,
        'Target_Effort_Hours': target_effort
    }

    df = pd.DataFrame(data)

    # Introduce missing values (NaN) - ~5 values across random columns
    for _ in range(5):
        row = np.random.randint(0, n_rows)
        col = np.random.choice(df.columns[1:]) # Don't mask Project_ID
        df.loc[row, col] = np.nan

    # Introduce duplicates - 2 rows
    dup1 = df.iloc[0:1]
    dup2 = df.iloc[10:11]
    df = pd.concat([df, dup1, dup2], ignore_index=True)

    # Shuffle to mix duplicates
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = generate_data()
    file_path = "software_projects_data.csv"
    df.to_csv(file_path, index=False)
    print(f"Dataset generated successfully: {file_path}")
    print(f"Total rows: {len(df)}")
    print(f"Missing values count:\n{df.isnull().sum()}")
    print(f"Duplicate rows count: {df.duplicated().sum()}")
