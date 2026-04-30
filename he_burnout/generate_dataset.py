"""
generate_dataset.py
Generates a synthetic Developer Burnout dataset (7000 samples)
matching the schema of: kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 7000

def generate():
    employee_ids = [f"EMP_{i:05d}" for i in range(1, N+1)]
    genders = np.random.choice(["Male", "Female", "Non-binary"], N, p=[0.55, 0.38, 0.07])
    company_types = np.random.choice(["Product", "Service"], N, p=[0.55, 0.45])
    wfh = np.random.choice(["Yes", "No"], N, p=[0.6, 0.4])
    designations = np.random.randint(1, 6, N)           # 1=Junior ... 5=Senior/Lead
    resource_alloc = np.round(np.random.uniform(1.0, 10.0, N), 1)
    mental_fatigue = np.round(np.random.uniform(0.0, 10.0, N), 2)
    hours_per_week = np.random.randint(30, 70, N)
    years_exp = np.random.randint(1, 20, N)
    team_size = np.random.randint(3, 30, N)

    # burn_rate: target variable in [0,1], correlated with fatigue & resource alloc
    noise = np.random.normal(0, 0.05, N)
    burn_rate = np.clip(
        0.05 * designations + 0.04 * resource_alloc + 0.06 * (mental_fatigue / 10)
        + 0.003 * hours_per_week - 0.008 * years_exp + noise,
        0.0, 1.0
    )
    burn_rate = np.round(burn_rate, 4)

    df = pd.DataFrame({
        "employee_id": employee_ids,
        "gender": genders,
        "company_type": company_types,
        "wfh_setup_available": wfh,
        "designation": designations,
        "resource_allocation": resource_alloc,
        "mental_fatigue_score": mental_fatigue,
        "hours_per_week": hours_per_week,
        "years_experience": years_exp,
        "team_size": team_size,
        "burn_rate": burn_rate,
    })
    return df

if __name__ == "__main__":
    df = generate()
    out = os.path.join(os.path.dirname(__file__), "developer_burnout_dataset.csv")
    df.to_csv(out, index=False)
    print(f"[✓] Dataset saved → {out}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(df.describe())
