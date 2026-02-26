"""
Generate a simple synthetic clinical dataset for demonstration.

Task: Binary classification of 30-day hospital readmission.
Features: age, bmi, systolic_bp, diastolic_bp, heart_rate, glucose,
          cholesterol, hemoglobin, num_prior_admissions, length_of_stay,
          is_smoker, has_diabetes.
Target:   readmitted (0/1).

Usage
-----
    python generate_synthetic_data.py              # writes data/clinical_readmission.csv
    python generate_synthetic_data.py --output_dir ./my_folder
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate_clinical_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic clinical DataFrame with realistic distributions."""
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=65, scale=12, size=n_samples).clip(18, 99).round(0)
    bmi = rng.normal(loc=27, scale=5, size=n_samples).clip(15, 50).round(1)
    systolic_bp = rng.normal(loc=130, scale=18, size=n_samples).clip(80, 200).round(0)
    diastolic_bp = rng.normal(loc=80, scale=10, size=n_samples).clip(50, 120).round(0)
    heart_rate = rng.normal(loc=75, scale=12, size=n_samples).clip(40, 130).round(0)
    glucose = rng.normal(loc=110, scale=30, size=n_samples).clip(60, 300).round(0)
    cholesterol = rng.normal(loc=200, scale=40, size=n_samples).clip(100, 350).round(0)
    hemoglobin = rng.normal(loc=13.5, scale=1.8, size=n_samples).clip(7, 18).round(1)
    num_prior_admissions = rng.poisson(lam=1.5, size=n_samples)
    length_of_stay = rng.exponential(scale=5, size=n_samples).clip(1, 30).round(0)
    is_smoker = rng.binomial(1, 0.25, size=n_samples)
    has_diabetes = rng.binomial(1, 0.30, size=n_samples)

    # Generate target with logistic model so it has realistic correlations
    logit = (
        -3.0
        + 0.02 * (age - 65)
        + 0.04 * (bmi - 27)
        + 0.01 * (glucose - 110)
        + 0.3 * num_prior_admissions
        + 0.05 * (length_of_stay - 5)
        + 0.4 * has_diabetes
        + 0.3 * is_smoker
    )
    prob = 1 / (1 + np.exp(-logit))
    readmitted = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "age": age.astype(int),
            "bmi": bmi,
            "systolic_bp": systolic_bp.astype(int),
            "diastolic_bp": diastolic_bp.astype(int),
            "heart_rate": heart_rate.astype(int),
            "glucose": glucose.astype(int),
            "cholesterol": cholesterol.astype(int),
            "hemoglobin": hemoglobin,
            "num_prior_admissions": num_prior_admissions,
            "length_of_stay": length_of_stay.astype(int),
            "is_smoker": is_smoker,
            "has_diabetes": has_diabetes,
            "readmitted": readmitted,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write the CSV file.",
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = generate_clinical_data(n_samples=args.n_samples, seed=args.seed)
    output_path = os.path.join(args.output_dir, "clinical_readmission.csv")
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    print(f"Readmission rate: {df['readmitted'].mean():.2%}")


if __name__ == "__main__":
    main()
