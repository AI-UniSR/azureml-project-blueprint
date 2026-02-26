"""
Data preparation component.

Reads the raw CSV, performs basic cleaning (no imputation needed for synthetic
data, but shows the pattern), and creates a stratified train/test split.
Outputs are written as CSVs inside uri_folder outputs.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.3)
    args = parser.parse_args()

    # ---- Load ---------------------------------------------------------------
    df = pd.read_csv(args.raw_data)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    target_col = "readmitted"

    # ---- Basic cleaning (pattern) ------------------------------------------
    # Drop rows with any NaN (synthetic data has none, but keeps pattern)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before - len(df)} rows with missing values")

    # ---- Stratified split ---------------------------------------------------
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=42
    )
    train_idx, test_idx = next(splitter.split(df, df[target_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train set: {len(train_df)} rows  (readmission rate: {train_df[target_col].mean():.2%})")
    print(f"Test  set: {len(test_df)} rows  (readmission rate: {test_df[target_col].mean():.2%})")

    # ---- Save to output folders --------------------------------------------
    os.makedirs(args.training_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_df.to_csv(os.path.join(args.training_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
