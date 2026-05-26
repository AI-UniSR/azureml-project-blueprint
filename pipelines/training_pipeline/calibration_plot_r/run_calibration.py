"""
Calibration Plot (R) — wrapper.

Loads the trained sklearn model, generates predicted probabilities on the
test set, writes a predictions CSV, then invokes Rscript for the
publication-quality calibration plot. Logs the resulting PNG to MLflow.
"""

import argparse
import os
import subprocess

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--calibration_output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.calibration_output, exist_ok=True)

    # Load model
    model_path = os.path.join(args.trained_model, "model.pkl")
    model = joblib.load(model_path)

    # Load test data
    test_path = os.path.join(args.test_data, "test.csv")
    df = pd.read_csv(test_path)
    target_col = "readmitted"
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    # Predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Save predictions CSV for R
    predictions_path = os.path.join(args.calibration_output, "predictions.csv")
    pd.DataFrame({"y_true": y_test, "y_prob": y_prob}).to_csv(
        predictions_path, index=False
    )

    # Run R script (handles MLflow logging internally via reticulate)
    r_script = os.path.join(os.path.dirname(__file__), "calibration_plot.R")
    subprocess.run(
        [
            "Rscript", r_script,
            "--predictions", predictions_path,
            "--output_dir", args.calibration_output,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
