"""
Evaluate component.

Loads the trained model and the test set, computes classification
metrics, logs them to MLflow, and writes a JSON report to the output
folder.
"""

import argparse
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--evaluation_output", type=str, required=True)
    args = parser.parse_args()

    # ---- Load model & data -------------------------------------------------
    model = joblib.load(os.path.join(args.trained_model, "model.pkl"))
    df = pd.read_csv(os.path.join(args.test_data, "test.csv"))

    target_col = "readmitted"
    feature_cols = [c for c in df.columns if c != target_col]

    X_test = df[feature_cols]
    y_test = df[target_col]
    print(f"Test set: {len(df)} rows")

    # ---- Predict -----------------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---- Metrics -----------------------------------------------------------
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # Log to MLflow
    for k, v in metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    print("=== Test Set Results ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred))

    # ---- Save report -------------------------------------------------------
    os.makedirs(args.evaluation_output, exist_ok=True)
    report_path = os.path.join(args.evaluation_output, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
