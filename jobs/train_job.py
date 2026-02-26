"""
Minimal standalone training job.

Reads the clinical readmission CSV, trains a Logistic Regression,
logs metrics and the model to MLflow, and saves the model to the
output folder.
"""

import argparse
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    args = parser.parse_args()

    # ---- Load data --------------------------------------------------------
    df = pd.read_csv(args.input_data)
    target_col = "readmitted"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # ---- Train with cross-validation --------------------------------------
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    # Fit on full training set
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    # ---- MLflow logging ----------------------------------------------------
    mlflow.log_metric("cv_roc_auc_mean", float(np.mean(cv_scores)))
    mlflow.log_metric("cv_roc_auc_std", float(np.std(cv_scores)))
    mlflow.log_metric("train_accuracy", float(accuracy_score(y, y_pred)))
    mlflow.log_metric("train_f1", float(f1_score(y, y_pred)))
    mlflow.log_metric("train_roc_auc", float(roc_auc_score(y, y_prob)))
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_samples", len(df))

    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    print(f"CV ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(classification_report(y, y_pred))

    # ---- Save model to output folder ---------------------------------------
    os.makedirs(args.output_model, exist_ok=True)
    model_path = os.path.join(args.output_model, "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
