"""
Training component.

Trains a Logistic Regression with stratified cross-validation.
Logs metrics and the model to MLflow, and saves the serialised
pipeline to the output folder.
"""

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--cv_n_splits", type=int, default=5)
    args = parser.parse_args()

    # ---- Load training data ------------------------------------------------
    train_path = os.path.join(args.training_data, "train.csv")
    df = pd.read_csv(train_path)
    target_col = "readmitted"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]
    print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")

    # ---- Build pipeline ----------------------------------------------------
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    # ---- Cross-validation --------------------------------------------------
    cv = StratifiedKFold(n_splits=args.cv_n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    mlflow.log_metric("cv_roc_auc_mean", float(np.mean(cv_scores)))
    mlflow.log_metric("cv_roc_auc_std", float(np.std(cv_scores)))
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_samples", len(df))

    print(f"CV ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # ---- Fit on full training set ------------------------------------------
    pipeline.fit(X, y)

    # Log model artifact with MLflow
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    # ---- Save to output uri_folder -----------------------------------------
    os.makedirs(args.trained_model, exist_ok=True)

    # Save as MLflow model format so register_model can use mlflow.register_model
    mlflow_model_path = os.path.join(args.trained_model, "mlflow_model")
    mlflow.sklearn.save_model(pipeline, path=mlflow_model_path)

    # Save feature list for downstream components
    with open(os.path.join(args.trained_model, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    print("Training complete.")


if __name__ == "__main__":
    main()
