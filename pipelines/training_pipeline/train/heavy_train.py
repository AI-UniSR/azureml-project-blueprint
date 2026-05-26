"""
Heavy Training component — Random Forest + Optuna hyperparameter tuning.

Dependencies (add to the conda env before using this component):
  - optuna==3.6.1      # latest 3.x stable; or pin to optuna==4.x if available
    pip install optuna==3.6.1
  Suggested env: corvaglia-blueprint-example-sklearn-env-optuna
  (copy conda_dependencies.yaml and append `- optuna==3.6.1` under pip)

Usage: python heavy_train.py --training_data <path> --trained_model <path>
         [--n_trials 50] [--cv_n_splits 5] [--timeout 600]
"""

import argparse
import json
import os

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Silence Optuna's per-trial INFO logs (MLflow already captures metrics)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_objective(X, y, cv_n_splits: int):
    """Return an Optuna objective function closed over the dataset."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state": 42,
            "n_jobs": -1,
        }

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(**params)),
            ]
        )

        cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(np.mean(scores))

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Optuna search timeout in seconds (0 = no limit).")
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

    # ---- Optuna study ------------------------------------------------------
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = build_objective(X, y, args.cv_n_splits)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_score = study.best_value
    print(f"Best ROC-AUC (CV): {best_score:.4f}")
    print(f"Best params: {best_params}")

    # ---- Log Optuna results to MLflow -------------------------------------
    mlflow.log_metric("optuna_best_cv_roc_auc", best_score)
    mlflow.log_metric("optuna_n_trials", len(study.trials))
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_samples", len(df))
    for k, v in best_params.items():
        mlflow.log_param(f"best_{k}", v)

    # ---- Fit final model with best params on full training set ------------
    best_rf_params = {k: v for k, v in best_params.items()}
    best_rf_params["random_state"] = 42
    best_rf_params["n_jobs"] = -1

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**best_rf_params)),
        ]
    )
    final_pipeline.fit(X, y)

    # Log model artifact with MLflow
    mlflow.sklearn.log_model(final_pipeline, artifact_path="model")

    # ---- Save to output uri_folder -----------------------------------------
    os.makedirs(args.trained_model, exist_ok=True)
    joblib.dump(final_pipeline, os.path.join(args.trained_model, "model.pkl"))

    # Save in MLflow model format for registration
    mlflow.sklearn.save_model(final_pipeline, path=os.path.join(args.trained_model, "mlflow_model"))

    # Save feature list and tuning summary for downstream components
    with open(os.path.join(args.trained_model, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    tuning_summary = {
        "best_cv_roc_auc": best_score,
        "n_trials_completed": len(study.trials),
        "best_params": best_params,
    }
    with open(os.path.join(args.trained_model, "tuning_summary.json"), "w") as f:
        json.dump(tuning_summary, f, indent=2)

    print("Heavy training complete.")


if __name__ == "__main__":
    main()
