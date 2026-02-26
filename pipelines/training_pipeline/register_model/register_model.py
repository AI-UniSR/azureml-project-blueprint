"""
Register Model component.

Registers the trained model in the Azure ML Model Registry via MLflow.
"""

import argparse
import json
import os

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--evaluation_output", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="blueprint-example-readmission-model")
    parser.add_argument("--register_output", type=str, required=True)
    args = parser.parse_args()

    # ---- Read evaluation report (log metrics for context) ------------------
    report_path = os.path.join(args.evaluation_output, "evaluation_report.json")
    with open(report_path) as f:
        metrics = json.load(f)
    print(f"Test metrics: {json.dumps(metrics, indent=2)}")

    # ---- Register model ----------------------------------------------------
    model_uri = f"file://{os.path.abspath(args.trained_model)}"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name,
    )
    print(f"Model registered: {registered.name} v{registered.version}")

    # ---- Save registration info --------------------------------------------
    os.makedirs(args.register_output, exist_ok=True)
    registration_info = {
        "registered": True,
        "model_name": registered.name,
        "model_version": registered.version,
        "test_metrics": metrics,
    }
    with open(os.path.join(args.register_output, "registration_info.json"), "w") as f:
        json.dump(registration_info, f, indent=2)

    print("Registration step complete.")


if __name__ == "__main__":
    main()
