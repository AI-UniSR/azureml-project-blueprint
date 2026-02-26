# azureml-project-blueprint

End-to-end Azure Machine Learning reference project demonstrating **Azure ML CLI v2** workflows with **YAML-defined** data assets, environments, jobs, component-based pipelines, MLflow tracking, and model registration.

---

## Overview

`azureml-project-blueprint` is a reference implementation showing how to structure and manage an Azure Machine Learning project using **Azure ML CLI v2** and **declarative YAML configurations**.

The repository follows a clear separation between:

- **Assets and environments (infrastructure layer)** — data and environment definitions
- **Components (reusable computational steps)** — each in its own folder with YAML + Python
- **Pipelines (orchestration layer)** — a multi-step pipeline wired via YAML

A synthetic clinical dataset is included to demonstrate preprocessing, training, MLflow tracking, and model registration in a reproducible way.

---

## Compute Concepts in Azure ML

Azure ML provides two distinct types of compute, serving different purposes:

### Compute Instance — your personal development machine
A Compute Instance is a **managed cloud VM assigned to a single user**. You access it directly from **Azure ML Studio → Compute → Compute Instances**, where each team member has their own instance. From there you can open it with any of the available interfaces:

> JupyterLab · Jupyter · VS Code (Web) · VS Code (Desktop) · Terminal · Notebook (native Azure UI)

**For this guide we recommend VS Code** (Web or Desktop). The Compute Instance is used here purely as an **orchestrator**: you edit files and invoke the Azure ML CLI (`az ml job create …`) from its terminal. The actual training code never runs on it.

All Compute Instances in the same workspace share a **common filesystem** (`~/cloudfiles/`), so files written by one user are immediately visible to all others. This is where the project repository lives.

### Azure ML Compute Cluster ← *this is where the code runs*
A Compute Cluster is an **auto-scaling pool of VMs** managed entirely by Azure ML. When you submit a job it spins up the required nodes, runs the containerised workload, and scales back to zero when idle (no idle cost). Each job runs in a fresh environment defined by its `environment` YAML — no manual setup required.

When you see `compute: azureml:<YOUR-COMPUTE-CLUSTER>` in any YAML in this repo, that string references a Compute Cluster registered in your workspace. **You must set this field before submitting any job.**

---

## What This Project Demonstrates

| Capability | Where |
|---|---|
| Data asset registration and versioning via YAML | `data/` |
| Environment definition and reproducibility | `environments/` |
| Standalone command job | `jobs/` |
| Reusable CLI components | `pipelines/training_pipeline/*/` |
| Pipeline orchestration via YAML | `pipelines/training_pipeline/training_pipeline.yml` |
| MLflow experiment tracking | Inside every training / evaluation script |
| Model registration in Azure ML Model Registry | `pipelines/training_pipeline/register_model/` |

---

## Project Structure

```
azureml-project-blueprint/
├── data/
│   ├── generate_synthetic_data.py      # Generate synthetic clinical CSV
│   ├── clinical_readmission.csv        # Generated dataset (500 rows)
│   └── clinical_readmission.yml        # Data asset YAML (uri_file)
│
├── environments/
│   ├── Dockerfile                      # Base Docker image definition
│   ├── environment.yaml                # Conda env spec (scikit-learn, MLflow)
│   └── requirements.txt                # Pip dependencies
│
├── jobs/
│   ├── train_job.py                    # Standalone training script
│   └── train_job.yml                   # Command job YAML
│
├── pipelines/
│   └── training_pipeline/
│       ├── training_pipeline.yml       # Pipeline YAML (orchestration)
│       ├── data_prep/
│       │   ├── data_prep.py            # Stratified train/test split
│       │   └── data_prep.yml           # Component YAML
│       ├── train/
│       │   ├── train.py                # Logistic Regression + CV
│       │   └── train.yml               # Component YAML
│       ├── evaluate/
│       │   ├── evaluate.py             # Test-set evaluation & metrics
│       │   └── evaluate.yml            # Component YAML
│       └── register_model/
│           ├── register_model.py       # Model registration
│           └── register_model.yml      # Component YAML
│
└── README.md
```

---

## Prerequisites

### Azure setup
- An **Azure ML workspace** with access granted to your account
- A **Compute Instance** assigned to you (visible in Studio → Compute → Compute Instances)
- A **Compute Cluster** registered in the workspace — set its name in every YAML as `azureml:<YOUR-COMPUTE-CLUSTER>`
- **Azure CLI** with the `ml` extension installed on your Compute Instance (`az extension add -n ml`)

### Required skills
- **Git** — clone, commit, push, pull, branching
- **Python** — reading and writing scripts that use `argparse` for inputs/outputs
- **MLflow** — `log_param`, `log_metric`, `log_artifact` basics
- **YAML** — reading and editing structured configuration files
- **Azure ML CLI v2** — `az ml job create`, `az ml environment create`, `az ml data create`

---

## Workflow – Recommended Setup (Team Standard)

The steps below describe the recommended workflow, from connecting to the VM all the way to running jobs and pipelines.

---

### Step 1 — Connect to the Compute Instance with VS Code

1. Go to **Azure ML Studio → Compute → Compute instances**.
2. Select your compute instance.
3. If it is stopped, click **Start**.
4. In the **Applications** column:
   - Click **VS Code (Web)**, or  
   - Click the three dots `...` → **VS Code (Desktop)** if the desktop app is installed.

VS Code opens automatically and connects to the remote compute instance.

The integrated terminal runs directly on the VM (`azureuser@<vm-name>`), and the working directory  
`~/cloudfiles/code/Users/<username>/` is persistent and shared across sessions.

---

### Step 2 — Authenticate with Azure CLI

Open the integrated terminal in VS Code and run:

```bash
# Interactive login (opens the browser or uses device code flow)
az login

# Check the active subscription
az account show --output table

# If needed, select the correct subscription
az account set --subscription "<subscription-id>"
```

Verify that the `ml` extension is installed:

```bash
az extension show -n ml --query version -o tsv
# If missing:
az extension add -n ml
```

---

### Step 3 — Run a Standalone Job or a Pipeline

#### Option A — Standalone command job

A standalone job is useful for quick tests or one-off runs. Everything is defined in a single YAML that specifies inputs, outputs, environment, and command.

```bash
# 1. Register the environment (first time only, or when it changes)
az ml environment create --file environments/environment.yml

# 2. Register the data asset (first time only, or for new versions)
az ml data create --file data/clinical_readmission.yml

# 3. Submit the job
az ml job create --file jobs/train_job.yml
```

Monitor execution:

```bash
# Check status from the CLI
az ml job show --name <job-name> --query status -o tsv

# Stream logs in real time
az ml job stream --name <job-name>
```

The job also appears in **Azure ML Studio → Experiments → blueprint_standalone_job**.

---

#### Option B — Pipeline (recommended for structured workflows)

The pipeline splits the work into **reusable components**, each in its own folder with separate code and YAML. This enables **automatic caching**: Azure ML only re-runs components whose inputs or code have changed.

```bash
# 1. Register environment and data (if not already done)
az ml environment create --file environments/environment.yml
az ml data create --file data/clinical_readmission.yml

# 2. Submit the pipeline
az ml job create --file pipelines/training_pipeline/training_pipeline.yml
```

The pipeline runs 4 steps in sequence:

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | `data_prep` | Stratified train/test split (70/30) |
| 2 | `train` | Logistic Regression with cross-validation |
| 3 | `evaluate` | Test-set metrics (accuracy, F1, ROC-AUC) |
| 4 | `register_model` | Registers the model in the Model Registry |

Monitor the pipeline:

```bash
# Pipeline status
az ml job show --name <pipeline-job-name> --query status -o tsv

# Stream logs (includes child step logs)
az ml job stream --name <pipeline-job-name>
```

The pipeline graph is visible in **Azure ML Studio → Jobs → blueprint_training_pipeline**.

---

### Common Commands Reference

| Action | Command |
|--------|---------|
| Login | `az login` |
| Set defaults | `az configure --defaults group=<rg> workspace=<ws>` |
| Create environment | `az ml environment create --file environments/environment.yml` |
| Register data asset | `az ml data create --file data/clinical_readmission.yml` |
| Run standalone job | `az ml job create --file jobs/train_job.yml` |
| Run pipeline | `az ml job create --file pipelines/training_pipeline/training_pipeline.yml` |
| Job status | `az ml job show --name <name> --query status` |
| Stream logs | `az ml job stream --name <name>` |
| List registered models | `az ml model list --query "[].{name:name, version:version}" -o table` |

---

## Quick Start

```bash
# Authenticate and set defaults
az login
az configure --defaults group=<resource-group> workspace=<workspace-name>

# One-time setup
az ml environment create --file environments/environment.yml
az ml data create --file data/clinical_readmission.yml

# Run the pipeline
az ml job create --file pipelines/training_pipeline/training_pipeline.yml
```


## Key Design Decisions

| Decision | Rationale |
|---|---|
| **One folder per component** with separate `code: ./` | Enables Azure ML **component-level caching** — only re-runs when that component's code or inputs change |
| **`uri_folder` outputs** | Standard way to pass data between pipeline steps; compatible with data asset registration |
| **MLflow logging in every script** | Unified experiment tracking across standalone jobs and pipeline components |
| **Model registration as pipeline step** | Demonstrates the full lifecycle: train → evaluate → register |

---

## Customisation

- **Compute**: set `azureml:<YOUR-COMPUTE-CLUSTER>` in `jobs/train_job.yml` and `training_pipeline.yml` — this is workspace-specific and has no default
- **Environment version**: bump the `version` field in `environments/environment.yml` and each component YAML
- **Data**: swap `clinical_readmission.csv` with your own dataset; adapt `data_prep.py` accordingly
- **Model**: replace Logistic Regression with any scikit-learn estimator in `train/train.py`
