# azureml-project-blueprint
End-to-end Azure Machine Learning reference project demonstrating **Azure ML CLI v2** workflows with **YAML-defined** data assets, environments, jobs, component-based pipelines, MLflow tracking, and model registration.

# Overview
`azureml-project-blueprint` is a reference implementation showing how to structure and manage an Azure Machine Learning project using **Azure ML CLI v2** and **declarative YAML configurations**.

The repository follows a clear separation between:

- **Assets and environments (infrastructure layer)** — data and environment definitions
- **Components (reusable computational steps)** — each in its own folder with YAML + Python
- **Pipelines (orchestration layer)** — a multi-step pipeline wired via YAML

A synthetic clinical dataset is included to demonstrate preprocessing, training, MLflow tracking, and model registration in a reproducible way.

# Compute Concepts in Azure ML
Azure ML provides two distinct types of compute, serving different purposes:

## Compute instance — your personal development machine
A Compute instance is a **managed cloud VM assigned to a single user**. You access it directly from **Azure ML Studio → Compute → Compute instances**, where each team member has their own instance. From there you can open it with any of the available interfaces:

> JupyterLab · Jupyter · VS Code (Web) · VS Code (Desktop) · Terminal · Notebook (native Azure UI)

**For this guide we recommend VS Code** (Web or Desktop). The Compute Instance is used here purely as an **orchestrator**: you edit files and invoke the Azure ML CLI (`az ml job create …`) from its terminal. The actual training code never runs on it.

All Compute Instances in the same workspace share a **common filesystem** (`~/cloudfiles/`), so files written by one user are immediately visible to all others. This is where the project repository lives.

## Compute cluster ← *this is where the code from this guide runs*
A compute cluster is an **auto-scaling pool of VMs** managed entirely by Azure ML. When you submit a job it spins up the required nodes, runs the containerised workload, and scales back to zero when idle. Each job runs in a fresh environment defined by its `environment` YAML — no manual setup required.

🛎️ When you see `compute: azureml:<YOUR-COMPUTE-CLUSTER>` in any YAML in this repo, that string references a compute cluster registered in your workspace. **You must set this field before submitting any job**: go to Azure ML Studio → Compute → Compute clusters, and copy the name.

e.g. `clusterprdwe-wldtii` on the UI → `compute: azureml:clusterprdwe-wldtii` on the YAML

# What this project demonstrates
| Capability | Where |
|---|---|
| Data asset registration and versioning via YAML | `data/` |
| Environment definition and reproducibility | `environments/` |
| Standalone command job | `jobs/` |
| Reusable CLI components | `pipelines/training_pipeline/*/` |
| Pipeline orchestration via YAML | `pipelines/training_pipeline/training_pipeline.yml` |
| MLflow experiment tracking | Inside every training / evaluation script |
| Model registration in Azure ML Model Registry | `pipelines/training_pipeline/register_model/` |

# Prerequisites
## Azure setup
- An **Azure ML workspace** with access granted to your account
- A **Compute Instance** assigned to you (visible in Azure ML Studio → Compute → Compute instances)
- A **Compute Cluster** registered in the workspace — set its name in every YAML as `compute: azureml:<YOUR-COMPUTE-CLUSTER>`
- **Azure CLI** with the `ml` extension installed on your compute instance (should be present by default; else run `az extension add -n ml`)

## Mandatory skills
- **Python** — reading and writing scripts that use `argparse` for inputs/outputs
- **YAML** — reading and editing structured configuration files

## Nice-to-have skills
- **Azure ML CLI v2** — `az ml job create`, `az ml environment create`, `az ml data create`
- **MLflow** — `log_param`, `log_metric`, `log_artifact` basics
- **Git** — clone, commit, push, pull, branching

# Workflow – Recommended setup (team standard)
The steps below describe the recommended workflow, from connecting to the VM all the way to running jobs and pipelines.

## Step 1 — Connect to the Compute instance with VS Code
1. Go to **Azure ML Studio → Compute → Compute instances**.
2. Select your compute instance.
3. If it is stopped, click **Start**.
4. In the **Applications** column:
   - Click **VS Code (Web)**, or  
   - Click the three dots `...` → **VS Code (Desktop)** if the desktop app is installed.

VS Code opens automatically and connects to the remote compute instance.

The integrated terminal runs directly on the VM (`azureuser@<vm-name>`), and the working directory  
`~/cloudfiles/code/Users/<username>/` is persistent and shared across sessions.

## Step 2 — Authenticate with Azure CLI
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

## Step 3 — Register the environment (one-time or on change)
### Step 3a — Environment for compute cluster
The environment created with the following command will have the following characteristics:

- It will be available for remote jobs on the compute cluster.
- It will not be directly usable on the compute instance.
- It will be shared among all users of the workspace.

```bash
az ml environment create --file environments/blueprint-env/env.yml
```

Alternatively, right-click `environments/blueprint-env/env.yml` in VS Code and select **Azure ML: Execute YAML**.

The creation of the environment for remote jobs leverages the compute cluster itself: you can monitor the build progress in **Azure ML Studio → Jobs → prepare_image**.

If you wish to update an environment that has already been registered, edit `requirements.txt` and `environments/blueprint-env/env.yml`, bumping the version (e.g., `version: 1` to `version: 2`), and run the above command.

The environment appears in **Azure ML Studio → Environments → blueprint-env**.

### Step 3b (optional) — Environment for compute instance

Feel free to skip this step, as the current guide leverages the computation of a compute cluster, with the compute instance being used solely for orchestration (there is no need for a specific environment on the compute instance). However, it is sometimes useful to develop or debug on the compute instance first, using the same environment that will be used later on the compute cluster. 

If you wish to obtain the same environment, as well as the linked Jupyter kernel on your compute instance, run the shell script in the environments directory:

```bash
chmod +x environments/build_kernel_vm.sh
./environments/build_kernel_vm.sh blueprint-env
```

After the script completes:

1. **Reload VS Code** — press `CTRL+SHIFT+P`, type and select **Developer: Reload Window**.
2. **Select the interpreter** — press `CTRL+SHIFT+P`, type and select **Python: Select Interpreter**.  
   The interpreter **blueprint-env** should appear automatically in the list — select it.

Note that the resulting Python environment and Jupyter kernel are linked to your compute instance, so they are not shared among all users in the workspace.

## Step 4 — Register the data asset
In a real project, you would typically source data from an API call or a data lake. For simplicity, this blueprint works with a synthetic CSV dataset that you generate locally on your compute instance — no special environment is needed:

```bash
python data/generate_synthetic_data.py
```

A **data asset** is a named, versioned reference registered in the Azure ML workspace. It does not store a copy of the data — it stores a **pointer** to an underlying object:

| Asset type | Points to |
|---|---|
| `uri_file` | A single file (e.g. a CSV) |
| `uri_folder` | A directory |

In this project we register a `uri_file` asset pointing to the generated CSV.

**Key advantages:**
- **Versioning & reproducibility** — you can bump the version each time the data changes and always re-run an experiment against the exact version it was trained on.
- **Compact consumption code** — both compute cluster jobs and compute instance scripts can load the asset by name and version, instead of hardcoding paths.

🛎️ A data asset is just a convenient pointer. If you delete or overwrite the underlying file (e.g. the CSV), the asset will stop working. Always treat the underlying object as immutable once an asset version references it.

Create the actual data asset:

```bash
az ml data create --file data/clinical_readmission.yml
```

Alternatively, right-click `data/clinical_readmission.yml` in VS Code and select **Azure ML: Execute YAML**.
The data asset appears in **Azure ML Studio → Data → blueprint-clinical-readmission**.

## Step 5 — Run a standalone job or a pipeline
### Option A — Standalone command job
A standalone job is useful for quick tests or one-off runs. Everything is defined in a single YAML that specifies inputs, outputs, environment, and command.

```bash
az ml job create --file jobs/train_job.yml
```

Monitor execution in **Azure ML Studio → Experiments → blueprint_standalone_job**.

### Option B — Pipeline (recommended for structured workflows)
The pipeline splits the work into **reusable components**, each in its own folder with separate code and YAML.

The pipeline in this repository runs 4 steps in sequence:

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | `data_prep` | Stratified train/test split (70/30) |
| 2 | `train` | Logistic Regression with cross-validation |
| 3 | `evaluate` | Test-set metrics (accuracy, F1, ROC-AUC) |
| 4 | `register_model` | Registers the model in the Model Registry |

Key design decisions:

| Decision | Rationale |
|---|---|
| **One folder per component** with separate `code: ./` | Enables Azure ML **component-level caching** — only re-runs when that component's code or inputs change |
| **`uri_folder` outputs** | Standard way to pass data between pipeline steps; compatible with data asset registration |
| **MLflow logging in every script** | Unified experiment tracking across standalone jobs and pipeline components |
| **Model registration as pipeline step** | Demonstrates the full lifecycle: train → evaluate → register |

Execute the pipeline:
```bash
az ml job create --file pipelines/training_pipeline/training_pipeline.yml
```

Monitor execution in **Azure ML Studio → Experiments → blueprint_training_pipeline**.

# Customisation

- **Compute**: set `azureml:<YOUR-COMPUTE-CLUSTER>` in `jobs/train_job.yml` and `pipelines/training_pipeline/training_pipeline.yml` — this is workspace-specific and has no default
- **Environment version**: bump the `version` field in `environments/blueprint-env/env.yml` and each component YAML
- **Data**: swap `clinical_readmission.csv` with your own dataset; adapt `data_prep.py` accordingly
- **Model**: replace Logistic Regression with any scikit-learn estimator in `train/train.py`
