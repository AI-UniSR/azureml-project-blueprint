# azureml-project-blueprint

End-to-end Azure Machine Learning reference project demonstrating **Azure ML CLI v2** workflows with **YAML-defined** data assets, environments, jobs, component-based pipelines, MLflow tracking, and model registration.

---

## Overview

`azureml-project-blueprint` is a reference implementation showing how to structure and manage an Azure Machine Learning project using **Azure ML CLI v2** and **declarative YAML configurations**.

The repository follows a clear separation between:

- **Pipelines (orchestration layer)** — a multi-step pipeline wired via YAML  
- **Components (reusable computational steps)** — each in its own folder with YAML + Python  
- **Assets and environments (infrastructure layer)** — data and environment definitions  

A synthetic clinical dataset is included to demonstrate preprocessing, training, MLflow tracking, and model registration in a reproducible way.

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
│   ├── conda_dependencies.yaml         # Conda env spec (scikit-learn, MLflow)
│   └── environment.yml                 # Azure ML environment YAML
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

- Azure CLI with the `ml` extension (`az extension add -n ml`)
- An Azure ML workspace
- A compute cluster — **must be set** in every YAML (`azureml:<YOUR-COMPUTE-CLUSTER>`)

---

## Workflow – Come lavorare sul progetto (standard di team)

Di seguito il flusso operativo consigliato, dalla connessione alla VM fino all'esecuzione di job e pipeline.

---

### Step 1 — Collegarsi alla Compute Instance con VS Code

1. Aprire **VS Code** in locale.
2. Installare l'estensione **Azure Machine Learning** (se non già presente).
3. Dalla palette comandi (`Ctrl+Shift+P`) selezionare **Azure ML: Connect to Compute Instance** oppure, in alternativa, usare la connessione **Remote - SSH**:
   - Dal portale Azure ML Studio → **Compute** → **Compute instances** → copiare il comando SSH o selezionare *Connect with VS Code*.
   - VS Code aprirà una sessione remota sulla VM; il file system montato (`~/cloudfiles/code/Users/<username>/`) è condiviso e persistente.
4. Verificare che il terminale integrato di VS Code punti alla compute instance (il prompt mostra `azureuser@<vm-name>`).

> **Tip:** Tutte le modifiche al codice fatte via VS Code sono salvate direttamente sul file share di Azure ML e visibili anche da altri notebook o terminali sulla stessa workspace.

---

### Step 2 — Autenticazione con Azure CLI

Aprire il terminale integrato in VS Code ed eseguire:

```bash
# Login interattivo (apre il browser o usa il device code)
az login

# Verificare la sottoscrizione attiva
az account show --output table

# Se necessario, selezionare la sottoscrizione corretta
az account set --subscription "<subscription-id>"
```

Impostare i **default** di workspace e resource group per evitare di ripeterli ad ogni comando:

```bash
az configure --defaults group=<resource-group> workspace=<workspace-name>
```

Da questo momento in poi tutti i comandi `az ml ...` useranno automaticamente il gruppo e la workspace configurati.

Verificare che l'estensione `ml` sia installata:

```bash
az extension show -n ml --query version -o tsv
# Se mancante:
az extension add -n ml
```

---

### Step 3 — Esecuzione: Job singolo o Pipeline

#### Opzione A — Job singolo (command job)

Il job singolo è utile per test rapidi o esecuzioni one-shot. Tutto è definito in un unico YAML che specifica input, output, environment e comando.

```bash
# 1. Registrare l'environment (solo la prima volta o quando cambia)
az ml environment create --file environments/environment.yml

# 2. Registrare il data asset (solo la prima volta o per nuove versioni)
az ml data create --file data/clinical_readmission.yml

# 3. Lanciare il job
az ml job create --file jobs/train_job.yml
```

Monitorare l'esecuzione:

```bash
# Dalla CLI
az ml job show --name <job-name> --query status -o tsv

# Oppure dallo stream dei log in tempo reale
az ml job stream --name <job-name>
```

Il job apparirà anche in **Azure ML Studio → Experiments → blueprint_standalone_job**.

---

#### Opzione B — Pipeline (consigliato per workflow strutturati)

La pipeline suddivide il lavoro in **componenti riutilizzabili**, ciascuno nella propria cartella con codice e YAML separati. Questo abilita il **caching automatico**: Azure ML esegue solo i componenti i cui input o codice sono cambiati.

```bash
# 1. Registrare environment e data (se non già fatto)
az ml environment create --file environments/environment.yml
az ml data create --file data/clinical_readmission.yml

# 2. Lanciare la pipeline
az ml job create --file pipelines/training_pipeline/training_pipeline.yml
```

La pipeline esegue 4 step in sequenza:

| Step | Componente | Cosa fa |
|------|-----------|---------|
| 1 | `data_prep` | Split stratificato train/test (70/30) |
| 2 | `train` | Logistic Regression con cross-validation |
| 3 | `evaluate` | Metriche sul test set (accuracy, F1, ROC-AUC) |
| 4 | `register_model` | Registra il modello nel Model Registry |

Monitorare la pipeline:

```bash
# Status della pipeline
az ml job show --name <pipeline-job-name> --query status -o tsv

# Log di un singolo step (child job)
az ml job stream --name <pipeline-job-name>
```

Dalla UI è possibile visualizzare il grafo della pipeline in **Azure ML Studio → Jobs → blueprint_training_pipeline**.

---

### Riepilogo comandi frequenti

| Azione | Comando |
|--------|---------|
| Login | `az login` |
| Impostare default | `az configure --defaults group=<rg> workspace=<ws>` |
| Creare environment | `az ml environment create --file environments/environment.yml` |
| Registrare data asset | `az ml data create --file data/clinical_readmission.yml` |
| Lanciare job singolo | `az ml job create --file jobs/train_job.yml` |
| Lanciare pipeline | `az ml job create --file pipelines/training_pipeline/training_pipeline.yml` |
| Stato di un job | `az ml job show --name <name> --query status` |
| Stream log | `az ml job stream --name <name>` |
| Elenco modelli registrati | `az ml model list --query "[].{name:name, version:version}" -o table` |

---

## Quick Start (sintesi rapida)

```bash
# Autenticarsi e configurare i default
az login
az configure --defaults group=<rg> workspace=<ws>

# Setup una tantum
az ml environment create --file environments/environment.yml
az ml data create --file data/clinical_readmission.yml

# Lanciare la pipeline
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
