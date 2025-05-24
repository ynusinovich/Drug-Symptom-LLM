# Drug-Symptom LLM
Goal: Predict likely health conditions from a list of medications using decoder-only LLMs fine-tuned with LoRA.
Fine-tunes 400M–3B parameter models on a DrugBank-derived dataset.

## Environment Setup

### Option 1: Local (Conda + Pipenv)
Create Conda base environment with Python 3.11, install pipenv inside Conda env, install dependencies (from Pipfile)
```bash
make setup
make install
```

### Option 2: Docker (for GPU training)
Build development image and then run notebook container with GPU + shared memory
```bash
make docker-dev-build
make docker-dev-run
```

## Dataset
- Source: DrugBank
  - Knox C, Wilson M, Klinger CM, et al.DrugBank 6.0: the DrugBank Knowledgebase for 2024.Nucleic Acids Res. 2024 Jan 5;52(D1):D1265-D1275. doi: 10.1093/nar/gkad976
- License: Academic access only (register and download XML)
- Reproduction steps:
  - Place XML zip into data/
  - Run notebooks/0_explore_drugbank.ipynb to parse
  - Run notebooks/1_prepare_prompts.ipynb to generate training data
  - Run scripts/merge_datasets.py to combine training and validation data

Note: Data is not redistributed in this repository.

## Utilities
Launch MLflow UI for experiment tracking
```bash
make mlflow
```

## Model Training
All models are trained using LoRA and src/train.py.

Train all initial models:
```bash
make train-all
```

Train initial models individually:
```bash
make train-gpt2
make train-falcon
make train-openllama
make train-pythia
```

Train final Llama model:
```bash
make train-final-openllama
```

## Model Evaluation
Each model is evaluated on validation prompts using exact match, ROUGE-L, and BLEU.

Evaluate all models:
```bash
make eval-all
```

Evaluate models individually:
```bash
make eval-gpt2
make eval-falcon
make eval-openllama
make eval-pythia
```

Evaluate final Llama model:
```bash
make eval-final-openllama
```

## Model Inference

### Docker (for GPU inference)
Build inference container and run API server (FastAPI on port 8000) with GPU + shared memory
```bash
make docker-inf-build
make docker-inf-run
```

### Single Prediction

Run model predictions on a single test prompts (e.g., `--instruction "Given the following drugs and their categories, predict the associated health condition."
                                                --drug "Prednisone"
                                                --categories "Glucocorticoids; Steroids; Anti-inflammatory Agents"`):
```bash
make predict-single
```

### Batch Prediction

Run model predictions on a JSONL file of test prompts (e.g., `llm_test.json`):
```bash
make predict-batch
```
The file must follow the same JSONL schema (instruction, input, max_tokens) produced by the notebook.

## Disclaimer

This repo is research‑only and provides no medical advice.