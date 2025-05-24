# Makefile for evaluating LLMs on DrugBank prompts

# Environment

setup:
	conda create -n drug_symptom_llm python=3.11 -y
	conda activate drug_symptom_llm && pip install pipenv

install:
	pipenv install --dev

# Code formatting

format:
	pipenv run black ./src
	pipenv run isort ./src
	pipenv run black ./scripts
	pipenv run isort ./scripts

# Docker targets

docker-dev-build:
	docker build -f Dockerfile_dev -t drug_symptom_llm_dev .

docker-dev-run:
	docker run --gpus all -it --rm \
		--shm-size=32g \
		-v $$PWD:/app \
		-p 8888:8888 \
		drug_symptom_llm_dev

# MLflow

mlflow:
	pipenv run mlflow ui -h 0.0.0.0 -p 5000

# Training

train-gpt2:
	pipenv run python src/train.py --base_model gpt2 \
		--output_dir models/gpt2_lora \
		--num_epochs 4

train-falcon:
	pipenv run python src/train.py --base_model tiiuae/falcon-rw-1b \
		--output_dir models/falcon_lora \
		--num_epochs 4

train-openllama:
	pipenv run python src/train.py --base_model openlm-research/open_llama_3b \
		--output_dir models/openllama_lora \
		--num_epochs 4

train-pythia:
	pipenv run python src/train.py --base_model EleutherAI/pythia-410m \
		--output_dir models/pythia_lora \
		--num_epochs 4

train-all:
	train-gpt2 train-falcon train-openllama train-pythia

# Individual model evaluation targets

eval-gpt2:
	pipenv run python src/eval.py --model_dir models/gpt2_lora

eval-falcon:
	pipenv run python src/eval.py --model_dir models/falcon_lora

eval-openllama:
	pipenv run python src/eval.py --model_dir models/openllama_lora

eval-pythia:
	pipenv run python src/eval.py --model_dir models/pythia_lora

# Evaluate all models

eval-all: eval-gpt2 eval-falcon eval-openllama eval-pythia

# Train final model

train-final-openllama:
	pipenv run python src/train.py --base_model openlm-research/open_llama_3b \
		--dataset_path data/llm_train_and_val.json --val_path data/llm_test.json \
		--output_dir models/openllama_lora_final --num_epochs 8

eval-final-openllama:
	pipenv run python src/eval.py --model_dir models/openllama_lora_final --val_path data/llm_test.json

# Inference Docker build

docker-inf-build:
	docker build -f Dockerfile_inf -t drug_symptom_llm_inf .

# Run inference container with model mounted

docker-inf-run:
	docker run --gpus all -it --rm \
		--shm-size=32g \
		-v $(PWD):/app \
		-p 8000:8000 \
		drug_symptom_llm_inf

# Test the API with one drug

predict-single:
	pipenv run python src/predict_through_api.py \
		--instruction "Given the following drug(s) and their categories, predict the associated health condition.\nOutput answer in the form: 'Indication: <condition>'." \
		--drug "curcumin" \
		--categories "Diarylheptanoids; Heptanes; Cytochrome P-450 CYP2B6 Inhibitors (moderate); Cytochrome P-450 CYP2C9 Inhibitors (strong); Coloring Agents"

predict-batch:
	pipenv run python src/predict_through_api.py \
		--data_path data/llm_test.json \
		--output_path results/api_predictions.jsonl

.PHONY: setup install format docker-dev-build docker-dev-run \
        train-gpt2 train-falcon train-openllama train-pythia train-all \
        eval-gpt2 eval-falcon eval-openllama eval-pythia eval-all \
        train-final-openllama eval-final-openllama \
        docker-inf-build docker-inf-run predict-single predict-batch mlflow
