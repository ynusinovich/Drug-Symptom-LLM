import argparse
import json
from datetime import datetime
from pathlib import Path

import evaluate
import mlflow
import torch
from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from utils import safe_tokenizer


def load_val_dataset(path):
    with open(path, "r") as f:
        rows = [json.loads(line) for line in f]
    return rows


def eval(model_path, val_path, max_tokens=64):
    print(f"Loading model from: {model_path}")
    config = PeftConfig.from_pretrained(model_path)

    # Conditional quantized loading for large models
    if any(x in config.base_model_name_or_path.lower() for x in ["falcon", "llama"]):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            quantization_config=quant_config,
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, device_map="auto"
        )

    model = PeftModel.from_pretrained(base_model, model_path).to("cuda").eval()
    tokenizer = safe_tokenizer(config.base_model_name_or_path)
    val_data = load_val_dataset(val_path)

    preds, refs, inputs = [], [], []
    for row in tqdm(val_data, desc="Generating predictions"):
        prompt = f"{row['instruction']}\n{row['input']}"
        inputs.append(prompt)
        enc = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to("cuda")

        with torch.no_grad():
            output = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = decoded.replace(prompt, "").strip()
        preds.append(generated)
        refs.append(row["output"].strip())

    # Evaluation
    rouge = evaluate.load("rouge")
    sacrebleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")

    rouge_score = rouge.compute(predictions=preds, references=refs)["rougeL"]
    sacrebleu_score = sacrebleu.compute(predictions=preds, references=refs)["score"]
    bertscore_result = bertscore.compute(predictions=preds, references=refs, lang="en")
    exact_match = sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / len(refs)

    metrics = {
        "exact_match": round(exact_match, 4),
        "rougeL": round(rouge_score, 4),
        "sacrebleu": round(sacrebleu_score, 4),
        "bertscore_f1": round(
            sum(bertscore_result["f1"]) / len(bertscore_result["f1"]), 4
        ),
    }

    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k:14}: {v:.4f}")

    # Save to results/
    out_path = (
        Path("results")
        / f"{Path(model_path).stem}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": model_path,
                "val_path": val_path,
                "metrics": metrics,
                "predictions": [
                    {"input": i, "generated": p, "reference": r}
                    for i, p, r in zip(inputs, preds, refs)
                ],
            },
            f,
            indent=2,
        )

    print(f"Saved evaluation to {out_path}")

    # Log to MLflow
    mlflow.set_experiment("drug_condition_prediction")
    with mlflow.start_run(run_name=f"eval_{Path(model_path).stem}"):
        mlflow.log_params(
            {"model_path": model_path, "val_path": val_path, "max_tokens": max_tokens}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--val_path", default="data/llm_val.json")
    args = parser.parse_args()
    eval(args.model_dir, args.val_path)
