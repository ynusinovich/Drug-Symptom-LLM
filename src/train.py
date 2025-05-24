# train.py
import argparse, json
from pathlib import Path

import mlflow, torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils import safe_tokenizer


def load_dataset(path: str | Path) -> Dataset:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    data = [
        {"text": f"{r['instruction']}\n{r['input']}\n{r['output']}"} for r in records
    ]
    return Dataset.from_list(data)


def resolve_target_modules(model_name):
    name = model_name.lower()
    if "gpt2" in name:
        return ["c_attn", "c_proj"]
    if "falcon" in name:
        return ["query_key_value", "dense"]
    if "pythia" in name or "gptneox" in name:
        return ["query_key_value"]
    if "llama" in name or "open_llama" in name:
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return ["q_proj", "v_proj"]


def main(args):
    print(f"Loading base model: {args.base_model}")
    # 8‑bit / 4‑bit depending on model size
    if "pythia" in args.base_model.lower():  # fits in fp16
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    else:  # 4‑bit quant
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map="auto", quantization_config=qcfg
        )
        model = prepare_model_for_kbit_training(model)

    tokenizer = safe_tokenizer(args.base_model)

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,  # 2 × r is a common heuristic
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=resolve_target_modules(args.base_model),
    )
    model = get_peft_model(model, lora_cfg)

    # data
    print("Loading data …")
    train_ds = load_dataset(args.dataset_path)
    val_ds = load_dataset(args.val_path)

    def tok_fn(ex):
        return tokenizer(
            ex["text"], truncation=True, padding="max_length", max_length=256
        )  # from 512

    train_tok = train_ds.map(tok_fn, batched=True)
    val_tok = val_ds.map(tok_fn, batched=True)

    # training schedule
    fp16_ok = "pythia" not in args.base_model.lower()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,  # default 8 (see CLI)
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_dir=f"{args.output_dir}/logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=fp16_ok,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # tracking & run
    mlflow.set_experiment("drug_condition_prediction")
    with mlflow.start_run(run_name=f"train_{Path(args.output_dir).stem}"):
        mlflow.log_params(
            dict(
                base_model=args.base_model,
                num_epochs=args.num_epochs,
                lr_initial=targs.learning_rate,
                scheduler="cosine",
                batch_size=targs.per_device_train_batch_size,
                lora_r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                max_len=256,
            )
        )
        trainer.train()

        # save artifacts
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        mlflow.log_artifacts(args.output_dir, artifact_path="model")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--dataset_path", default="data/llm_train.json")
    p.add_argument("--val_path", default="data/llm_val.json")
    p.add_argument("--output_dir", default="models/openllama_lora_final")
    p.add_argument("--num_epochs", type=int, default=8)  # was 3
    args = p.parse_args()
    main(args)
