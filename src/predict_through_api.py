import argparse
import json
from pathlib import Path
from typing import Dict, List

import requests

"""
Quick client for the FastAPI Drug‑Condition predictor.

Two modes
--------- 
1. Single example

   python src/predict_through_api.py \
         --instruction "Given the following drugs and their categories, predict the associated health condition." \
         --drug "Prednisone" \
         --categories "Glucocorticoids; Steroids; Anti‑inflammatory Agents" \
         --max_tokens 64

2. Batch (JSON‑Lines file, one row from your dataset)

   python src/predict_through_api.py \
         --data_path data/llm_test.json \
         --output_path results/api_predictions.jsonl \
         --max_tokens 64
"""

API_ROOT = "http://localhost:8000"  # adjust if you mapped a different port


def call_single(payload: Dict, max_tokens: int):
    payload["max_tokens"] = max_tokens
    r = requests.post(f"{API_ROOT}/predict", json=payload, timeout=120)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


def call_batch(rows: List[Dict], out_path: Path, max_tokens: int):
    for row in rows:
        row["max_tokens"] = max_tokens

    r = requests.post(f"{API_ROOT}/predict_batch", json=rows, timeout=3_600)
    r.raise_for_status()
    preds = r.json()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in preds:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(preds)} predictions to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", type=str)
    p.add_argument("--drug", type=str)
    p.add_argument("--categories", type=str)
    p.add_argument("--data_path", type=str, help="JSONL file for batch mode")
    p.add_argument("--output_path", default="results/api_predictions.jsonl")
    p.add_argument("--max_tokens", type=int, default=64)
    args = p.parse_args()

    # ---------- batch mode ----------
    if args.data_path:
        rows = []
        with open(args.data_path) as f:
            for line in f:
                row = json.loads(line)
                rows.append(
                    {
                        "instruction": row["instruction"],
                        "input": row["input"],
                    }
                )
        call_batch(rows, Path(args.output_path), args.max_tokens)
        return

    # ---------- single mode ----------
    if not (args.instruction and args.drug and args.categories):
        raise SystemExit("Provide either --data_path OR the 3 single‑sample flags.")
    single_payload = {
        "instruction": args.instruction,
        "input": f"drugs: {args.drug} || categories: {args.categories}",
    }
    call_single(single_payload, args.max_tokens)


if __name__ == "__main__":
    main()
