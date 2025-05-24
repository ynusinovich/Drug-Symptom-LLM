import json
from pathlib import Path

train_path = Path("data/llm_train.json")
val_path = Path("data/llm_val.json")
out_path = Path("data/llm_train_and_val.json")

with train_path.open("r") as f_train, val_path.open("r") as f_val, out_path.open(
    "w"
) as f_out:
    for line in f_train:
        f_out.write(line)
    for line in f_val:
        f_out.write(line)

print(f"Merged {train_path.name} and {val_path.name} → {out_path.name}")
