# app.py  – trimmed for clarity
from __future__ import annotations  # ↪ so we can keep |‑style unions in 3.10+

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training

from .utils import safe_tokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "models/openllama_lora_final")  # ← configurable

app: FastAPI = FastAPI(
    title="Drug‑Symptom Predictor",
    description="LoRA‑finetuned OpenLLaMA 3 B – predicts likely indications from DrugBank prompts.",
    version="0.1.0",
)

model, tokenizer = None, None
MAX_CONTEXT = 512  # keep in one place


# ─────────────────────────── lifecycle ──────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, tokenizer

    cfg = PeftConfig.from_pretrained(MODEL_DIR)
    is_llama_family = any(
        k in cfg.base_model_name_or_path.lower() for k in ("llama", "falcon")
    )

    quant_args = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if is_llama_family
        else None
    )

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        device_map="auto",
        quantization_config=quant_args,
    )
    if quant_args:
        base = prepare_model_for_kbit_training(base)

    model = PeftModel.from_pretrained(base, MODEL_DIR).eval().to("cuda")
    tokenizer = safe_tokenizer(cfg.base_model_name_or_path)

    print("✅  Model loaded.")
    yield
    print("🛑  Shutting down.")


app.router.lifespan_context = lifespan  # register


# ─────────────────────────── Pydantic DTOs ──────────────────────────
class PromptRequest(BaseModel):
    instruction: str
    input: str
    max_tokens: int = Field(128, le=256, ge=1)
    stop_sequence: Optional[str] = None


class PromptRequestV2(PromptRequest):
    generation_kwargs: Optional[Dict[str, Any]] = None


# ─────────────────────────── helpers ────────────────────────────────
def generate(
    prompt: str,
    max_tokens: int,
    stop: str | None = None,
    **overrides,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_CONTEXT,
    ).to("cuda")

    default = dict(
        max_new_tokens=max_tokens,
        do_sample=False,
        num_beams=6,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if stop:
        default["eos_token_id"] = tokenizer.encode(stop)[-1]

    with torch.no_grad():
        out = model.generate(**enc, **{**default, **overrides})

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    result = text.replace(prompt, "").strip()
    return result.split(stop)[0].strip() if stop else result


# ─────────────────────────── routes ─────────────────────────────────
@app.post("/predict")
def predict(req: PromptRequest):
    try:
        prompt = f"{req.instruction}\n{req.input}"
        return {"generated": generate(prompt, req.max_tokens, req.stop_sequence)}
    except Exception as err:  # noqa: BLE001
        raise HTTPException(500, detail=str(err)) from err


@app.post("/predict_batch")
def predict_batch(payload: List[PromptRequest]):
    try:
        return [
            {
                "instruction": r.instruction,
                "input": r.input,
                "generated": generate(
                    f"{r.instruction}\n{r.input}",
                    r.max_tokens,
                    r.stop_sequence,
                ),
            }
            for r in payload
        ]
    except Exception as err:  # noqa: BLE001
        raise HTTPException(500, detail=str(err)) from err


@app.post("/predict_v2")
def predict_v2(req: PromptRequestV2):
    try:
        prompt = f"{req.instruction}\n{req.input}"
        gkw = req.generation_kwargs or {}
        gkw.setdefault("max_new_tokens", req.max_tokens)
        return {"generated": generate(prompt, req.max_tokens, req.stop_sequence, **gkw)}
    except Exception as err:  # noqa: BLE001
        raise HTTPException(500, detail=str(err)) from err


@app.get("/health", tags=["infra"])
def health():
    """Lightweight readiness probe for container orchestrators."""
    return {"status": "ok"}
