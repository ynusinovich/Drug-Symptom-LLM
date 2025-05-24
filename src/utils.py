from transformers import AutoTokenizer, GPTNeoXTokenizerFast


def safe_tokenizer(model_name):
    if "pythia" in model_name.lower():
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Set pad token to EOS if undefined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Some LLaMA models need left padding
    tokenizer.padding_side = (
        "left"
        if "gpt" in model_name.lower() or "pythia" in model_name.lower()
        else "right"
    )
    return tokenizer
