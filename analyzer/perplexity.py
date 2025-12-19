import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Lazy-load globals
_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if _model is None:
        _model = GPT2LMHeadModel.from_pretrained("gpt2")
        _model.to(_device)
        _model.eval()

def get_perplexity(text: str) -> float:
    """
    Returns perplexity estimate using GPT-2.
    Lower perplexity => more predictable text (often AI-like, not always).
    """
    _load()

    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(_device)

    with torch.no_grad():
        outputs = _model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(math.exp(loss.item()))
