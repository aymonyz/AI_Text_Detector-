import re
import numpy as np
from collections import Counter
from typing import Dict, List

_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(text)]

def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter (good enough for graduation project)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def lexical_diversity(text: str) -> float:
    words = tokenize_words(text)
    if len(words) < 20:
        return 0.0
    return len(set(words)) / max(1, len(words))

def repetition_score(text: str) -> float:
    """
    Measures repetition via top-word concentration.
    Higher -> more repetitive.
    """
    words = tokenize_words(text)
    if len(words) < 30:
        return 0.0
    c = Counter(words)
    top = sum(freq for _, freq in c.most_common(10))
    return top / len(words)

def sentence_stats(text: str) -> Dict[str, float]:
    sents = split_sentences(text)
    if not sents:
        return {"sentences": 0, "avg_len": 0.0, "std_len": 0.0}
    lengths = [len(tokenize_words(s)) for s in sents]
    return {
        "sentences": len(sents),
        "avg_len": float(np.mean(lengths)),
        "std_len": float(np.std(lengths)),
    }

def burstiness_score(text: str) -> float:
    """
    Burstiness proxy:
    higher std of sentence lengths relative to mean -> more "human-like" variance.
    We compute std/mean (bounded).
    """
    st = sentence_stats(text)
    mean = st["avg_len"]
    std = st["std_len"]
    if mean <= 0:
        return 0.0
    return float(std / mean)
