from __future__ import annotations
from typing import Dict, Any

from analyzer.perplexity import get_perplexity
from analyzer.features import (
    lexical_diversity,
    repetition_score,
    sentence_stats,
    burstiness_score
)
from analyzer.scoring import score_to_percent, classify
from analyzer.sentence_ai import analyze_sentences


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Main analysis function:
    - Document-level AI percentage
    - Explainable reasons
    - Sentence-level highlighting
    """

    if not text or not text.strip():
        return {
            "ok": False,
            "error": "Empty text provided."
        }

    # Truncate very long input for performance
    max_chars = 6000
    raw_text = text
    if len(text) > max_chars:
        text = text[:max_chars]

    # ===== Document-level analysis =====
    perplexity = get_perplexity(text)
    lexical = lexical_diversity(text)
    repetition = repetition_score(text)
    sent_stats = sentence_stats(text)
    burstiness = burstiness_score(text)

    ai_percent, reasons = classify(
        perplexity=perplexity,
        lexical=lexical,
        repetition=repetition,
        burstiness=burstiness,
        sent_stats=sent_stats
    )

    label, confidence = score_to_percent(ai_percent)

    # ===== Sentence-level analysis =====
    sentence_results = analyze_sentences(text)

    return {
        "ok": True,
        "truncated": len(raw_text) > max_chars,
        "ai_percentage": ai_percent,
        "human_percentage": max(0, 100 - ai_percent),
        "classification": label,
        "confidence": confidence,
        "metrics": {
            "perplexity": round(perplexity, 2),
            "lexical_diversity": round(lexical, 4),
            "repetition_score": round(repetition, 4),
            "burstiness": round(burstiness, 4),
            "sentences": sent_stats["sentences"],
            "avg_sentence_length": round(sent_stats["avg_len"], 2),
            "std_sentence_length": round(sent_stats["std_len"], 2),
        },
        "reasons": reasons,
        "sentences": sentence_results,
    }
