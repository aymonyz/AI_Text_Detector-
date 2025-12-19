from typing import Dict, Any, List, Tuple

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_perplexity(ppl: float) -> float:
    """
    Map perplexity into [0,1] where lower ppl => higher AI-likeness.
    Typical GPT-2 ppl: AI text often 20-60, human can be higher.
    We'll squash around these ranges.
    """
    # 10 -> very AI-like, 120 -> very human-like (rough)
    x = (120 - ppl) / (120 - 10)
    return clamp(x, 0.0, 1.0)

def normalize_lexical(lex: float) -> float:
    """
    Lower lexical diversity => more AI-like.
    lex in [0..1], common ranges ~0.25-0.65
    """
    x = (0.55 - lex) / (0.55 - 0.20)  # below 0.20 very repetitive
    return clamp(x, 0.0, 1.0)

def normalize_repetition(rep: float) -> float:
    """
    Higher repetition => more AI-like.
    rep in [0..1], common ranges 0.08 - 0.25
    """
    x = (rep - 0.10) / (0.28 - 0.10)
    return clamp(x, 0.0, 1.0)

def normalize_burstiness(b: float) -> float:
    """
    Lower burstiness => more AI-like.
    b = std/mean. Human tends to have more variation.
    """
    x = (0.28 - b) / (0.28 - 0.08)
    return clamp(x, 0.0, 1.0)

def combine_score(ppl_n: float, lex_n: float, rep_n: float, burst_n: float) -> float:
    """
    Weighted sum in [0,1].
    """
    return clamp(
        0.45 * ppl_n +
        0.20 * lex_n +
        0.20 * rep_n +
        0.15 * burst_n,
        0.0, 1.0
    )

def to_percent(score01: float) -> int:
    return int(round(clamp(score01, 0.0, 1.0) * 100))

def score_to_percent(ai_percent: int) -> Tuple[str, str]:
    """
    Returns label + confidence.
    """
    p = ai_percent
    if p >= 75:
        return "AI Generated", "High"
    if p >= 55:
        return "Mixed / Possibly AI", "Medium"
    return "Human Written", "Medium" if p >= 35 else "High"

def classify(perplexity: float, lexical: float, repetition: float, burstiness: float, sent_stats: Dict[str, Any]) -> Tuple[int, List[str]]:
    ppl_n = normalize_perplexity(perplexity)
    lex_n = normalize_lexical(lexical)
    rep_n = normalize_repetition(repetition)
    burst_n = normalize_burstiness(burstiness)

    score01 = combine_score(ppl_n, lex_n, rep_n, burst_n)
    ai_percent = to_percent(score01)

    reasons: List[str] = []

    # Reasons (Explainability)
    if perplexity < 45:
        reasons.append("Low perplexity: text is highly predictable (often seen in LLM outputs).")
    elif perplexity > 90:
        reasons.append("High perplexity: text is less predictable, often closer to human writing.")

    if lexical > 0 and lexical < 0.35:
        reasons.append("Low lexical diversity: limited vocabulary variety.")
    elif lexical >= 0.55:
        reasons.append("High lexical diversity: broader vocabulary usage.")

    if repetition > 0.18:
        reasons.append("High repetition score: frequent reuse of common words/phrases.")
    elif repetition > 0 and repetition < 0.12:
        reasons.append("Low repetition score: less repetitive word usage.")

    if burstiness < 0.12:
        reasons.append("Low burstiness: sentence lengths are very uniform (common in generated text).")
    elif burstiness > 0.22:
        reasons.append("High burstiness: sentence length varies more (common in human writing).")

    if sent_stats.get("sentences", 0) < 3:
        reasons.append("Very short input: detection confidence can be unreliable on short text.")

    if not reasons:
        reasons.append("Signals are mixed; model relies on combined statistical indicators.")

    return ai_percent, reasons
