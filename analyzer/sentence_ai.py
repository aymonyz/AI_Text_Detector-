from analyzer.perplexity import get_perplexity
from analyzer.features import split_sentences


def analyze_sentences(text: str):
    sentences = split_sentences(text)
    results = []

    for s in sentences:
        if len(s.split()) < 6:
            continue

        ppl = get_perplexity(s)
        ai_like = ppl < 50  # threshold

        results.append({
            "text": s,
            "perplexity": round(ppl, 2),
            "ai_like": ai_like
        })

    return results
