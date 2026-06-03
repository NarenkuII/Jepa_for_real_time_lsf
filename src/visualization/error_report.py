from __future__ import annotations

from src.text.metrics_text import cer, chrf, corpus_bleu, empty_prediction_rate, exact_match, repetition_rate, rouge_l, wer


def summarize_text_errors(preds: list[str], refs: list[str]) -> dict:
    return {
        "bleu": corpus_bleu(preds, refs),
        "chrf": chrf(preds, refs),
        "rouge_l": rouge_l(preds, refs),
        "wer": wer(preds, refs),
        "cer": cer(preds, refs),
        "exact_match": exact_match(preds, refs),
        "empty_prediction_rate": empty_prediction_rate(preds),
        "repetition_rate": repetition_rate(preds),
    }

