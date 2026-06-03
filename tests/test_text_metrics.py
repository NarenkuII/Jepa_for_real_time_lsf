from src.text.metrics_text import cer, chrf, corpus_bleu, exact_match, wer


def test_text_metrics():
    preds = ["bonjour monde"]
    refs = ["bonjour le monde"]
    assert 0 <= corpus_bleu(preds, refs) <= 1
    assert 0 <= chrf(preds, refs) <= 1
    assert wer(preds, refs) > 0
    assert cer(preds, refs) > 0
    assert exact_match(preds, refs) == 0

