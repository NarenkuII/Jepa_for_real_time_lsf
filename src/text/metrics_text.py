from __future__ import annotations

from collections import Counter


def _tokens(s: str) -> list[str]:
    return s.strip().split()


def edit_distance(a: list, b: list) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def corpus_bleu(preds: list[str], refs: list[str]) -> float:
    matches = 0
    total = 0
    for pred, ref in zip(preds, refs):
        pc = Counter(_tokens(pred))
        rc = Counter(_tokens(ref))
        matches += sum((pc & rc).values())
        total += sum(pc.values())
    return matches / max(total, 1)


def chrf(preds: list[str], refs: list[str], n: int = 6, beta: float = 2.0) -> float:
    scores = []
    for pred, ref in zip(preds, refs):
        p_total = r_total = match = 0
        for k in range(1, n + 1):
            p = Counter(pred[i : i + k] for i in range(max(0, len(pred) - k + 1)))
            r = Counter(ref[i : i + k] for i in range(max(0, len(ref) - k + 1)))
            match += sum((p & r).values())
            p_total += sum(p.values())
            r_total += sum(r.values())
        precision = match / max(p_total, 1)
        recall = match / max(r_total, 1)
        denom = beta * beta * precision + recall
        scores.append((1 + beta * beta) * precision * recall / denom if denom else 0.0)
    return sum(scores) / max(len(scores), 1)


def rouge_l(preds: list[str], refs: list[str]) -> float:
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, x in enumerate(a, 1):
            for j, y in enumerate(b, 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    vals = []
    for p, r in zip(preds, refs):
        rt = _tokens(r)
        vals.append(lcs(_tokens(p), rt) / max(len(rt), 1))
    return sum(vals) / max(len(vals), 1)


def wer(preds: list[str], refs: list[str]) -> float:
    return sum(edit_distance(_tokens(p), _tokens(r)) for p, r in zip(preds, refs)) / max(sum(len(_tokens(r)) for r in refs), 1)


def cer(preds: list[str], refs: list[str]) -> float:
    return sum(edit_distance(list(p), list(r)) for p, r in zip(preds, refs)) / max(sum(len(r) for r in refs), 1)


def exact_match(preds: list[str], refs: list[str]) -> float:
    return sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / max(len(refs), 1)


def repetition_rate(preds: list[str]) -> float:
    rates = []
    for pred in preds:
        toks = _tokens(pred)
        rates.append(1.0 - len(set(toks)) / max(len(toks), 1))
    return sum(rates) / max(len(rates), 1)


def empty_prediction_rate(preds: list[str]) -> float:
    return sum(not p.strip() for p in preds) / max(len(preds), 1)

