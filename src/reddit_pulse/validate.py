"""Validation against hand labels: annotator agreement and model quality.

Gold file (CSV): `comment_id,annotator,label` with label negative, neutral or positive.
- Cohen's kappa between the first two annotators, on the comments that both labelled.
- The adjudicated gold label is the majority label. A tie is left out and counted.
- Model quality: accuracy and macro-F1 with a 95% bootstrap interval, and a confusion matrix.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .sentiment import LABELS


class LabelError(ValueError):
    pass


def read_gold(path: str | Path) -> dict[str, dict[str, str]]:
    """Return {comment_id: {annotator: label}}."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"comment_id", "annotator", "label"} - set(reader.fieldnames or [])
        if missing:
            raise LabelError(f"{path}: missing column(s) {sorted(missing)}")
        for i, r in enumerate(reader, 2):
            label = (r["label"] or "").strip().lower()
            if label not in LABELS:
                raise LabelError(f"{path} line {i}: label {label!r} is not one of {LABELS}")
            out[r["comment_id"]][r["annotator"]] = label
    return dict(out)


def agreement(gold: dict[str, dict[str, str]]) -> dict:
    annotators = sorted({a for labels in gold.values() for a in labels})
    if len(annotators) < 2:
        return {"annotators": annotators, "kappa": None, "items": 0}
    a, b = annotators[:2]
    both = [(v[a], v[b]) for v in gold.values() if a in v and b in v]
    if not both:
        return {"annotators": annotators, "kappa": None, "items": 0}
    kappa = cohen_kappa_score([x for x, _ in both], [y for _, y in both], labels=list(LABELS))
    return {"annotators": [a, b], "kappa": round(float(kappa), 4), "items": len(both),
            "raw_agreement": round(sum(x == y for x, y in both) / len(both), 4)}


def adjudicate(gold: dict[str, dict[str, str]]) -> tuple[dict[str, str], int]:
    out, ties = {}, 0
    for cid, labels in gold.items():
        counts = Counter(labels.values()).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            ties += 1
            continue
        out[cid] = counts[0][0]
    return out, ties


def score_model(pred: dict[str, str | None], gold: dict[str, str], n_boot: int = 1000, seed: int = 0) -> dict:
    ids = sorted(cid for cid in gold if cid in pred and pred[cid] is not None)
    missing = sum(1 for cid in gold if cid not in pred or pred[cid] is None)
    if not ids:
        raise LabelError("no gold comment has a model prediction")
    y = np.array([gold[c] for c in ids])
    p = np.array([pred[c] for c in ids])
    rng = np.random.default_rng(seed)
    boots = [f1_score(y[idx], p[idx], labels=list(LABELS), average="macro", zero_division=0)
             for idx in (rng.integers(0, len(ids), len(ids)) for _ in range(n_boot))]
    return {"items": len(ids), "missing_predictions": missing,
            "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0)), 4),
            "macro_f1_ci95": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])],
            "confusion_rows_gold_cols_pred": confusion_matrix(y, p, labels=list(LABELS)).tolist()}


def sample_for_labels(comments: list, n: int, seed: int = 0) -> list:
    """A sample stratified by subreddit (equal share for each subreddit, as far as possible)."""
    by_sub: dict[str, list] = defaultdict(list)
    for c in comments:
        by_sub[c.subreddit].append(c)
    rng = random.Random(seed)
    for v in by_sub.values():
        rng.shuffle(v)
    out, subs = [], sorted(by_sub)
    while len(out) < n and any(by_sub[s] for s in subs):
        for s in subs:
            if by_sub[s] and len(out) < n:
                out.append(by_sub[s].pop())
    return out
