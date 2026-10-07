"""Reports: label shares with 95% bootstrap intervals, overall and for each subreddit, and aspect tables.

Each share has its count. Failed predictions are a separate count, never a label. Subreddits are
reported apart, because each subreddit has its own lean.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .sentiment import LABELS


def shares(labels: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    n = len(labels)
    out: dict = {"n": n}
    if n == 0:
        return out
    arr = np.array(labels)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    for lab in LABELS:
        hits = (arr == lab).astype(float)
        boot = hits[idx].mean(axis=1)
        out[lab] = {"share": round(float(hits.mean()), 4),
                    "ci95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]}
    return out


def build_report(topic: str, comments: dict[str, object], results: list[dict], filter_counts: dict[str, int],
                 seed: int = 0) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    failed = len(results) - len(ok)
    by_sub: dict[str, list[str]] = defaultdict(list)
    for r in ok:
        by_sub[comments[r["comment_id"]].subreddit].append(r["sentiment"])
    aspects: dict[str, list[str]] = defaultdict(list)
    aspect_failed: dict[str, int] = defaultdict(int)
    for r in results:
        for a in r["aspects"]:
            if a["status"] == "ok":
                aspects[a["aspect"]].append(a["label"])
            else:
                aspect_failed[a["aspect"]] += 1
    analysed = len(results)
    return {
        "topic": topic,
        "filter_counts": filter_counts,
        "analysed": analysed,
        "failed": failed,
        "sentiment": shares([r["sentiment"] for r in ok], seed=seed),
        "by_subreddit": {s: shares(v, seed=seed) for s, v in sorted(by_sub.items())},
        "aspects": {a: {"mentions": len(v), "mention_rate": round(len(v) / analysed, 4) if analysed else 0.0,
                        "failed": aspect_failed[a], **shares(v, seed=seed)} for a, v in sorted(aspects.items())},
    }


def to_markdown(rep: dict) -> str:
    def row(name, s):
        if not s.get("n"):
            return f"| {name} | 0 | - | - | - |"
        cells = [f"{s[l]['share']:.2f} [{s[l]['ci95'][0]:.2f}, {s[l]['ci95'][1]:.2f}]" for l in LABELS]
        return f"| {name} | {s['n']} | " + " | ".join(cells) + " |"

    lines = [f"# Topic: {rep['topic']}", "", f"Filter counts: {rep['filter_counts']}. Failed predictions: {rep['failed']}.",
             "", "| Group | n | negative | neutral | positive |", "|---|---|---|---|---|", row("all", rep["sentiment"])]
    lines += [row(f"r/{s}", v) for s, v in rep["by_subreddit"].items()]
    lines += ["", "| Aspect | mentions | negative | neutral | positive |", "|---|---|---|---|---|"]
    lines += [row(a, v) for a, v in rep["aspects"].items()]
    return "\n".join(lines)
