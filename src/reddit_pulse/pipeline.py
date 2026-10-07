"""One parameterised pipeline for every topic: filter -> sentiment -> aspects -> results and report."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from .absa import AbsaModel, find_mentions
from .clean import filter_counts, filter_reason, model_view
from .config import TopicConfig
from .report import build_report
from .sentiment import SentimentModel
from .store import Comment, Store


@dataclass
class RunOutput:
    run_id: str
    kept: list[Comment]
    results: list[dict]
    report: dict


def filter_comments(comments: list[Comment]) -> tuple[list[Comment], dict[str, int]]:
    seen: set[str] = set()
    kept, reasons = [], []
    for c in comments:
        reason = "bot" if c.is_bot else filter_reason(c.text, None, seen)
        reasons.append(reason)
        if reason is None:
            kept.append(c)
    return kept, filter_counts(reasons)


def analyse(cfg: TopicConfig, comments: list[Comment], sentiment: SentimentModel, absa: AbsaModel,
            seed: int = 42) -> tuple[list[Comment], list[dict], dict[str, int]]:
    kept, counts = filter_comments(comments)
    views = [model_view(c.text) for c in kept]
    preds = sentiment.predict(views)
    mentions = [find_mentions(v, cfg.aspects) for v in views]
    pairs = [(m.sentence, m.aspect) for ms in mentions for m in ms]
    absa_preds = iter(absa.predict(pairs)) if pairs else iter(())
    results = []
    for c, p, ms in zip(kept, preds, mentions):
        aspects = []
        for m in ms:
            ap = next(absa_preds)
            aspects.append({"aspect": m.aspect, "label": ap.label, "score": ap.score, "status": ap.status})
        results.append({"comment_id": c.comment_id, "sentiment": p.label, "sentiment_score": p.score,
                        "status": p.status, "aspects": aspects})
    return kept, results, counts


def run(cfg: TopicConfig, store: Store, sentiment: SentimentModel, absa: AbsaModel, seed: int = 42) -> RunOutput:
    comments = store.comments(cfg.name)
    if not comments:
        raise ValueError(f"no comments for topic {cfg.name!r}; run collect or synth first")
    kept, results, counts = analyse(cfg, comments, sentiment, absa, seed)
    params = {"sentiment": sentiment.name, "absa": absa.name, "seed": seed, "comments": len(comments),
              "aspects": [a.name for a in cfg.aspects]}
    run_id = hashlib.sha1(json.dumps([cfg.name, params, [c.comment_id for c in kept]]).encode()).hexdigest()[:12]
    store.add_run(run_id, cfg.name, "analyse", datetime.now(timezone.utc).isoformat(timespec="seconds"), params)
    store.add_results(run_id, results)
    report = build_report(cfg.name, {c.comment_id: c for c in kept}, results, counts, seed)
    report["run_id"] = run_id
    return RunOutput(run_id, kept, results, report)
