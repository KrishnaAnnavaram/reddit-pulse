"""The `reddit-pulse` command line."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .absa import build_absa
from .clean import model_view, topic_view
from .config import ConfigError, Settings, TopicConfig
from .legacy import old_clean
from .pipeline import filter_comments, run
from .report import to_markdown
from .sentiment import build_sentiment
from .store import Store
from .topics import sweep_topics
from .validate import LabelError, adjudicate, agreement, read_gold, sample_for_labels, score_model


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_collect(args, s):
    from .collect import RedditCollector

    cfg = TopicConfig.from_toml(args.config)
    store = Store(args.db or s.db_path)
    added = store.add_comments(RedditCollector(s).collect(cfg))
    _print({"topic": cfg.name, "new_comments": added})


def cmd_synth(args, s):
    from .synthetic import annotate, generate

    cfg = TopicConfig.from_toml(args.config)
    comments, gold, _ = generate(cfg, args.n, args.seed)
    store = Store(args.db or s.db_path)
    added = store.add_comments(comments)
    if args.gold:
        ids = [c.comment_id for c in sample_for_labels([c for c in comments if c.comment_id in gold], args.labels,
                                                       args.seed)]
        with Path(args.gold).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["comment_id", "annotator", "label"])
            w.writerows(annotate(gold, ids, seed=args.seed))
    _print({"topic": cfg.name, "new_comments": added, "gold_file": str(args.gold) if args.gold else None})


def cmd_analyse(args, s):
    cfg = TopicConfig.from_toml(args.config)
    store = Store(args.db or s.db_path)
    out = run(cfg, store, build_sentiment(s.sentiment_model), build_absa(s.absa_model), s.seed)
    print(to_markdown(out.report) if args.markdown else json.dumps(out.report, indent=2))


def cmd_topics(args, s):
    cfg = TopicConfig.from_toml(args.config)
    kept, _ = filter_comments(Store(args.db or s.db_path).comments(cfg.name))
    res = sweep_topics([topic_view(c.text) for c in kept], args.k_min, args.k_max, s.seed)
    _print({"topic": cfg.name, "documents": len(kept), "k": res.k, "npmi": res.coherence, "sweep": res.sweep,
            "topics": [{"words": w, "size": n} for w, n in zip(res.topics, res.sizes)]})


def cmd_sample_labels(args, s):
    cfg = TopicConfig.from_toml(args.config)
    kept, _ = filter_comments(Store(args.db or s.db_path).comments(cfg.name))
    sample = sample_for_labels(kept, args.n, s.seed)
    with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["comment_id", "subreddit", "text", "annotator", "label"])
        for c in sample:
            w.writerow([c.comment_id, c.subreddit, model_view(c.text), "", ""])
    _print({"out": str(args.out), "rows": len(sample)})


def _predictions(args, s, cfg):
    store = Store(args.db or s.db_path)
    out = run(cfg, store, build_sentiment(s.sentiment_model), build_absa(s.absa_model), s.seed)
    return {r["comment_id"]: r["sentiment"] for r in out.results}, out


def cmd_validate(args, s):
    cfg = TopicConfig.from_toml(args.config)
    gold_raw = read_gold(args.gold)
    gold, ties = adjudicate(gold_raw)
    pred, out = _predictions(args, s, cfg)
    _print({"topic": cfg.name, "run_id": out.run_id, "agreement": agreement(gold_raw), "ties_left_out": ties,
            "model": s.sentiment_model, "scores": score_model(pred, gold, seed=s.seed)})


def cmd_compare_cleaning(args, s):
    cfg = TopicConfig.from_toml(args.config)
    gold, _ = adjudicate(read_gold(args.gold))
    kept, _ = filter_comments(Store(args.db or s.db_path).comments(cfg.name))
    kept = [c for c in kept if c.comment_id in gold]
    model = build_sentiment(s.sentiment_model)
    new = model.predict([model_view(c.text) for c in kept])
    old = model.predict([old_clean(c.text) for c in kept])
    _print({"items": len(kept), "model": model.name,
            "model_view": score_model({c.comment_id: p.label for c, p in zip(kept, new)}, gold, seed=s.seed),
            "old_cleaning": score_model({c.comment_id: p.label for c, p in zip(kept, old)}, gold, seed=s.seed)})


def cmd_runs(args, s):
    _print(Store(args.db or s.db_path).runs())


def cmd_demo(args, s):
    from .synthetic import annotate, generate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TopicConfig.from_toml(args.config)
    db = out / "pulse.db"
    store = Store(db)
    comments, gold, _ = generate(cfg, 600, s.seed)
    store.add_comments(comments)
    res = run(cfg, store, build_sentiment("lexicon"), build_absa("lexicon"), s.seed)
    labelled = sample_for_labels([c for c in res.kept if c.comment_id in gold], 300, s.seed)
    gold_path = out / "gold.csv"
    with gold_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["comment_id", "annotator", "label"])
        w.writerows(annotate(gold, [c.comment_id for c in labelled], seed=s.seed))
    gold_raw = read_gold(gold_path)
    adj, ties = adjudicate(gold_raw)
    pred = {r["comment_id"]: r["sentiment"] for r in res.results}
    kept_lab = [c for c in res.kept if c.comment_id in adj]
    model = build_sentiment("lexicon")
    old = model.predict([old_clean(c.text) for c in kept_lab])
    # aspect-level gold: comments with an aspect label whose aspect the detector found
    asp_pred, asp_gold = {}, {}
    for r in res.results:
        g = gold.get(r["comment_id"], {}).get("aspects", {})
        for a in r["aspects"]:
            if a["aspect"] in g:
                key = f"{r['comment_id']}:{a['aspect']}"
                asp_pred[key], asp_gold[key] = a["label"], g[a["aspect"]]
    should_mention = sum(1 for g in gold.values() if g["aspects"])
    topics = sweep_topics([topic_view(c.text) for c in res.kept], 2, 8, s.seed)
    summary = {
        "topic": cfg.name, "comments": len(comments), "report": res.report,
        "validation": {"agreement": agreement(gold_raw), "ties_left_out": ties,
                       "sentiment_model_view": score_model(pred, adj, seed=s.seed),
                       "sentiment_old_cleaning": score_model({c.comment_id: p.label for c, p in zip(kept_lab, old)},
                                                             adj, seed=s.seed),
                       "aspect_sentiment": score_model(asp_pred, asp_gold, seed=s.seed),
                       "aspect_mentions_found": len(asp_gold), "aspect_mentions_in_gold": should_mention},
        "topics": {"k": topics.k, "npmi": topics.coherence, "sweep": topics.sweep, "words": topics.topics},
    }
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(to_markdown(res.report))
    _print({"validation": summary["validation"], "topics": summary["topics"]})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reddit-pulse", description="Validated discourse analysis of Reddit topics.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, func, help_):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--config", type=Path, required=name != "runs" and name != "demo",
                       default=Path("configs/topics/ice-raids.toml") if name == "demo" else None)
        c.add_argument("--db", type=Path)
        c.set_defaults(func=func)
        return c

    add("collect", cmd_collect, "collect comments with PRAW (extra reddit, env credentials)")
    c = add("synth", cmd_synth, "write synthetic comments (and optional simulated labels)")
    c.add_argument("--n", type=int, default=600)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--gold", type=Path)
    c.add_argument("--labels", type=int, default=300)
    c = add("analyse", cmd_analyse, "filter, sentiment and aspect sentiment, with a report")
    c.add_argument("--markdown", action="store_true")
    c = add("topics", cmd_topics, "NMF topics with an NPMI k sweep")
    c.add_argument("--k-min", type=int, default=2)
    c.add_argument("--k-max", type=int, default=8)
    c = add("sample-labels", cmd_sample_labels, "write a subreddit-stratified sample for annotation")
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--out", type=Path, required=True)
    c = add("validate", cmd_validate, "agreement and model scores against hand labels")
    c.add_argument("--gold", type=Path, required=True)
    c = add("compare-cleaning", cmd_compare_cleaning, "model view against the old cleaning, on hand labels")
    c.add_argument("--gold", type=Path, required=True)
    add("runs", cmd_runs, "list the analysis runs")
    c = add("demo", cmd_demo, "offline demo on synthetic comments")
    c.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args, Settings.from_env())
    except (ConfigError, LabelError, ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
