"""Synthetic comments with known labels, for the offline demo and the tests.

The generator writes comments from templates: positive, negative, negated, neutral and sarcastic
sentences about the aspects of a topic, plus sub-theme sentences for the topic model. It adds bots,
copy-paste duplicates, deleted comments, links in the middle of a sentence and the word "ice" in
an unrelated sense. Each subreddit has its own lean. Two simulated annotators copy the true label
with independent random flips. No real comment or user is in this data.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from .config import TopicConfig
from .store import Comment, hash_author

POS = ["I think the {A} plan is good for everyone.", "Honestly the {A} decision is great.",
       "This {A} policy is reasonable and fair.", "I support the {A} changes, they protect people.",
       "Glad to see {A} handled well for once!"]
NEG = ["The {A} plan is terrible.", "This {A} decision is cruel and wrong.", "I hate how the {A} rules hurt families.",
       "What a disaster the {A} rollout is.", "The {A} policy is unconstitutional and dangerous."]
NEG_NEGATED = ["The {A} plan is not good at all.", "I don't support the {A} changes.",
               "This {A} decision is not fair to anyone.", "Nobody thinks {A} is helpful."]
POS_NEGATED = ["The {A} plan is not a disaster.", "Honestly the {A} rules are not bad.", "I don't hate the {A} decision."]
NEUTRAL = ["Does anyone have the link to the {A} report?", "The hearing on {A} is scheduled for Tuesday.",
           "Here is the full text of the {A} order.", "How many states filed about {A} so far?"]
SARCASM = ["Oh great, another {A} mess. Just great.", "Wow, what a brilliant {A} idea, said nobody."]
OFF_TOPIC = ["Put some ice on it and wait an hour.", "What time does the thread close today?",
             "I read this in the morning paper on the train."]
OPENERS = ["", "Honestly, ", "Look, ", "Update: ", "As a local, ", "From what I read, ", "My take: ", "Well, ",
           "Again, ", "For the record, "]
BOT = "I am a bot, and this action was performed automatically. Please contact the moderators."
COPYPASTA = "Copy this to every thread: the {A} plan is a total disaster for all of us."

TOPIC_TERMS: dict[str, dict] = {
    "ice-raids": {"aspects": {"ICE": ["ICE", "ICE agents"], "raids": ["raids", "immigration raids"]},
                  "themes": [["workplace", "factory", "workers", "detained"], ["schools", "children", "parents", "fear"],
                             ["courts", "warrant", "judge", "lawyers"]]},
    "tariffs": {"aspects": {"tariffs": ["tariffs", "steel tariffs"], "prices": ["prices", "grocery prices"]},
                "themes": [["farmers", "soybeans", "exports", "harvest"], ["china", "canada", "mexico", "trade"],
                           ["cars", "steel", "factories", "jobs"]]},
}
LEANS = {0: (0.45, 0.25, 0.30), 1: (0.15, 0.25, 0.60), 2: (0.30, 0.40, 0.30)}   # (pos, neutral, neg)


def generate(cfg: TopicConfig, n: int = 600, seed: int = 0, salt: str = "synthetic-salt") -> tuple[list[Comment], dict, dict]:
    """Return comments, gold labels {comment_id: {"sentiment", "aspects": {name: label}}} and authors {id: name}."""
    rng = random.Random(seed)
    terms = TOPIC_TERMS.get(cfg.name, TOPIC_TERMS["tariffs"])
    aspect_names = [a.name for a in cfg.aspects] or list(terms["aspects"])
    forms = {a: terms["aspects"].get(a, [a]) for a in aspect_names}
    start = datetime(cfg.start.year, cfg.start.month, cfg.start.day, tzinfo=timezone.utc).timestamp()
    end = datetime(cfg.end.year, cfg.end.month, cfg.end.day, 23, tzinfo=timezone.utc).timestamp()
    comments: list[Comment] = []
    gold: dict[str, dict] = {}
    authors: dict[str, str] = {}
    for i in range(n):
        sub_i = i % len(cfg.subreddits)
        sub = cfg.subreddits[sub_i]
        cid = f"c{seed}_{i:05d}"
        user = f"user{rng.randint(1, n // 3)}"
        r = rng.random()
        aspect = rng.choice(aspect_names)
        A = rng.choice(forms[aspect])
        is_bot = False
        if r < 0.03:
            text, label, asp_label, user, is_bot = BOT, None, None, "AutoModerator", True
        elif r < 0.05:
            text, label, asp_label = "[deleted]", None, None
        elif r < 0.08:
            text, label, asp_label = COPYPASTA.format(A=A), "negative", "negative"
        elif r < 0.13:
            text, label, asp_label = rng.choice(OFF_TOPIC), "neutral", None
        elif r < 0.17:
            text, label, asp_label = rng.choice(SARCASM).format(A=A), "negative", "negative"
        else:
            pos, neu, _neg = LEANS[sub_i % 3]
            q = rng.random()
            if q < pos:
                tpl = rng.choice(POS + POS + POS_NEGATED)
                label = "positive"
            elif q < pos + neu:
                tpl, label = rng.choice(NEUTRAL), "neutral"
            else:
                tpl = rng.choice(NEG + NEG + NEG_NEGATED)
                label = "negative"
            opener = rng.choice(OPENERS)
            if opener and not tpl.startswith(("I ", "{A}")):
                tpl = tpl[0].lower() + tpl[1:]
            text, asp_label = opener + tpl.format(A=A), label
        if label is not None and rng.random() < 0.8:
            w1, w2, w3 = rng.sample(rng.choice(terms["themes"]), 3)
            text += f" Think about {w1}, {w2} and {w3}."
        if label is not None and rng.random() < 0.15:
            text = f"Source https://example.org/item{i} says: {text}"
        comments.append(Comment(cid, cfg.name, f"t_{sub}_{i % 7}", sub, int(rng.uniform(start, end)), text,
                                hash_author(user, salt), rng.randint(-5, 50), is_bot))
        authors[cid] = user
        if label is not None:
            gold[cid] = {"sentiment": label, "aspects": {aspect: asp_label} if asp_label else {}}
    # copy-paste duplicates of the copypasta, posted again by other users
    dupes = [c for c in comments if c.text.startswith("Copy this")][:5]
    for k, c in enumerate(dupes):
        comments.append(Comment(f"{c.comment_id}_d{k}", c.topic, c.thread_id, c.subreddit, c.created_utc + 60, c.text,
                                hash_author(f"copy{k}", salt), 1, False))
    return comments, gold, authors


def annotate(gold: dict[str, dict], ids: list[str], flip: float = 0.08, seed: int = 0) -> list[tuple[str, str, str]]:
    """Two simulated annotators: (comment_id, annotator, label) rows."""
    rng = random.Random(seed)
    rows = []
    for cid in ids:
        true = gold[cid]["sentiment"]
        for ann in ("ann_a", "ann_b"):
            lab = true if rng.random() > flip else rng.choice([l for l in ("negative", "neutral", "positive") if l != true])
            rows.append((cid, ann, lab))
    return rows
