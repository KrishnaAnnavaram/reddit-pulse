"""Two text views and the comment filters.

- `model_view`: the text for transformer and lexicon models. It keeps case, punctuation, emojis,
  negations and all words. It only replaces URLs with `http`, user mentions with `@user`, and
  markdown links with their text. A URL is removed alone, never the rest of the line.
- `topic_view`: tokens for topic models only: lower case, no URL, no stop word, no number.
- Filters: deleted or removed comments, bots, duplicates (normalised text), non-English text,
  and comments that are too short.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

URL = re.compile(r"https?://[^\s)\]]+|www\.[^\s)\]]+", re.I)
MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]*\)", re.I)
MENTION = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+")
QUOTE = re.compile(r"^\s*&gt;.*$|^\s*>.*$", re.M)

TOPIC_STOP = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers him
his how i if in into is it its itself just me more most my no nor not of off on once only or other our out over own
same she should so some such than that the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours im dont doesnt cant wont
isnt arent thats theyre youre ive id get got like one people think even really know going make much many well still
right yes yeah lol gt amp http https www com
""".split())
BOT_NAMES = re.compile(r"(?i)^(automoderator|.*bot)$")
BOT_TEXT = re.compile(r"(?i)i am a bot|this action was performed automatically")
EN_COMMON = frozenset("the be to of and a in that have it for not on with he as you do at this but his by from they we "
                      "say her she or an will my one all would there their what so up out if about who get which go me "
                      "is are was".split())


def model_view(text: str) -> str:
    t = MD_LINK.sub(r"\1", text or "")
    t = URL.sub("http", t)
    t = MENTION.sub("@user", t)
    return re.sub(r"\s+", " ", t).strip()


def topic_view(text: str) -> list[str]:
    t = QUOTE.sub(" ", text or "")
    t = MD_LINK.sub(r"\1", t)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t).lower().replace("'", "")
    return [w for w in re.findall(r"[a-z]{3,}", t) if w not in TOPIC_STOP]


def is_deleted(text: str) -> bool:
    return (text or "").strip() in ("[deleted]", "[removed]", "")


def is_bot(author: str | None, text: str) -> bool:
    return bool((author and BOT_NAMES.match(author)) or BOT_TEXT.search(text or ""))


def is_english(text: str, min_ratio: float = 0.05, min_words: int = 8) -> bool:
    """A rough check: a text of 8 words or more must have at least 5% common English words.
    A shorter text passes when it has a Latin letter."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    if len(words) < min_words:
        return bool(words)
    return sum(w in EN_COMMON for w in words) / len(words) >= min_ratio


def dedupe_key(text: str) -> str:
    norm = " ".join(re.findall(r"[a-z0-9]+", model_view(text).lower()))
    return hashlib.sha1(norm.encode()).hexdigest()


def filter_reason(text: str, author: str | None, seen: set[str], min_words: int = 3) -> str | None:
    """Return why a comment is dropped, or None to keep it. Adds kept texts to `seen`."""
    if is_deleted(text):
        return "deleted"
    if is_bot(author, text):
        return "bot"
    if len(re.findall(r"\w+", text)) < min_words:
        return "too_short"
    if not is_english(text):
        return "not_english"
    key = dedupe_key(text)
    if key in seen:
        return "duplicate"
    seen.add(key)
    return None


def filter_counts(reasons: list[str | None]) -> dict[str, int]:
    c = Counter(r or "kept" for r in reasons)
    return dict(sorted(c.items()))
