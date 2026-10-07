"""Config-driven collection: search each subreddit with each query inside the date window, then read
the FULL comment tree of each thread (`replace_more(limit=None)`). Credentials come only from the
environment. Rate-limit errors get an exponential back-off.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Iterator, Protocol

from .config import Settings, TopicConfig
from .store import Comment, hash_author


class CollectError(RuntimeError):
    pass


class Collector(Protocol):
    def collect(self, cfg: TopicConfig) -> Iterator[Comment]: ...


def in_window(created_utc: float, cfg: TopicConfig) -> bool:
    d = datetime.fromtimestamp(created_utc, tz=timezone.utc).date()
    return cfg.start <= d <= cfg.end


def with_backoff(fn: Callable, retries: int = 4, base_s: float = 2.0, sleep: Callable = time.sleep):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # praw raises several exception types for HTTP 429
            if attempt == retries or "429" not in str(exc) and "RATELIMIT" not in str(exc).upper():
                raise
            sleep(base_s * 2 ** attempt)
    return None


class RedditCollector:  # pragma: no cover - network
    def __init__(self, settings: Settings):
        if not (settings.reddit_client_id and settings.reddit_client_secret and settings.reddit_user_agent):
            raise CollectError("set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT in the environment")
        if not settings.hash_salt:
            raise CollectError("set PULSE_HASH_SALT: authors are stored only as salted hashes")
        import praw

        self.reddit = praw.Reddit(client_id=settings.reddit_client_id, client_secret=settings.reddit_client_secret,
                                  user_agent=settings.reddit_user_agent, check_for_async=False)
        self.reddit.read_only = True
        self.salt = settings.hash_salt

    def collect(self, cfg: TopicConfig) -> Iterator[Comment]:
        seen_threads: set[str] = set()
        for sub in cfg.subreddits:
            for query in cfg.queries:
                results = with_backoff(lambda: list(self.reddit.subreddit(sub).search(
                    query, sort="new", time_filter="all", limit=cfg.max_threads_per_query)))
                for thread in results:
                    if thread.id in seen_threads or not in_window(thread.created_utc, cfg):
                        continue
                    seen_threads.add(thread.id)
                    with_backoff(lambda: thread.comments.replace_more(limit=None))
                    for c in thread.comments.list():
                        if not in_window(c.created_utc, cfg):
                            continue
                        author = str(c.author) if c.author else None
                        yield Comment(c.id, cfg.name, thread.id, sub, int(c.created_utc), c.body,
                                      hash_author(author, self.salt), int(c.score),
                                      is_bot=bool(author and (author.lower() == "automoderator" or author.lower().endswith("bot"))))
