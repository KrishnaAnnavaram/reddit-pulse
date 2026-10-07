"""SQLite store. One row for each comment (primary key `comment_id`), no usernames.

The author is stored only as `author_hash` = HMAC-SHA256(PULSE_HASH_SALT, username), first 16 hex
digits. Without a salt, the author column stays empty. User mentions in the text are replaced.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .clean import MENTION

DDL = """
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY, topic TEXT NOT NULL, thread_id TEXT NOT NULL, subreddit TEXT NOT NULL,
    created_utc INTEGER NOT NULL, text TEXT NOT NULL, author_hash TEXT, score INTEGER, is_bot INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, topic TEXT NOT NULL, kind TEXT NOT NULL, created TEXT NOT NULL, params TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL, comment_id TEXT NOT NULL, sentiment TEXT, sentiment_score REAL, status TEXT NOT NULL,
    aspects TEXT NOT NULL, PRIMARY KEY (run_id, comment_id));
CREATE INDEX IF NOT EXISTS ix_comments_topic ON comments(topic);
"""


@dataclass(frozen=True)
class Comment:
    comment_id: str
    topic: str
    thread_id: str
    subreddit: str
    created_utc: int
    text: str
    author_hash: str | None = None
    score: int | None = None
    is_bot: bool = False


def hash_author(author: str | None, salt: str | None) -> str | None:
    if not author or not salt or author in ("[deleted]", "None"):
        return None
    return hmac.new(salt.encode(), author.encode(), hashlib.sha256).hexdigest()[:16]


def scrub(text: str) -> str:
    return MENTION.sub("u/[user]", text or "")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(DDL)

    def close(self) -> None:
        self.con.close()

    def add_comments(self, comments: Iterable[Comment]) -> int:
        """Insert new comments. A known `comment_id` is ignored. Return the number of new rows."""
        before = self.con.total_changes
        with self.con:
            self.con.executemany(
                "INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?,?,?,?,?)",
                [(c.comment_id, c.topic, c.thread_id, c.subreddit, int(c.created_utc), scrub(c.text), c.author_hash,
                  c.score, int(c.is_bot)) for c in comments])
        return self.con.total_changes - before

    def comments(self, topic: str) -> list[Comment]:
        rows = self.con.execute("SELECT * FROM comments WHERE topic = ? ORDER BY created_utc, comment_id", (topic,))
        return [Comment(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], bool(r[8])) for r in rows]

    def topics(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT topic FROM comments ORDER BY topic")]

    def add_run(self, run_id: str, topic: str, kind: str, created: str, params: dict) -> None:
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                             (run_id, topic, kind, created, json.dumps(params, sort_keys=True)))

    def add_results(self, run_id: str, rows: list[dict]) -> None:
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?)",
                                 [(run_id, r["comment_id"], r["sentiment"], r["sentiment_score"], r["status"],
                                   json.dumps(r["aspects"], sort_keys=True)) for r in rows])

    def results(self, run_id: str) -> list[dict]:
        rows = self.con.execute("SELECT comment_id, sentiment, sentiment_score, status, aspects FROM results "
                                "WHERE run_id = ? ORDER BY comment_id", (run_id,))
        return [{"comment_id": r[0], "sentiment": r[1], "sentiment_score": r[2], "status": r[3],
                 "aspects": json.loads(r[4])} for r in rows]

    def runs(self) -> list[dict]:
        return [{"run_id": r[0], "topic": r[1], "kind": r[2], "created": r[3], "params": json.loads(r[4])}
                for r in self.con.execute("SELECT * FROM runs ORDER BY created")]


def to_dict(c: Comment) -> dict:
    return asdict(c)
