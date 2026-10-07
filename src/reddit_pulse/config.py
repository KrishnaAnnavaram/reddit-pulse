"""Settings (environment variables, optional .env) and topic configs (TOML)."""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    pass


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    p = Path(path)
    out: dict[str, str] = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip().strip("'\"")
    return out


@dataclass(frozen=True)
class Settings:
    db_path: Path
    hash_salt: str | None
    seed: int
    sentiment_model: str
    absa_model: str
    reddit_client_id: str | None
    reddit_client_secret: str | None
    reddit_user_agent: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: str | Path | None = ".env") -> "Settings":
        e: dict[str, str] = {}
        if dotenv is not None:
            e.update(load_dotenv(dotenv))
        e.update(os.environ if env is None else env)
        try:
            seed = int(e.get("PULSE_SEED") or 42)
        except ValueError as exc:
            raise ConfigError("PULSE_SEED must be an integer") from exc
        return cls(
            db_path=Path(e.get("PULSE_DB") or "artifacts/pulse.db"),
            hash_salt=e.get("PULSE_HASH_SALT") or None,
            seed=seed,
            sentiment_model=e.get("PULSE_SENTIMENT_MODEL") or "lexicon",
            absa_model=e.get("PULSE_ABSA_MODEL") or "lexicon",
            reddit_client_id=e.get("REDDIT_CLIENT_ID") or None,
            reddit_client_secret=e.get("REDDIT_CLIENT_SECRET") or None,
            reddit_user_agent=e.get("REDDIT_USER_AGENT") or None,
        )


@dataclass(frozen=True)
class Aspect:
    name: str
    patterns: tuple[str, ...]
    case_sensitive: bool = False

    def regex(self) -> re.Pattern:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile("|".join(f"(?:{p})" for p in self.patterns), flags)


@dataclass(frozen=True)
class TopicConfig:
    name: str
    subreddits: tuple[str, ...]
    queries: tuple[str, ...]
    start: date
    end: date
    aspects: tuple[Aspect, ...] = field(default_factory=tuple)
    max_threads_per_query: int = 25

    @classmethod
    def from_toml(cls, path: str | Path) -> "TopicConfig":
        d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            aspects = tuple(Aspect(a["name"], tuple(a["patterns"]), bool(a.get("case_sensitive", False)))
                            for a in d.get("aspects", []))
            cfg = cls(d["name"], tuple(d["subreddits"]), tuple(d["queries"]), date.fromisoformat(str(d["start"])),
                      date.fromisoformat(str(d["end"])), aspects, int(d.get("max_threads_per_query", 25)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        if cfg.end < cfg.start:
            raise ConfigError(f"{path}: end is before start")
        if not cfg.subreddits or not cfg.queries:
            raise ConfigError(f"{path}: give at least one subreddit and one query")
        for a in cfg.aspects:
            try:
                a.regex()
            except re.error as exc:
                raise ConfigError(f"{path}: bad pattern in aspect {a.name}: {exc}") from exc
        return cfg
