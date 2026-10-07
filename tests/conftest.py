import os
from pathlib import Path

import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from reddit_pulse.config import TopicConfig  # noqa: E402
from reddit_pulse.synthetic import generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ice_cfg():
    return TopicConfig.from_toml(ROOT / "configs" / "topics" / "ice-raids.toml")


@pytest.fixture(scope="session")
def synth(ice_cfg):
    return generate(ice_cfg, 300, seed=3)
