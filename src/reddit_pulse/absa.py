"""Aspect-based sentiment, only where a comment mentions the aspect.

1. `find_mentions` applies the aspect patterns of the topic config (case-sensitive if the config
   says so, so `ICE` the agency is not `ice` the frozen water).
2. The ABSA model scores the sentence that holds the mention, with the aspect name as the pair.
A comment without a mention gets no aspect sentiment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import Aspect
from .sentiment import Prediction, lexicon_compound

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Mention:
    aspect: str
    sentence: str
    start: int


def find_mentions(text: str, aspects: Sequence[Aspect]) -> list[Mention]:
    out = []
    pos = 0
    for sent in SENT_SPLIT.split(text or ""):
        for a in aspects:
            m = a.regex().search(sent)
            if m:
                out.append(Mention(a.name, sent.strip(), pos + m.start()))
        pos += len(sent) + 1
    return out


class AbsaModel(Protocol):
    name: str

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[Prediction]: ...


class LexiconAbsa:
    """Offline baseline: the lexicon score of the sentence that holds the mention."""

    name = "lexicon"

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[Prediction]:
        out = []
        for sentence, _aspect in pairs:
            c = lexicon_compound(sentence)
            out.append(Prediction("positive" if c > 0.05 else "negative" if c < -0.05 else "neutral", round(c, 4)))
        return out


class TransformerAbsa:  # pragma: no cover - needs the optional extra and a model download
    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1", batch_size: int = 16, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
        self.device = device
        self.batch_size = batch_size
        self.name = f"hf:{model_name}"
        self.labels = [self.model.config.id2label[i].lower() for i in range(len(self.model.config.id2label))]

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = list(pairs[i:i + self.batch_size])
            try:
                enc = self.tok([s for s, _ in batch], [a for _, a in batch], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(self.device)
                with self.torch.inference_mode():
                    probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                for p in probs:
                    k = int(p.argmax())
                    out.append(Prediction(self.labels[k], round(float(p[k]), 4)))
            except Exception:
                out.extend(Prediction(None, None, "failed") for _ in batch)
        return out


def build_absa(name: str) -> AbsaModel:
    if name == "lexicon":
        return LexiconAbsa()
    if name.startswith("hf:") or "/" in name:
        return TransformerAbsa(name.removeprefix("hf:"))
    raise ValueError(f"unknown ABSA model {name!r}; use lexicon or a Hugging Face model id")
