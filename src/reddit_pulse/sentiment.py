"""Sentiment models with three labels: negative, neutral, positive. Neutral stays neutral.

- `LexiconSentiment`: an offline baseline with negation scope (3 tokens), intensifiers and a
  "but" rule. It reads the model view, so "not good" is negative.
- `TransformerSentiment`: a Hugging Face sequence classifier (default
  `cardiffnlp/twitter-roberta-base-sentiment-latest`, extra `transformers`), batched, imported lazily.
A failure gives the status `failed` for that comment. A failure is never a sentiment label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

LABELS = ("negative", "neutral", "positive")

POSITIVE = {
    "good": 1.5, "great": 2.0, "excellent": 2.5, "fair": 1.0, "support": 1.5, "supports": 1.5, "love": 2.0,
    "like": 1.0, "agree": 1.0, "helpful": 1.5, "reasonable": 1.2, "smart": 1.5, "right": 0.8, "win": 1.5,
    "benefit": 1.5, "benefits": 1.5, "protect": 1.2, "protects": 1.2, "safe": 1.2, "glad": 1.5, "happy": 1.8,
    "hope": 1.0, "hopeful": 1.2, "positive": 1.5, "strong": 1.0, "works": 1.0, "welcome": 1.2, "best": 2.0,
    "better": 1.2, "thanks": 1.0, "thank": 1.0, "proud": 1.5, "approve": 1.5, "effective": 1.5, "wise": 1.5,
    "brilliant": 2.2, "needed": 1.0, "finally": 0.8, "justice": 1.2, "freedom": 1.0, "equal": 1.0, "respect": 1.2,
}
NEGATIVE = {
    "bad": -1.5, "terrible": -2.5, "awful": -2.5, "horrible": -2.5, "wrong": -1.5, "hate": -2.2, "disgusting": -2.5,
    "cruel": -2.2, "unfair": -1.8, "illegal": -1.5, "unconstitutional": -2.0, "stupid": -2.0, "dumb": -1.8,
    "disaster": -2.5, "harm": -1.8, "hurts": -1.5, "hurt": -1.5, "scary": -1.8, "afraid": -1.5, "fear": -1.5,
    "angry": -1.8, "outrage": -2.2, "outrageous": -2.2, "worse": -1.5, "worst": -2.5, "fail": -1.8, "fails": -1.8,
    "failed": -1.8, "chaos": -2.0, "mess": -1.5, "against": -0.8, "oppose": -1.5, "abuse": -2.2, "racist": -2.2,
    "corrupt": -2.2, "lies": -2.0, "lie": -1.8, "expensive": -1.2, "costly": -1.2, "inflation": -1.0,
    "sad": -1.5, "shameful": -2.2, "dangerous": -2.0, "threat": -1.5, "attack": -1.5, "ruin": -2.0, "ruins": -2.0,
}
NEGATORS = {"not", "no", "never", "nobody", "nothing", "neither", "nor", "isnt", "isn't", "dont", "don't", "doesnt",
            "doesn't", "cant", "can't", "wont", "won't", "aren't", "arent", "wasn't", "wasnt", "hardly", "without"}
INTENSIFIERS = {"very": 1.4, "really": 1.3, "extremely": 1.6, "so": 1.2, "totally": 1.4, "absolutely": 1.5,
                "completely": 1.4, "slightly": 0.6, "somewhat": 0.7}


class SentimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str | None
    score: float | None      # compound score (lexicon) or probability of the label (transformer)
    status: str = "ok"       # ok / failed


class SentimentModel(Protocol):
    name: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[!?]", text.lower())


def lexicon_compound(text: str, lexicon: dict[str, float] | None = None) -> float:
    lex = lexicon or {**POSITIVE, **NEGATIVE}
    toks = _tokens(text)
    total = 0.0
    but_at = max((i for i, t in enumerate(toks) if t == "but"), default=-1)
    for i, t in enumerate(toks):
        v = lex.get(t)
        if v is None:
            continue
        window = toks[max(0, i - 3):i]
        if any(w in NEGATORS for w in window):
            v = -0.75 * v
        if i > 0 and toks[i - 1] in INTENSIFIERS:
            v *= INTENSIFIERS[toks[i - 1]]
        if but_at >= 0:
            v *= 1.5 if i > but_at else 0.5
        total += v
    if total and toks.count("!"):
        total *= 1 + 0.1 * min(3, toks.count("!"))
    return total / ((total * total + 15) ** 0.5)  # normalised to (-1, 1)


class LexiconSentiment:
    name = "lexicon"

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out = []
        for t in texts:
            c = lexicon_compound(t)
            label = "positive" if c > self.threshold else "negative" if c < -self.threshold else "neutral"
            out.append(Prediction(label, round(c, 4)))
        return out


class TransformerSentiment:  # pragma: no cover - needs the optional extra and a model download
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", batch_size: int = 32,
                 device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
        self.device = device
        self.batch_size = batch_size
        self.name = f"hf:{model_name}"
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        legacy = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
        self.labels = [legacy.get(id2label[i], id2label[i]) for i in range(len(id2label))]
        if set(self.labels) != set(LABELS):
            raise SentimentError(f"model labels {self.labels} are not {LABELS}")

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            try:
                enc = self.tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                with self.torch.inference_mode():
                    probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                for p in probs:
                    k = int(p.argmax())
                    out.append(Prediction(self.labels[k], round(float(p[k]), 4)))
            except Exception:  # one bad batch: mark each comment of the batch as failed, keep going
                out.extend(Prediction(None, None, "failed") for _ in batch)
        return out


def build_sentiment(name: str) -> SentimentModel:
    if name == "lexicon":
        return LexiconSentiment()
    if name.startswith("hf:") or "/" in name:
        return TransformerSentiment(name.removeprefix("hf:"))
    raise ValueError(f"unknown sentiment model {name!r}; use lexicon or a Hugging Face model id")
