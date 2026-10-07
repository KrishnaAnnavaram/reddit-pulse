"""Topics on the topic view, with a seeded k sweep chosen by NPMI coherence.

The default is NMF on TF-IDF (scikit-learn). `BertopicModel` (extra `bertopic`) uses the same
interface with a seeded UMAP. Coherence is the mean NPMI of the top-word pairs of each topic, with
document co-occurrence counts from the same corpus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TopicResult:
    k: int
    topics: list[list[str]]
    coherence: float
    sizes: list[int]
    sweep: dict[int, float]


def npmi_coherence(topics: list[list[str]], docs: list[set[str]]) -> float:
    n = len(docs)
    if n == 0:
        return 0.0
    df: dict[str, int] = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    scores = []
    for words in topics:
        for a, b in combinations(words, 2):
            pa, pb = df.get(a, 0) / n, df.get(b, 0) / n
            pab = sum(1 for d in docs if a in d and b in d) / n
            if pab == 0 or pa == 0 or pb == 0:
                scores.append(-1.0)
            elif pab == 1.0:
                scores.append(1.0)
            else:
                scores.append(math.log(pab / (pa * pb)) / -math.log(pab))
    return float(np.mean(scores)) if scores else 0.0


def fit_nmf(token_docs: list[list[str]], k: int, seed: int, top_n: int = 8):
    texts = [" ".join(t) for t in token_docs]
    vec = TfidfVectorizer(min_df=2, max_df=0.9)
    X = vec.fit_transform(texts)
    model = NMF(n_components=k, random_state=seed, init="nndsvda", max_iter=500)
    W = model.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    topics = [list(vocab[np.argsort(row)[::-1][:top_n]]) for row in model.components_]
    sizes = np.bincount(W.argmax(axis=1), minlength=k).tolist()
    return topics, sizes


def sweep_topics(token_docs: list[list[str]], k_min: int = 2, k_max: int = 8, seed: int = 42,
                 top_n: int = 8) -> TopicResult:
    docs = [d for d in token_docs if d]
    if len(docs) < 10:
        raise ValueError("topic modelling needs at least 10 non-empty documents")
    sets = [set(d) for d in docs]
    best: TopicResult | None = None
    sweep: dict[int, float] = {}
    for k in range(k_min, min(k_max, len(docs) - 1) + 1):
        topics, sizes = fit_nmf(docs, k, seed, top_n)
        c = round(npmi_coherence(topics, sets), 4)
        sweep[k] = c
        if best is None or c > best.coherence:
            best = TopicResult(k, topics, c, sizes, sweep)
    assert best is not None
    best.sweep = sweep
    return best


class BertopicModel:  # pragma: no cover - optional extra
    def __init__(self, seed: int = 42):
        from bertopic import BERTopic
        from umap import UMAP

        self.model = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=seed))

    def fit(self, texts: list[str]) -> list[list[str]]:
        self.model.fit(texts)
        info = self.model.get_topics()
        return [[w for w, _ in words[:8]] for t, words in sorted(info.items()) if t != -1]
