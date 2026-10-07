"""A copy of the OLD cleaning steps, kept only to measure their effect (see `reddit-pulse compare-cleaning`).

Do not use these functions for analysis. They remove negations with the stop words, and the greedy
URL pattern deletes all text after a link.
"""
from __future__ import annotations

import re

OLD_STOP = frozenset("""
i me my myself we our ours you your he him his she her it its they them their what which who whom this that these
those am is are was were be been being have has had do does did a an the and but if or because as until while of at
by for with about against between into through during before after to from in out on off over under again then once
here there when where why how all any both each few more most other some such no nor not only own same so than too
very s t can will just don don't should now isn't aren't wasn't doesn't didn't won't can't
""".split())


def old_clean(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"https*://.*", "", t)          # greedy: removes the rest of the line
    t = re.sub(r"[^a-z\s']", " ", t)
    return " ".join(w for w in t.split() if w not in OLD_STOP)
