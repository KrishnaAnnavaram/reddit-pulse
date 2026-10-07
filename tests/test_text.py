"""Cleaning views, filters, sentiment and aspects (problems 2, 3, 4, 7 and 9)."""
import pytest

from reddit_pulse.absa import LexiconAbsa, find_mentions
from reddit_pulse.clean import dedupe_key, filter_reason, is_bot, is_english, model_view, topic_view
from reddit_pulse.config import Aspect
from reddit_pulse.legacy import old_clean
from reddit_pulse.sentiment import LexiconSentiment, Prediction, build_sentiment, lexicon_compound


def test_model_view_keeps_negations_case_and_punctuation():
    """Problem 2: the model text keeps 'not', case and '!'."""
    t = "This is NOT good!!! See [the report](https://x.org/a) by u/someone https://y.org/b ok"
    v = model_view(t)
    assert "NOT good!!!" in v and "the report" in v and "@user" in v and v.endswith("http ok")


def test_url_removal_keeps_the_rest_of_the_line():
    """Problem 7: the old greedy pattern deleted all text after a link."""
    t = "Source https://example.org/a says the plan is terrible"
    assert "terrible" not in old_clean(t)
    assert "terrible" in model_view(t) and "terrible" in topic_view(t)


def test_old_cleaning_drops_negations():
    assert old_clean("This is not good") == "good"


def test_topic_view():
    toks = topic_view("> quoted line\nThe Tariffs on STEEL will hurt farmers in 2025 https://x.org")
    assert toks == ["tariffs", "steel", "hurt", "farmers"]


def test_filters():
    seen: set[str] = set()
    assert filter_reason("[deleted]", None, seen) == "deleted"
    assert filter_reason("I am a bot, and this action was performed automatically.", None, seen) == "bot"
    assert is_bot("AutoModerator", "hello there") and is_bot("tariff_bot", "x")
    assert filter_reason("ok", None, seen) == "too_short"
    assert filter_reason("The plan is terrible for us.", None, seen) is None
    assert filter_reason("the PLAN is terrible for us!!", None, seen) == "duplicate"
    assert dedupe_key("A b") == dedupe_key("a  B")
    assert not is_english("zzzz qqqq xxxx yyyy wwww vvvv uuuu tttt")
    assert is_english("Short text")


def test_lexicon_negation_intensifier_and_but():
    s = LexiconSentiment()
    labels = [p.label for p in s.predict(["This is good.", "This is not good.", "The weather report is out.",
                                           "Not a disaster at all.", "It looks fine but the plan is terrible."])]
    assert labels == ["positive", "negative", "neutral", "positive", "negative"]
    assert lexicon_compound("very good") > lexicon_compound("good") > 0
    assert -1 < lexicon_compound("terrible awful horrible disaster") < -0.5


def test_neutral_is_kept_and_output_is_deterministic():
    """Problem 3: no random relabelling of neutral comments."""
    s = LexiconSentiment()
    texts = ["Here is the link.", "When is the hearing?"] * 5
    a, b = s.predict(texts), s.predict(texts)
    assert a == b and {p.label for p in a} == {"neutral"}


def test_failure_is_a_status_not_a_label():
    """Problem 9: a failed prediction has status 'failed' and no label."""
    p = Prediction(None, None, "failed")
    assert p.label is None and p.status == "failed"
    with pytest.raises(ValueError):
        build_sentiment("vader2")


def test_aspects_need_a_mention_and_respect_case():
    """Problem 4: ABSA runs only where the aspect is named. 'ice' is not 'ICE'."""
    aspects = [Aspect("ICE", (r"\bICE\b",), case_sensitive=True), Aspect("raids", (r"\braids?\b",))]
    assert find_mentions("Put some ice on it.", aspects) == []
    ms = find_mentions("Great weather today. The ICE raids are cruel.", aspects)
    assert [(m.aspect, m.sentence) for m in ms] == [("ICE", "The ICE raids are cruel."),
                                                   ("raids", "The ICE raids are cruel.")]
    assert LexiconAbsa().predict([(ms[0].sentence, "ICE")])[0].label == "negative"
