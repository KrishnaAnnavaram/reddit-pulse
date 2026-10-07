"""Config, store, pipeline, validation, report, topics and CLI (problems 1, 5, 6, 8 and 10)."""
import csv
import json
import os
from datetime import date

import pytest

from reddit_pulse.absa import LexiconAbsa
from reddit_pulse.cli import main
from reddit_pulse.clean import topic_view
from reddit_pulse.collect import CollectError, RedditCollector, in_window, with_backoff
from reddit_pulse.config import ConfigError, Settings, TopicConfig
from reddit_pulse.pipeline import analyse, filter_comments, run
from reddit_pulse.report import shares, to_markdown
from reddit_pulse.sentiment import LexiconSentiment
from reddit_pulse.store import Comment, Store, hash_author, scrub
from reddit_pulse.synthetic import annotate
from reddit_pulse.topics import npmi_coherence, sweep_topics
from reddit_pulse.validate import LabelError, adjudicate, agreement, read_gold, sample_for_labels, score_model

from conftest import ROOT


def test_all_topic_configs_load():
    names = [TopicConfig.from_toml(p).name for p in sorted((ROOT / "configs" / "topics").glob("*.toml"))]
    assert names == ["birthright-citizenship", "ice-raids", "lgbtq-rights", "tariffs"]


def test_bad_topic_config(tmp_path):
    p = tmp_path / "t.toml"
    p.write_text('name = "x"\nsubreddits = ["a"]\nqueries = ["q"]\nstart = "2025-03-01"\nend = "2025-01-01"\n',
                 encoding="utf-8")
    with pytest.raises(ConfigError):
        TopicConfig.from_toml(p)
    p.write_text('name = "x"\nsubreddits = ["a"]\nqueries = ["q"]\nstart = "2025-01-01"\nend = "2025-02-01"\n'
                 '[[aspects]]\nname = "bad"\npatterns = ["(unclosed"]\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        TopicConfig.from_toml(p)


def test_credentials_only_from_the_environment():
    """Problem 1: no credential in the code. The collector refuses to start without them."""
    s = Settings.from_env({}, dotenv=None)
    assert s.reddit_client_id is None
    with pytest.raises(CollectError):
        RedditCollector(s)
    s2 = Settings.from_env({"REDDIT_CLIENT_ID": "a", "REDDIT_CLIENT_SECRET": "b", "REDDIT_USER_AGENT": "c"}, dotenv=None)
    with pytest.raises(CollectError):
        RedditCollector(s2)  # no PULSE_HASH_SALT


def test_window_and_backoff(ice_cfg):
    assert in_window(1738000000, ice_cfg) and not in_window(1600000000, ice_cfg)
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("received 429 HTTP response")
        return "ok"

    assert with_backoff(flaky, sleep=waits.append) == "ok" and waits == [2.0, 4.0]
    with pytest.raises(ValueError):
        with_backoff(lambda: (_ for _ in ()).throw(ValueError("other")), sleep=waits.append)


def test_store_hashes_authors_scrubs_mentions_and_dedupes(tmp_path):
    """Problem 10: no username is stored."""
    assert hash_author("alice", "salt") == hash_author("alice", "salt") != hash_author("alice", "pepper")
    assert hash_author("alice", None) is None and "alice" not in hash_author("alice", "s")
    assert scrub("thanks u/alice and /u/bob") == "thanks u/[user] and u/[user]"
    st = Store(tmp_path / "p.db")
    c = Comment("c1", "t", "th", "politics", 1738000000, "hi u/alice", hash_author("alice", "s"))
    assert st.add_comments([c, c]) == 1 and st.add_comments([c]) == 0
    got = st.comments("t")[0]
    assert got.text == "hi u/[user]" and got.author_hash == c.author_hash
    raw = (tmp_path / "p.db").read_bytes()
    assert b"alice" not in raw
    st.close()


def test_pipeline_end_to_end(tmp_path, ice_cfg, synth):
    comments, gold, _ = synth
    st = Store(tmp_path / "p.db")
    st.add_comments(comments)
    out = run(ice_cfg, st, LexiconSentiment(), LexiconAbsa(), seed=1)
    rep = out.report
    assert rep["filter_counts"]["bot"] > 0 and rep["filter_counts"]["deleted"] > 0
    assert rep["filter_counts"]["duplicate"] >= 5 and rep["failed"] == 0
    assert set(rep["by_subreddit"]) == set(ice_cfg.subreddits)                 # problem 6: stratified
    assert set(rep["aspects"]) == {"ICE", "raids"} and rep["aspects"]["ICE"]["mentions"] > 0
    assert sum(rep["sentiment"][l]["share"] for l in ("negative", "neutral", "positive")) == pytest.approx(1)
    assert len(st.results(out.run_id)) == rep["analysed"] and st.runs()[0]["params"]["sentiment"] == "lexicon"
    again = run(ice_cfg, st, LexiconSentiment(), LexiconAbsa(), seed=1)
    assert again.run_id == out.run_id and again.results == out.results          # deterministic
    assert "| r/politics |" in to_markdown(rep)


def test_comments_without_a_mention_get_no_aspect(ice_cfg):
    c = [Comment("a", "ice-raids", "t", "news", 1738000000, "Put some ice on the wound and rest today."),
         Comment("b", "ice-raids", "t", "news", 1738000000, "The ICE raids are cruel and wrong today.")]
    _, results, _ = analyse(ice_cfg, c, LexiconSentiment(), LexiconAbsa())
    assert results[0]["aspects"] == [] and {a["aspect"] for a in results[1]["aspects"]} == {"ICE", "raids"}


def test_shares_have_intervals():
    s = shares(["positive"] * 30 + ["negative"] * 70, seed=0)
    assert s["n"] == 100 and s["negative"]["share"] == 0.7
    lo, hi = s["negative"]["ci95"]
    assert lo < 0.7 < hi and s["neutral"]["share"] == 0


def test_validation_tools(tmp_path, synth):
    """Problem 5: agreement, adjudicated gold, macro-F1 with an interval."""
    comments, gold, _ = synth
    ids = [c for c in gold][:120]
    p = tmp_path / "g.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["comment_id", "annotator", "label"])
        w.writerows(annotate(gold, ids, flip=0.1, seed=1))
    raw = read_gold(p)
    ag = agreement(raw)
    assert 0.5 < ag["kappa"] < 1 and ag["items"] == 120
    adj, ties = adjudicate(raw)
    assert len(adj) + ties == 120
    perfect = score_model({k: v for k, v in adj.items()}, adj, n_boot=50)
    assert perfect["macro_f1"] == 1.0 and perfect["macro_f1_ci95"] == [1.0, 1.0]
    p.write_text("comment_id,annotator,label\nx,a,angry\n", encoding="utf-8")
    with pytest.raises(LabelError):
        read_gold(p)
    with pytest.raises(LabelError):
        score_model({}, adj)


def test_sample_for_labels_is_stratified(synth):
    comments, _, _ = synth
    sample = sample_for_labels(comments, 30, seed=0)
    subs = [c.subreddit for c in sample]
    assert len(sample) == 30 and max(subs.count(s) for s in set(subs)) == 10


def test_topics_sweep_and_npmi(synth):
    comments, _, _ = synth
    kept, _ = filter_comments(comments)
    res = sweep_topics([topic_view(c.text) for c in kept], 2, 5, seed=0)
    assert res.k in res.sweep and set(res.sweep) == {2, 3, 4, 5}
    assert sweep_topics([topic_view(c.text) for c in kept], 2, 5, seed=0).topics == res.topics  # seeded
    assert npmi_coherence([["a", "b"]], [{"a", "b"}, {"a", "b"}, {"c"}]) > 0.5
    with pytest.raises(ValueError):
        sweep_topics([["a"]] * 3)


def test_cli_flow(tmp_path, capsys):
    cfg = str(ROOT / "configs" / "topics" / "tariffs.toml")
    db = str(tmp_path / "p.db")
    gold = str(tmp_path / "gold.csv")
    assert main(["synth", "--config", cfg, "--db", db, "--n", "200", "--gold", gold, "--labels", "60"]) == 0
    assert main(["analyse", "--config", cfg, "--db", db, "--markdown"]) == 0
    assert "| all |" in capsys.readouterr().out
    assert main(["validate", "--config", cfg, "--db", db, "--gold", gold]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["agreement"]["items"] > 0 and 0 <= out["scores"]["macro_f1"] <= 1
    assert main(["compare-cleaning", "--config", cfg, "--db", db, "--gold", gold]) == 0
    cmp_ = json.loads(capsys.readouterr().out)
    assert cmp_["model_view"]["macro_f1"] > cmp_["old_cleaning"]["macro_f1"]
    assert main(["topics", "--config", cfg, "--db", db, "--k-max", "4"]) == 0
    assert main(["sample-labels", "--config", cfg, "--db", db, "--n", "20", "--out", str(tmp_path / "s.csv")]) == 0
    assert main(["runs", "--db", db]) == 0
    assert main(["analyse", "--config", str(ROOT / "configs" / "topics" / "ice-raids.toml"), "--db", db]) == 2
    assert main(["collect", "--config", cfg, "--db", db]) == 2


def test_cli_demo(tmp_path):
    assert main(["demo", "--config", str(ROOT / "configs" / "topics" / "ice-raids.toml"), "--out", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "demo_summary.json").read_text())
    v = s["validation"]
    assert v["sentiment_model_view"]["macro_f1"] > v["sentiment_old_cleaning"]["macro_f1"]
    assert v["sentiment_model_view"]["missing_predictions"] == 0


def test_transformer_sentiment_when_installed():
    if os.environ.get("PULSE_TEST_HF") != "1":
        pytest.skip("set PULSE_TEST_HF=1 to run the Hugging Face model test (it can download a model)")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from reddit_pulse.sentiment import TransformerSentiment
    try:
        m = TransformerSentiment()
    except Exception as exc:  # the model download needs network
        pytest.skip(f"model not available: {exc}")
    assert m.predict(["I love this.", "I hate this."])[0].label == "positive"


def test_window_dates(ice_cfg):
    assert ice_cfg.start == date(2025, 1, 21) and ice_cfg.end == date(2025, 3, 18)
