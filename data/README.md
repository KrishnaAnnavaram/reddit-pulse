# Data

No comment, username or label is in this repository. Reddit comments are personal data, and
their use is governed by the Reddit Data API terms. Git ignores all database and label files.

## Source

| Source | Terms | How reddit-pulse reads it |
|---|---|---|
| Reddit Data API (PRAW) | Reddit Data API terms and the Reddit User Agreement. Read them before you collect | `reddit-pulse collect --config configs/topics/<topic>.toml` (extra `reddit`) |
| Synthetic generator | MIT, no real person | `reddit-pulse synth --config ...` or `reddit-pulse demo` |

The collector needs `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` and
`PULSE_HASH_SALT` in the environment. It searches each subreddit with each query, keeps threads
and comments inside the date window, and reads the full comment tree of each thread.

## Store (`PULSE_DB`, SQLite)

| Table | Columns |
|---|---|
| `comments` | `comment_id` (key), `topic`, `thread_id`, `subreddit`, `created_utc`, `text` (mentions replaced by `u/[user]`), `author_hash` (salted HMAC, 16 hex), `score`, `is_bot` |
| `runs` | `run_id`, `topic`, `kind`, `created`, `params` (JSON) |
| `results` | `run_id`, `comment_id`, `sentiment`, `sentiment_score`, `status` (`ok` or `failed`), `aspects` (JSON) |

No username is stored. Deleted comments stay in the store and the filter drops them.

## Labels

`reddit-pulse sample-labels` writes a subreddit-stratified sample. Annotators follow
`labels/GUIDELINES.md` and write `comment_id,annotator,label`.

## Models (optional, extra `transformers`)

| Model | Use | Licence (see the model card) |
|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment | See the Hugging Face model card |
| `yangheng/deberta-v3-base-absa-v1.1` | Aspect sentiment | See the Hugging Face model card |
