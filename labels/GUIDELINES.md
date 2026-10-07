# Annotation guidelines

Use these rules when you label the sample from `reddit-pulse sample-labels`.

1. Read the full comment. Label the opinion of the writer about the topic of the thread.
2. Use `positive` if the writer supports or praises the policy, the action or the actor.
3. Use `negative` if the writer opposes, criticises or mocks the policy, the action or the actor.
4. Use `neutral` for questions, facts, links, jokes without a side, and comments about other subjects.
5. Sarcasm: label the meaning, not the words. "Oh great, another raid" is `negative`.
6. If a comment has two opinions, label the opinion of the last sentence.
7. Do not look at the subreddit name, the score or the model output before you label.
8. Two annotators label each comment without talking to each other. Then they discuss only the
   comments with different labels. Keep the first labels for the kappa value.

Write one row for each comment and annotator: `comment_id,annotator,label`.
Keep the label files outside git (git ignores everything in `labels/` except this file).
