# Frozen splits

`split_seed42.json` is generated **once** by the first run of
`get_or_create_split` and committed. It contains the full `index` lists for:

- `eval_5k` — the immutable 5,000-row evaluation set
- `train_subset` — the 10,000 rows used by run 001
- `pool` — every non-eval index, so a future run can draw more training data
  without ever touching the eval set

**Every run evaluates on exactly these `eval_5k` rows.** On load the split is
regenerated and compared; a disagreement raises `SplitMismatch`. Do not resolve
that by deleting this file — a moving eval set makes the leaderboard
meaningless, and the damage is silent.

The file is absent until you run `python scripts/run_local.py splits` (or any
notebook) against the real `train.csv`. Commit it as soon as it appears.
