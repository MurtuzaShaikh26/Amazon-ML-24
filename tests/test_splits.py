"""Split determinism, disjointness, stratification, and immutability."""

from __future__ import annotations

import json

import pytest

from amlc24.data.splits import (
    SplitMismatch,
    build_stratum_key,
    entity_proportion_table,
    load_split,
    make_splits,
    save_split,
    verify_split,
)

EVAL_N, TRAIN_N = 100, 200


@pytest.fixture
def split(synthetic_train_df):
    return make_splits(synthetic_train_df, seed=42, eval_size=EVAL_N, train_size=TRAIN_N)


# --- determinism -----------------------------------------------------------
def test_same_seed_gives_identical_splits(synthetic_train_df):
    a = make_splits(synthetic_train_df, seed=42, eval_size=EVAL_N, train_size=TRAIN_N)
    b = make_splits(synthetic_train_df, seed=42, eval_size=EVAL_N, train_size=TRAIN_N)
    assert a["eval_5k"] == b["eval_5k"]
    assert a["train_subset"] == b["train_subset"]


def test_row_order_does_not_affect_the_split(synthetic_train_df):
    """The split sorts by `index` first, so a shuffled input must not matter."""
    shuffled = synthetic_train_df.sample(frac=1.0, random_state=7)
    a = make_splits(synthetic_train_df, seed=42, eval_size=EVAL_N, train_size=TRAIN_N)
    b = make_splits(shuffled, seed=42, eval_size=EVAL_N, train_size=TRAIN_N)
    assert a["eval_5k"] == b["eval_5k"]


def test_different_seed_gives_a_different_split(synthetic_train_df):
    a = make_splits(synthetic_train_df, seed=42, eval_size=EVAL_N, train_size=TRAIN_N)
    b = make_splits(synthetic_train_df, seed=1234, eval_size=EVAL_N, train_size=TRAIN_N)
    assert a["eval_5k"] != b["eval_5k"]


# --- disjointness and sizes ------------------------------------------------
def test_eval_and_train_are_disjoint(split):
    assert not set(split["eval_5k"]) & set(split["train_subset"])


def test_train_subset_is_drawn_from_the_non_eval_pool(split):
    assert set(split["train_subset"]) <= set(split["pool"])
    assert not set(split["pool"]) & set(split["eval_5k"])


def test_sizes_are_exact(split):
    assert len(split["eval_5k"]) == EVAL_N
    assert len(split["train_subset"]) == TRAIN_N


def test_no_duplicate_indices_within_a_split(split):
    for key in ("eval_5k", "train_subset", "pool"):
        assert len(split[key]) == len(set(split[key])), key


def test_indices_are_sorted_for_stable_diffs(split):
    assert split["eval_5k"] == sorted(split["eval_5k"])


def test_requesting_more_rows_than_available_raises(synthetic_train_df):
    with pytest.raises(ValueError, match="too few"):
        make_splits(synthetic_train_df, eval_size=5000, train_size=10000)


# --- stratification --------------------------------------------------------
def test_entity_proportions_match_within_one_percentage_point(split, synthetic_train_df):
    by_index = synthetic_train_df.set_index("index")
    eval_df = by_index.loc[split["eval_5k"]]
    train_df = by_index.loc[split["train_subset"]]

    table = entity_proportion_table(synthetic_train_df, eval_df, train_df)
    assert (table["eval_delta_pp"].abs() <= 1.0).all()
    assert (table["train_delta_pp"].abs() <= 1.0).all()


def test_empty_value_rows_are_represented_in_eval(split, synthetic_train_df):
    """Empty labels drive the FN/TN terms, so they must not be sampled away."""
    by_index = synthetic_train_df.set_index("index")
    eval_df = by_index.loc[split["eval_5k"]]
    assert (eval_df["entity_value"].fillna("").str.strip() == "").sum() > 0


def test_stratum_key_separates_empty_from_non_empty(synthetic_train_df):
    key = build_stratum_key(synthetic_train_df, use_magnitude=False)
    is_empty = synthetic_train_df["entity_value"].fillna("").str.strip().eq("")
    assert set(key[is_empty]).isdisjoint(set(key[~is_empty]))


def test_small_strata_are_merged_to_meet_the_minimum(synthetic_train_df):
    """Every stratum is either large enough, or is a whole (entity, emptiness)
    class that cannot be merged further without distorting the split.

    The fixture has only ~35 empty-valued rows spread across 8 entities, so the
    ``*|empty`` strata legitimately sit below the minimum. What must not survive
    is an undersized stratum that still carries a magnitude bin -- that one had
    somewhere to merge to and should have been absorbed.
    """
    key = build_stratum_key(synthetic_train_df, use_magnitude=True, min_stratum=10)
    counts = key.value_counts()

    undersized = counts[counts < 10]
    for stratum in undersized.index:
        assert len(str(stratum).split("|")) == 2, (
            f"{stratum!r} kept its magnitude bin despite being undersized; "
            "it should have been absorbed into a larger sibling"
        )

    # And each undersized stratum really is the whole of its class.
    is_empty = synthetic_train_df["entity_value"].fillna("").str.strip().eq("")
    for stratum, size in undersized.items():
        entity, emptiness = str(stratum).split("|")
        mask = (synthetic_train_df["entity_name"] == entity) & (
            is_empty if emptiness == "empty" else ~is_empty
        )
        assert size == int(mask.sum()), f"{stratum} is not the full class"


def test_magnitude_binned_strata_meet_the_minimum(synthetic_train_df):
    """Any stratum that kept all three components must be large enough."""
    key = build_stratum_key(synthetic_train_df, use_magnitude=True, min_stratum=10)
    counts = key.value_counts()
    three_part = [s for s in counts.index if len(str(s).split("|")) == 3]
    assert three_part, "expected at least one fully-binned stratum"
    for stratum in three_part:
        assert counts[stratum] >= 10, f"{stratum} kept a magnitude bin while thin"


def test_rare_buckets_merge_within_their_entity_not_across_entities(synthetic_train_df):
    """Rare rows must stay inside their entity, or entity proportions break."""
    key = build_stratum_key(synthetic_train_df, use_magnitude=True, min_stratum=10)
    entities = set(synthetic_train_df["entity_name"])
    for stratum in key.unique():
        assert stratum.split("|")[0] in entities, (
            f"stratum {stratum!r} does not belong to a single entity"
        )


def test_split_records_its_stratification_metadata(split):
    meta = split["stratify"]
    assert "entity_name" in meta["key"] and "is_empty" in meta["key"]
    assert meta["n_strata"] > 1


# --- immutability ----------------------------------------------------------
def test_save_then_load_round_trips(split, tmp_path):
    path = save_split(split, tmp_path / "split.json")
    loaded = load_split(path)
    assert loaded["eval_5k"] == split["eval_5k"]
    assert loaded["train_subset"] == split["train_subset"]


def test_saved_json_is_plain_and_diffable(split, tmp_path):
    path = save_split(split, tmp_path / "split.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload["eval_5k"], list)
    assert all(isinstance(i, int) for i in payload["eval_5k"][:10])


def test_save_refuses_to_overwrite_a_frozen_split(split, tmp_path):
    path = save_split(split, tmp_path / "split.json")
    with pytest.raises(SplitMismatch, match="frozen"):
        save_split(split, path)


def test_verify_passes_for_an_identical_regeneration(split, synthetic_train_df):
    fresh = make_splits(synthetic_train_df, seed=42, eval_size=EVAL_N, train_size=TRAIN_N)
    verify_split(split, fresh)  # must not raise


def test_verify_raises_when_the_eval_set_drifts(split, synthetic_train_df):
    drifted = make_splits(synthetic_train_df, seed=99, eval_size=EVAL_N, train_size=TRAIN_N)
    with pytest.raises(SplitMismatch, match="frozen"):
        verify_split(split, drifted)


def test_verify_raises_on_a_single_changed_index(split):
    tampered = {**split, "eval_5k": split["eval_5k"][:-1] + [999999]}
    with pytest.raises(SplitMismatch):
        verify_split(split, tampered)
