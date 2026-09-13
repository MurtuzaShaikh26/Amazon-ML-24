"""Stratified, deterministic, and **immutable** train/eval splits.

.. warning::

   **The 5,000-row evaluation split is frozen.** It is generated once, written
   to ``results/splits/split_seed42.json``, and committed to the repository.
   Every run -- this one and every future one -- evaluates on exactly those rows.
   If the stored file exists, it is loaded and *verified*, never regenerated. A
   regenerated split that disagrees with the stored one raises ``SplitMismatch``
   rather than overwriting it.

   This is what makes run-to-run numbers on the leaderboard comparable. A moving
   eval set turns the leaderboard into noise, and the damage is invisible until
   someone tries to trust it. Do not "fix" a mismatch by deleting the JSON.

``test.csv`` carries no ``entity_value`` and therefore cannot be evaluated
against; every split here is carved out of ``train.csv``.

Stratification key
------------------
``entity_name`` x ``is_empty(entity_value)``, crossed with a coarse
log-magnitude bin of the numeric value where strata stay large enough.

Undersized strata are merged by dropping key components from the right, never
across an ``entity_name`` or emptiness boundary -- see ``build_stratum_key``.
That keeps entity proportions matching to within 1 percentage point, which is
asserted, because per-entity F1 is only comparable across runs when every split
carries the same entity mix.

On the real ``train.csv`` the emptiness axis is degenerate: the EDA found
**zero** empty labels in all 263,859 rows. The axis is kept because it costs
nothing and guards against a future file that does contain them.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import REPO_ROOT, SPLITS_DIR

logger = logging.getLogger(__name__)


def _fmt3(value: float) -> str:
    """pandas >= 2 requires float_format to be a callable."""
    return f"{value:.3f}"

MIN_STRATUM = 10
PROPORTION_TOLERANCE_PP = 1.0  # percentage points
DEFAULT_SEED = 42
DEFAULT_EVAL_SIZE = 5000
DEFAULT_TRAIN_SIZE = 10000


class SplitMismatch(RuntimeError):
    """Raised when a regenerated split disagrees with the committed one."""


def split_path(seed: int = DEFAULT_SEED) -> Path:
    """Canonical location of the frozen split file for ``seed``."""
    return SPLITS_DIR / f"split_seed{seed}.json"


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def _numeric_part(value: Any) -> float | None:
    """Leading number of an ``"<x> <unit>"`` label, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    head = text.split(maxsplit=1)[0].replace(",", "")
    try:
        return float(head)
    except ValueError:
        return None


def _magnitude_bin(value: Any) -> str:
    """Coarse log10 bucket of the numeric part; stable across runs."""
    num = _numeric_part(value)
    if num is None:
        return "na"
    if num <= 0:
        return "nonpos"
    return f"e{int(math.floor(math.log10(num)))}"


def build_stratum_key(
    df: pd.DataFrame,
    use_magnitude: bool = True,
    min_stratum: int = MIN_STRATUM,
) -> pd.Series:
    """Build the stratification key and merge undersized strata.

    The key has up to three components, ``entity | emptiness | magnitude``, and
    merging always removes components from the right:

    1. An ``(entity, emptiness, magnitude)`` cell below ``min_stratum`` drops
       its magnitude bin and falls back to ``(entity, emptiness)``.
    2. Anything still undersized is absorbed into the largest stratum sharing
       its ``(entity, emptiness)`` prefix.

    Merging never crosses an ``entity`` or an ``emptiness`` boundary, which is
    what preserves the two invariants that matter downstream: entity_name
    proportions are asserted to match across splits, and empty-valued rows are a
    distinct scored population (they drive the FN/TN terms of the metric), so
    folding them in with real values would misrepresent both splits.

    The cost is that an entity with fewer than ``min_stratum`` rows in one
    emptiness class keeps a thin stratum. That is logged, and is the right
    trade: a slightly thin stratum is harmless, whereas a split whose empty rate
    or entity mix drifts silently invalidates every cross-run comparison.
    """
    entity = df["entity_name"].astype(str)
    is_empty = df["entity_value"].fillna("").astype(str).str.strip().eq("")
    base = pd.Series(entity + "|" + np.where(is_empty, "empty", "value"), index=df.index)

    if not use_magnitude:
        key = base.copy()
    else:
        mag = df["entity_value"].map(_magnitude_bin)
        fine = base + "|" + mag.astype(str)
        fine_counts = fine.value_counts()
        # Step 1: undersized fine cells fall back to (entity, emptiness).
        key = fine.where(fine.map(fine_counts) >= min_stratum, base)
        n_dropped = int((key != fine).sum())
        if n_dropped:
            logger.info(
                "Dropped the magnitude bin for %d row(s) in undersized cells", n_dropped
            )

    # Step 2: absorb anything still thin into the largest stratum with the same
    # (entity, emptiness) prefix.
    counts = key.value_counts()
    still_small = [s for s in counts.index if counts[s] < min_stratum]
    if still_small:
        absorbed = 0
        for stratum in still_small:
            stratum_prefix = "|".join(str(stratum).split("|")[:2])
            siblings = counts[[
                s for s in counts.index
                if s != stratum
                and "|".join(str(s).split("|")[:2]) == stratum_prefix
                and counts[s] >= min_stratum
            ]]
            if siblings.empty:
                continue
            key = key.where(key != stratum, siblings.idxmax())
            absorbed += 1
        logger.info(
            "Absorbed %d of %d undersized stratum/strata into a larger stratum "
            "with the same entity and emptiness", absorbed, len(still_small),
        )

    final = key.value_counts()
    undersized = final[final < min_stratum]
    if len(undersized):
        logger.warning(
            "%d stratum/strata remain below %d members; their (entity, emptiness) "
            "class is simply this small, and merging further would distort the "
            "split: %s",
            len(undersized), min_stratum, dict(undersized),
        )

    logger.info("Stratification: %d strata, smallest=%d, largest=%d",
                key.nunique(), int(key.value_counts().min()), int(key.value_counts().max()))
    return key


def _stratified_sample(
    df: pd.DataFrame,
    key: pd.Series,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` row labels, allocating per stratum by largest remainder.

    Largest-remainder allocation (rather than rounding each stratum
    independently) guarantees the total is exactly ``n`` while keeping every
    stratum's share within one row of proportional.
    """
    if n > len(df):
        raise ValueError(f"Cannot sample {n} rows from a frame of {len(df)}")

    counts = key.value_counts().sort_index()
    exact = counts / counts.sum() * n
    floor = np.floor(exact).astype(int)
    remainder = n - int(floor.sum())

    if remainder > 0:
        # Deterministic tie-break: fractional part desc, then stratum name.
        order = sorted(
            range(len(counts)),
            key=lambda i: (-(exact.iloc[i] - floor.iloc[i]), str(counts.index[i])),
        )
        for i in order[:remainder]:
            floor.iloc[i] += 1

    picked: list[np.ndarray] = []
    for stratum, take in floor.items():
        if take <= 0:
            continue
        members = df.index[key == stratum].to_numpy()
        take = min(int(take), len(members))
        picked.append(rng.choice(members, size=take, replace=False))

    out = np.concatenate(picked) if picked else np.array([], dtype=df.index.dtype)

    # Largest-remainder can still fall short when a stratum was capped by size.
    if len(out) < n:
        remaining = np.setdiff1d(df.index.to_numpy(), out)
        extra = rng.choice(remaining, size=min(n - len(out), len(remaining)), replace=False)
        out = np.concatenate([out, extra])

    return np.sort(out)


# ---------------------------------------------------------------------------
# Proportion checking
# ---------------------------------------------------------------------------
def entity_proportion_table(
    full: pd.DataFrame, eval_df: pd.DataFrame, train_df: pd.DataFrame
) -> pd.DataFrame:
    """Side-by-side ``entity_name`` share (%) for full / eval / train subset."""
    def share(df: pd.DataFrame, label: str) -> pd.Series:
        return (df["entity_name"].value_counts(normalize=True) * 100).rename(label)

    table = pd.concat(
        [share(full, "full_pct"), share(eval_df, "eval_pct"), share(train_df, "train_pct")],
        axis=1,
    ).fillna(0.0)
    table["eval_delta_pp"] = table["eval_pct"] - table["full_pct"]
    table["train_delta_pp"] = table["train_pct"] - table["full_pct"]
    table["n_full"] = full["entity_name"].value_counts()
    return table.sort_values("full_pct", ascending=False).reset_index(names="entity_name")


def _assert_proportions(table: pd.DataFrame, tolerance_pp: float) -> None:
    bad = table[
        (table["eval_delta_pp"].abs() > tolerance_pp)
        | (table["train_delta_pp"].abs() > tolerance_pp)
    ]
    if not bad.empty:
        raise AssertionError(
            "entity_name proportions drift beyond "
            f"{tolerance_pp} pp:\n{bad.to_string(index=False)}"
        )
    logger.info("entity_name proportions match within %.1f pp across all splits", tolerance_pp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_splits(
    train_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    eval_size: int = DEFAULT_EVAL_SIZE,
    train_size: int = DEFAULT_TRAIN_SIZE,
    use_magnitude: bool = True,
    tolerance_pp: float = PROPORTION_TOLERANCE_PP,
) -> dict[str, Any]:
    """Generate the frozen eval split and this run's training subset.

    Returns a dict with ``eval_5k``, ``train_subset``, ``pool`` (every non-eval
    index), the stratification metadata, and the comparison table. Pure: writing
    happens in :func:`save_split`.
    """
    df = train_df.sort_values("index", ascending=True).reset_index(drop=True)
    if len(df) < eval_size + train_size:
        raise ValueError(
            f"train.csv has {len(df)} rows, too few for eval_size={eval_size} "
            f"+ train_size={train_size}"
        )

    key = build_stratum_key(df, use_magnitude=use_magnitude)
    rng = np.random.default_rng(seed)

    eval_pos = _stratified_sample(df, key, eval_size, rng)
    eval_df = df.loc[eval_pos]

    pool_df = df.drop(index=eval_pos)
    pool_key = key.loc[pool_df.index]
    train_pos = _stratified_sample(pool_df, pool_key, train_size, rng)
    train_subset = df.loc[train_pos]

    eval_ids = sorted(int(i) for i in eval_df["index"])
    train_ids = sorted(int(i) for i in train_subset["index"])
    pool_ids = sorted(int(i) for i in pool_df["index"])

    overlap = set(eval_ids) & set(train_ids)
    if overlap:
        raise AssertionError(f"eval and train subsets overlap on {len(overlap)} indices")
    if set(train_ids) - set(pool_ids):
        raise AssertionError("train subset contains indices outside the non-eval pool")
    if len(eval_ids) != eval_size or len(train_ids) != train_size:
        raise AssertionError(
            f"Expected {eval_size}/{train_size}, got {len(eval_ids)}/{len(train_ids)}"
        )

    table = entity_proportion_table(df, eval_df, train_subset)
    _assert_proportions(table, tolerance_pp)
    logger.info("Split comparison table:\n%s", table.to_string(index=False, float_format=_fmt3))

    return {
        "seed": seed,
        "eval_size": eval_size,
        "train_size": train_size,
        "n_source_rows": len(df),
        "stratify": {
            "key": "entity_name x is_empty(entity_value)"
                   + (" x log10_magnitude" if use_magnitude else ""),
            "use_magnitude": use_magnitude,
            "min_stratum": MIN_STRATUM,
            "n_strata": int(key.nunique()),
        },
        "eval_5k": eval_ids,
        "train_subset": train_ids,
        "pool": pool_ids,
        "proportion_table": table,
    }


def _serialisable(split: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in split.items() if k != "proportion_table"}
    table = split.get("proportion_table")
    if isinstance(table, pd.DataFrame):
        out["proportion_table"] = table.round(4).to_dict(orient="records")
    return out


def save_split(split: dict[str, Any], path: str | Path | None = None,
               overwrite: bool = False) -> Path:
    """Write the split to JSON. Refuses to clobber an existing file.

    ``overwrite`` exists only for the deliberate act of regenerating a split
    under a *new* seed; it is never set by the pipeline.
    """
    target = Path(path) if path else split_path(split.get("seed", DEFAULT_SEED))
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        raise SplitMismatch(
            f"{target} already exists. The eval split is frozen and must not be "
            "regenerated. Load and verify it instead."
        )

    with target.open("w", encoding="utf-8") as fh:
        json.dump(_serialisable(split), fh, indent=2)
    logger.info("Wrote frozen split to %s (%d eval / %d train)",
                target, len(split["eval_5k"]), len(split.get("train_subset", [])))
    return target


def load_split(path: str | Path | None = None, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Load a committed split file."""
    target = Path(path) if path else split_path(seed)
    if not target.is_absolute() and not target.exists():
        candidate = REPO_ROOT / target
        if candidate.exists():
            target = candidate
    if not target.exists():
        raise FileNotFoundError(f"No split file at {target}")

    with target.open("r", encoding="utf-8") as fh:
        split = json.load(fh)
    logger.info("Loaded frozen split %s: %d eval, %d train",
                target, len(split["eval_5k"]), len(split.get("train_subset", [])))
    return split


def verify_split(stored: dict[str, Any], regenerated: dict[str, Any]) -> None:
    """Raise ``SplitMismatch`` unless the eval sets are byte-for-byte identical."""
    stored_eval = list(stored["eval_5k"])
    fresh_eval = list(regenerated["eval_5k"])

    if stored_eval != fresh_eval:
        only_stored = sorted(set(stored_eval) - set(fresh_eval))[:5]
        only_fresh = sorted(set(fresh_eval) - set(stored_eval))[:5]
        raise SplitMismatch(
            "Regenerated eval split does not match the committed one.\n"
            f"  stored n={len(stored_eval)}, regenerated n={len(fresh_eval)}\n"
            f"  example indices only in stored:      {only_stored}\n"
            f"  example indices only in regenerated: {only_fresh}\n"
            "The 5k eval set is frozen. This usually means train.csv changed or "
            "the stratification logic was edited. Do NOT delete the split file: "
            "investigate, because every committed score depends on it."
        )
    logger.info("Verified: regenerated eval split matches the committed one exactly.")


def get_or_create_split(
    train_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    eval_size: int = DEFAULT_EVAL_SIZE,
    train_size: int = DEFAULT_TRAIN_SIZE,
    path: str | Path | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """The entry point every pipeline uses.

    If the frozen split exists it is loaded and (by default) verified by
    regenerating and comparing. Otherwise it is created and written once.
    """
    target = Path(path) if path else split_path(seed)
    if not target.is_absolute() and not target.exists() and (REPO_ROOT / target).exists():
        target = REPO_ROOT / target

    if target.exists():
        stored = load_split(target)
        if verify:
            try:
                fresh = make_splits(train_df, seed=seed,
                                    eval_size=stored.get("eval_size", eval_size),
                                    train_size=stored.get("train_size", train_size))
                verify_split(stored, fresh)
                stored["proportion_table"] = fresh["proportion_table"]
            except SplitMismatch:
                raise
            except (ValueError, AssertionError) as exc:
                logger.warning("Could not re-verify split (%s); using stored indices", exc)
        return stored

    logger.info("No frozen split at %s; creating it now (this happens once).", target)
    split = make_splits(train_df, seed=seed, eval_size=eval_size, train_size=train_size)
    save_split(split, target)
    return split


__all__ = [
    "make_splits", "save_split", "load_split", "verify_split", "get_or_create_split",
    "build_stratum_key", "entity_proportion_table", "split_path", "SplitMismatch",
    "DEFAULT_SEED", "DEFAULT_EVAL_SIZE", "DEFAULT_TRAIN_SIZE", "MIN_STRATUM",
]
