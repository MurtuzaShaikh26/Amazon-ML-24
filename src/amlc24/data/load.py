"""Load the competition CSVs with schema validation and a parquet cache.

CSV parsing of the full train file is slow enough to be annoying on every
notebook restart, so the first parse is cached as parquet under
``results/cache/``. The cache is invalidated by source mtime and size, so
re-uploading the dataset transparently rebuilds it.

The ``entity_value`` assertions are deliberately hard failures: silently
treating the blind test set as if it were labelled would produce a meaningless
score, and that is exactly the mistake this repo is structured to prevent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..paths import CACHE_DIR, DATA_DIR

logger = logging.getLogger(__name__)

BASE_COLUMNS = ["index", "image_link", "group_id", "entity_name"]
TRAIN_COLUMNS = BASE_COLUMNS + ["entity_value"]
TEST_COLUMNS = BASE_COLUMNS

# Read everything as string first so the `index` column never becomes a float
# via NaN promotion and `entity_value` keeps its exact textual form.
_DTYPES = {
    "index": "int64",
    "image_link": "string",
    "group_id": "int64",
    "entity_name": "string",
    "entity_value": "string",
}


class DataError(RuntimeError):
    """Raised when a data file is missing or has an unexpected schema."""


def _find_csv(name: str) -> Path:
    """Locate ``name`` in the data dir, tolerating a nested ``dataset/`` layout."""
    candidates = [
        DATA_DIR / name,
        DATA_DIR / "dataset" / name,
        *sorted(DATA_DIR.glob(f"*/{name}")),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise DataError(
        f"{name} not found. Looked in: {[str(c) for c in candidates[:3]]}. "
        f"DATA_DIR is {DATA_DIR}; set AMLC24_DATA_DIR or attach the dataset."
    )


def _cache_path(source: Path) -> Path:
    stat = source.stat()
    stamp = f"{int(stat.st_mtime)}_{stat.st_size}"
    return CACHE_DIR / f"{source.stem}_{stamp}.parquet"


def _read_csv_cached(source: Path, use_cache: bool = True) -> pd.DataFrame:
    """Read a CSV, round-tripping through a parquet cache when possible."""
    cache = _cache_path(source)
    if use_cache and cache.exists():
        try:
            df = pd.read_parquet(cache)
            logger.info("Loaded %s rows from parquet cache %s", len(df), cache.name)
            return df
        except (OSError, ValueError) as exc:
            logger.warning("Cache %s unreadable (%s); re-parsing CSV", cache.name, exc)

    logger.info("Parsing %s", source)
    df = pd.read_csv(source)

    for col, dtype in _DTYPES.items():
        if col in df.columns:
            if dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df[col] = df[col].astype("string")

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
            logger.info("Cached %s -> %s", source.name, cache.name)
            for stale in CACHE_DIR.glob(f"{source.stem}_*.parquet"):
                if stale != cache:
                    stale.unlink(missing_ok=True)
        except (OSError, ImportError, ValueError) as exc:
            logger.warning("Could not write parquet cache (%s); continuing", exc)

    return df


def _validate(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataError(f"{name} is missing column(s) {missing}; got {list(df.columns)}")
    if df["index"].isna().any():
        raise DataError(f"{name} has null values in the `index` column")
    if df["index"].duplicated().any():
        n = int(df["index"].duplicated().sum())
        raise DataError(f"{name} has {n} duplicated `index` values; they must be unique")


def load_train(use_cache: bool = True) -> pd.DataFrame:
    """Load ``train.csv``. Asserts ``entity_value`` is present."""
    source = _find_csv("train.csv")
    df = _read_csv_cached(source, use_cache)
    _validate(df, TRAIN_COLUMNS, "train.csv")

    if "entity_value" not in df.columns:
        raise DataError("train.csv must contain `entity_value`; it is the target.")

    df["entity_value"] = df["entity_value"].fillna("").astype("string")
    n_empty = int((df["entity_value"].str.strip() == "").sum())
    logger.info(
        "train.csv: %d rows, %d entities, %d groups, %d empty entity_value (%.2f%%)",
        len(df), df["entity_name"].nunique(), df["group_id"].nunique(),
        n_empty, 100 * n_empty / max(len(df), 1),
    )
    return df.reset_index(drop=True)


def load_test(use_cache: bool = True) -> pd.DataFrame:
    """Load ``test.csv``. Asserts ``entity_value`` is ABSENT.

    test.csv is the blind leaderboard set. If it ever arrives with labels, the
    file is not what we think it is and every downstream score would be suspect,
    so this raises rather than warns.
    """
    source = _find_csv("test.csv")
    df = _read_csv_cached(source, use_cache)
    _validate(df, TEST_COLUMNS, "test.csv")

    if "entity_value" in df.columns:
        raise DataError(
            "test.csv unexpectedly contains `entity_value`. It is the blind test "
            "set and must not be used for evaluation -- all evaluation comes from "
            "the frozen 5k split carved out of train.csv."
        )

    logger.info("test.csv: %d rows, %d entities", len(df), df["entity_name"].nunique())
    return df.reset_index(drop=True)


def load_split_frames(split: dict, train_df: pd.DataFrame | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise ``(eval_df, train_subset_df)`` from a split dict of indices."""
    df = load_train() if train_df is None else train_df
    by_index = df.set_index("index", drop=False)

    eval_ids = split["eval_5k"]
    train_ids = split.get("train_subset") or split.get("pool", [])

    eval_df = by_index.loc[by_index.index.intersection(eval_ids)].reset_index(drop=True)
    train_subset = by_index.loc[by_index.index.intersection(train_ids)].reset_index(drop=True)

    if len(eval_df) != len(eval_ids):
        logger.warning(
            "Split references %d eval indices but only %d were found in train.csv",
            len(eval_ids), len(eval_df),
        )
    return eval_df, train_subset


__all__ = ["load_train", "load_test", "load_split_frames", "DataError",
           "TRAIN_COLUMNS", "TEST_COLUMNS"]
