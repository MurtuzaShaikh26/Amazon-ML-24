"""Exploratory analysis. Run this before any training.

``profile_dataset(df)`` returns a dict of DataFrames so a notebook can display
each one directly, and ``save_profile`` writes them to ``results/eda/``.

Seven analyses, in order of how much they change what we build:

1. ``entity_distribution``  -- label distribution over ``entity_name``
2. ``unit_distribution``    -- units within each entity (cross-tab)
3. ``empty_rate``           -- missing/empty ``entity_value``, overall + per entity
4. ``value_stats``          -- numeric min/max/median/quantiles + outlier flags
5. ``group_distribution``   -- ``group_id`` cardinality and long-tail shape
6. ``format_audit``         -- **the one that predicts post-processing work**
7. ``image_duplication``    -- rows sharing an image

The format audit drives ``postprocess/normalize.py``: it measures how labels are
actually written (integers vs decimals, trailing ``.0``, ranges, out-of-vocab
units) instead of letting us guess. Since the metric is exact string match, a
wrong guess there costs more score than a better model recovers.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import EDA_DIR
from ..postprocess.units import allowed_units, canonicalise_unit, entity_unit_map

logger = logging.getLogger(__name__)

OUTLIER_Z = 4.0  # robust z (MAD-based) beyond which a value is flagged
_RANGE_TOKEN = re.compile(r"\d\s*(?:to|-|–|—|~|and)\s*\d", re.IGNORECASE)
_NUM_HEAD = re.compile(r"^\s*([-+]?[\d,]*\.?\d+)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _value_series(df: pd.DataFrame) -> pd.Series:
    return df.get("entity_value", pd.Series([""] * len(df), index=df.index)) \
             .fillna("").astype(str).str.strip()


def _split_value(text: str) -> tuple[str, str]:
    """``"34 gram"`` -> ``("34", "gram")``; returns ``("", "")`` when empty."""
    text = text.strip()
    if not text:
        return "", ""
    parts = text.split(maxsplit=1)
    return (parts[0], parts[1].strip()) if len(parts) > 1 else (parts[0], "")


def _numeric(text: str) -> float | None:
    m = _NUM_HEAD.match(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 1. entity_name distribution
# ---------------------------------------------------------------------------
def entity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage per ``entity_name`` -- the primary label distribution."""
    counts = df["entity_name"].value_counts()
    out = pd.DataFrame({
        "entity_name": counts.index,
        "count": counts.to_numpy(),
        "pct": (counts / len(df) * 100).to_numpy(),
    })
    out["cumulative_pct"] = out["pct"].cumsum()
    out["n_allowed_units"] = out["entity_name"].map(lambda e: len(allowed_units(e)))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. units within each entity
# ---------------------------------------------------------------------------
def unit_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per (entity, unit) counts with in-vocabulary flags."""
    values = _value_series(df)
    units = values.map(lambda v: _split_value(v)[1])

    frame = pd.DataFrame({
        "entity_name": df["entity_name"].astype(str).to_numpy(),
        "unit": units.to_numpy(),
    })
    frame = frame[frame["unit"] != ""]

    grouped = frame.groupby(["entity_name", "unit"]).size().reset_index(name="count")
    totals = grouped.groupby("entity_name")["count"].transform("sum")
    grouped["pct_within_entity"] = grouped["count"] / totals * 100
    grouped["is_allowed"] = grouped.apply(
        lambda r: r["unit"] in allowed_units(r["entity_name"]), axis=1
    )
    grouped["canonicalises_to"] = grouped["unit"].map(
        lambda u: canonicalise_unit(u) or ""
    )
    return grouped.sort_values(
        ["entity_name", "count"], ascending=[True, False]
    ).reset_index(drop=True)


def unit_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Wide entity x unit count matrix, convenient for display."""
    long = unit_distribution(df)
    if long.empty:
        return pd.DataFrame()
    return (
        long.pivot_table(index="entity_name", columns="unit", values="count",
                         aggfunc="sum", fill_value=0)
        .astype(int)
    )


# ---------------------------------------------------------------------------
# 3. empty rate
# ---------------------------------------------------------------------------
def empty_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Empty/missing ``entity_value`` rate overall and per entity.

    Empty labels are not noise: the metric scores a correct empty prediction as
    a true negative, so the empty rate is the ceiling on how much of the score
    comes from abstaining correctly.
    """
    values = _value_series(df)
    is_empty = values.eq("")
    raw_null = df.get("entity_value", pd.Series([None] * len(df))).isna()

    rows = []
    for entity, chunk in df.groupby("entity_name", sort=False):
        chunk_empty = is_empty.loc[chunk.index]
        rows.append({
            "entity_name": str(entity),
            "n": len(chunk),
            "n_empty": int(chunk_empty.sum()),
            "pct_empty": float(chunk_empty.mean() * 100),
            "n_null": int(raw_null.loc[chunk.index].sum()),
        })

    out = pd.DataFrame(rows).sort_values("pct_empty", ascending=False, ignore_index=True)
    overall = pd.DataFrame([{
        "entity_name": "OVERALL",
        "n": len(df),
        "n_empty": int(is_empty.sum()),
        "pct_empty": float(is_empty.mean() * 100),
        "n_null": int(raw_null.sum()),
    }])
    return pd.concat([out, overall], ignore_index=True)


# ---------------------------------------------------------------------------
# 4. numeric value distribution
# ---------------------------------------------------------------------------
def value_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Min/max/median/quantiles of the numeric part, per entity, with outliers.

    Outliers are flagged with a MAD-based robust z-score rather than a standard
    deviation, because these distributions are extremely heavy-tailed (a single
    ``1000000 microgram`` row would otherwise inflate sigma and hide everything).
    """
    values = _value_series(df)
    nums = values.map(_numeric)

    frame = pd.DataFrame({
        "entity_name": df["entity_name"].astype(str).to_numpy(),
        "num": nums.to_numpy(dtype=object),
    })
    frame = frame[frame["num"].notna()]
    frame["num"] = frame["num"].astype(float)

    rows = []
    for entity, chunk in frame.groupby("entity_name", sort=False):
        x = chunk["num"].to_numpy()
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        scale = mad * 1.4826 if mad > 0 else 0.0
        n_out = int((np.abs(x - median) > OUTLIER_Z * scale).sum()) if scale > 0 else 0
        rows.append({
            "entity_name": str(entity),
            "n_numeric": len(x),
            "min": float(x.min()),
            "p01": float(np.percentile(x, 1)),
            "median": median,
            "p99": float(np.percentile(x, 99)),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "n_zero_or_neg": int((x <= 0).sum()),
            "n_outliers": n_out,
            "pct_outliers": 100.0 * n_out / len(x),
        })
    return pd.DataFrame(rows).sort_values("n_numeric", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. group_id distribution
# ---------------------------------------------------------------------------
def group_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Long-tail shape of ``group_id``: cardinality, largest, concentration."""
    counts = df["group_id"].value_counts()
    n_groups = len(counts)
    cumulative = counts.cumsum() / counts.sum()

    summary = {
        "n_distinct_groups": n_groups,
        "largest_group_id": int(counts.index[0]) if n_groups else -1,
        "largest_group_size": int(counts.iloc[0]) if n_groups else 0,
        "largest_group_pct": float(counts.iloc[0] / len(df) * 100) if n_groups else 0.0,
        "median_group_size": float(counts.median()) if n_groups else 0.0,
        "smallest_group_size": int(counts.iloc[-1]) if n_groups else 0,
        "n_groups_with_1_row": int((counts == 1).sum()),
        "n_groups_covering_50pct": int((cumulative <= 0.50).sum() + 1) if n_groups else 0,
        "n_groups_covering_90pct": int((cumulative <= 0.90).sum() + 1) if n_groups else 0,
    }
    return pd.DataFrame([summary])


def group_top(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """The ``top_k`` largest groups by row count."""
    counts = df["group_id"].value_counts().head(top_k)
    return pd.DataFrame({
        "group_id": counts.index,
        "count": counts.to_numpy(),
        "pct": counts.to_numpy() / len(df) * 100,
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. format audit -- drives post-processing
# ---------------------------------------------------------------------------
def format_audit(df: pd.DataFrame) -> pd.DataFrame:
    """How ``entity_value`` strings are actually written.

    Each row is one format property with its count and share. These numbers are
    the specification for ``postprocess.normalize.format_number``: if labels
    never carry a trailing ``.0``, our predictions must not either.
    """
    values = _value_series(df)
    non_empty = values[values != ""]
    n = len(non_empty)
    if n == 0:
        return pd.DataFrame(columns=["property", "count", "pct", "note"])

    numbers = non_empty.map(lambda v: _split_value(v)[0])
    units = non_empty.map(lambda v: _split_value(v)[1])
    entities = df.loc[non_empty.index, "entity_name"].astype(str)

    has_dot = numbers.str.contains(r"\.", regex=True)
    trailing_zero = numbers.str.match(r"^-?\d+\.0+$")
    trailing_zero_decimal = numbers.str.match(r"^-?\d+\.\d*[1-9]0+$")
    has_comma = numbers.str.contains(",")
    leading_zero = numbers.str.match(r"^0\d")
    is_range = non_empty.map(lambda v: bool(_RANGE_TOKEN.search(v)))
    unparseable = numbers.map(lambda s: _numeric(s) is None)
    no_unit = units.eq("")
    multiword_unit = units.str.contains(" ")
    upper = non_empty.str.contains(r"[A-Z]")

    invalid_unit = pd.Series(
        [u not in allowed_units(e) for u, e in zip(units, entities)],
        index=non_empty.index,
    )
    recoverable = pd.Series(
        [
            bad and (canonicalise_unit(u) in allowed_units(e))
            for bad, u, e in zip(invalid_unit, units, entities)
        ],
        index=non_empty.index,
    )

    checks: list[tuple[str, pd.Series, str]] = [
        ("non_empty_values", pd.Series(True, index=non_empty.index),
         "denominator for every percentage below"),
        ("integer_no_decimal_point", ~has_dot,
         "written with no '.' at all"),
        ("decimal_point_present", has_dot,
         "written with a fractional part"),
        ("trailing_dot_zero", trailing_zero,
         "e.g. '2.0' -- if ~0, never emit '.0' in predictions"),
        ("trailing_zeros_after_decimal", trailing_zero_decimal,
         "e.g. '12.50' -- tells us whether to strip trailing zeros"),
        ("thousands_separator", has_comma,
         "e.g. '1,000' -- if ~0, never emit commas"),
        ("leading_zero", leading_zero,
         "e.g. '012' -- would break naive float round-tripping"),
        ("range_value", is_range,
         "e.g. '10 to 20 gram' -- handled by postprocess.range_rule"),
        ("unparseable_number", unparseable,
         "number head did not parse as a float"),
        ("missing_unit", no_unit,
         "value has no unit token at all"),
        ("multiword_unit", multiword_unit,
         "e.g. 'fluid ounce', 'cubic foot' -- parser must not split on space"),
        ("uppercase_present", upper,
         "labels are expected lowercase; uppercase means normalisation risk"),
        ("unit_outside_allowed_list", invalid_unit,
         "unit not in constants.py for that entity"),
        ("invalid_unit_but_canonicalisable", recoverable,
         "our alias table would rescue these"),
    ]

    rows = [
        {"property": name, "count": int(mask.sum()),
         "pct": float(mask.sum() / n * 100), "note": note}
        for name, mask, note in checks
    ]
    out = pd.DataFrame(rows)
    logger.info(
        "Format audit over %d non-empty values: %.1f%% integers, %.2f%% trailing '.0', "
        "%.2f%% ranges, %.2f%% out-of-vocab units",
        n,
        out.loc[out["property"] == "integer_no_decimal_point", "pct"].iloc[0],
        out.loc[out["property"] == "trailing_dot_zero", "pct"].iloc[0],
        out.loc[out["property"] == "range_value", "pct"].iloc[0],
        out.loc[out["property"] == "unit_outside_allowed_list", "pct"].iloc[0],
    )
    return out


def format_examples(df: pd.DataFrame, per_property: int = 5) -> pd.DataFrame:
    """Concrete example values for the interesting format properties."""
    values = _value_series(df)
    non_empty = values[values != ""]
    numbers = non_empty.map(lambda v: _split_value(v)[0])

    buckets = {
        "trailing_dot_zero": non_empty[numbers.str.match(r"^-?\d+\.0+$").fillna(False)],
        "trailing_zeros_after_decimal": non_empty[numbers.str.match(r"^-?\d+\.\d*[1-9]0+$").fillna(False)],
        "thousands_separator": non_empty[numbers.str.contains(",")],
        "range_value": non_empty[non_empty.map(lambda v: bool(_RANGE_TOKEN.search(v)))],
        "multiword_unit": non_empty[non_empty.map(lambda v: " " in _split_value(v)[1])],
        "uppercase_present": non_empty[non_empty.str.contains(r"[A-Z]")],
    }

    rows = []
    for prop, series in buckets.items():
        for value in series.head(per_property):
            rows.append({"property": prop, "example": value})
    return pd.DataFrame(rows, columns=["property", "example"])


# ---------------------------------------------------------------------------
# 7. duplicate images
# ---------------------------------------------------------------------------
def image_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """How many rows share an ``image_link``.

    One product photo often carries several entities (weight *and* dimensions),
    so the number of images to download is well below the row count -- and any
    split that ignores this leaks the same picture across train and eval.
    """
    counts = df["image_link"].value_counts()
    n_rows, n_images = len(df), len(counts)

    rows_in_shared = int(counts[counts > 1].sum())
    summary = {
        "n_rows": n_rows,
        "n_distinct_images": n_images,
        "n_images_used_once": int((counts == 1).sum()),
        "n_images_used_multiple": int((counts > 1).sum()),
        "n_rows_sharing_an_image": rows_in_shared,
        "pct_rows_sharing_an_image": float(rows_in_shared / n_rows * 100) if n_rows else 0.0,
        "max_rows_per_image": int(counts.iloc[0]) if n_images else 0,
        "mean_rows_per_image": float(n_rows / n_images) if n_images else 0.0,
        "download_saving_pct": float((1 - n_images / n_rows) * 100) if n_rows else 0.0,
    }
    return pd.DataFrame([summary])


def entities_per_image(df: pd.DataFrame) -> pd.DataFrame:
    """Histogram of how many rows each image serves."""
    counts = df["image_link"].value_counts().value_counts().sort_index()
    return pd.DataFrame({
        "rows_per_image": counts.index,
        "n_images": counts.to_numpy(),
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def profile_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every analysis. Returns ``{name: DataFrame}`` for direct display."""
    logger.info("Profiling %d rows", len(df))
    profile = {
        "entity_distribution": entity_distribution(df),
        "unit_distribution": unit_distribution(df),
        "unit_crosstab": unit_crosstab(df),
        "empty_rate": empty_rate(df),
        "value_stats": value_stats(df),
        "group_distribution": group_distribution(df),
        "group_top": group_top(df),
        "format_audit": format_audit(df),
        "format_examples": format_examples(df),
        "image_duplication": image_duplication(df),
        "entities_per_image": entities_per_image(df),
        "allowed_units": pd.DataFrame(
            [{"entity_name": e, "allowed_units": ", ".join(sorted(u)), "n_units": len(u)}
             for e, u in sorted(entity_unit_map().items())]
        ),
    }
    logger.info("Profile complete: %s", ", ".join(profile))
    return profile


def save_profile(profile: dict[str, pd.DataFrame], out_dir: str | Path | None = None) -> Path:
    """Write every profile table to ``results/eda/`` as CSV."""
    target = Path(out_dir) if out_dir else EDA_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name, table in profile.items():
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        include_index = name == "unit_crosstab"
        table.to_csv(target / f"{name}.csv", index=include_index)
    logger.info("Saved %d EDA tables to %s", len(profile), target)
    return target


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    if matplotlib.get_backend().lower() not in {"agg", "module://matplotlib_inline.backend_inline"}:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_entity_distribution(df: pd.DataFrame, save_to: str | Path | None = None):
    """Bar chart of ``entity_name`` counts -- the headline label distribution."""
    plt = _plt()
    table = entity_distribution(df)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(table))))
    ax.barh(table["entity_name"][::-1], table["count"][::-1], color="#4C78A8")
    ax.set_xlabel("rows")
    ax.set_title(f"entity_name distribution (n={len(df):,})")
    for y, (count, pct) in enumerate(zip(table["count"][::-1], table["pct"][::-1])):
        ax.text(count, y, f" {count:,} ({pct:.1f}%)", va="center", fontsize=8)
    ax.margins(x=0.18)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


def plot_unit_distribution(df: pd.DataFrame, save_to: str | Path | None = None):
    """Stacked share of units within each entity."""
    plt = _plt()
    long = unit_distribution(df)
    fig, ax = plt.subplots(figsize=(11, max(3, 0.55 * long["entity_name"].nunique())))

    entities = sorted(long["entity_name"].unique())
    cmap = plt.get_cmap("tab20")
    for i, entity in enumerate(entities):
        chunk = long[long["entity_name"] == entity].sort_values("count", ascending=False)
        left = 0.0
        for j, (_, row) in enumerate(chunk.iterrows()):
            ax.barh(i, row["pct_within_entity"], left=left, color=cmap(j % 20),
                    edgecolor="white", linewidth=0.5)
            if row["pct_within_entity"] > 7:
                ax.text(left + row["pct_within_entity"] / 2, i, row["unit"],
                        ha="center", va="center", fontsize=7)
            left += row["pct_within_entity"]

    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities)
    ax.set_xlabel("% of non-empty values within entity")
    ax.set_title("Unit mix per entity_name")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


def plot_value_histograms(df: pd.DataFrame, save_to: str | Path | None = None):
    """Log-scale histogram of numeric values per entity."""
    plt = _plt()
    values = _value_series(df)
    nums = values.map(_numeric)
    frame = pd.DataFrame({"entity_name": df["entity_name"].astype(str), "num": nums})
    frame = frame[frame["num"].notna() & (frame["num"] > 0)]

    entities = sorted(frame["entity_name"].unique())
    ncols = 3
    nrows = int(np.ceil(len(entities) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, entity in zip(axes, entities):
        x = frame.loc[frame["entity_name"] == entity, "num"].to_numpy()
        ax.hist(np.log10(x), bins=40, color="#4C78A8")
        ax.set_title(f"{entity} (n={len(x):,})", fontsize=9)
        ax.set_xlabel("log10(value)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(entities):]:
        ax.set_visible(False)

    fig.suptitle("Numeric value distribution per entity (log10 scale)", y=1.01)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


def plot_group_tail(df: pd.DataFrame, save_to: str | Path | None = None):
    """Group size rank curve, showing the long tail."""
    plt = _plt()
    counts = df["group_id"].value_counts().to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

    ax1.plot(np.arange(1, len(counts) + 1), counts, color="#E45756")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("group rank"); ax1.set_ylabel("rows")
    ax1.set_title("group_id size by rank (log-log)")

    ax2.plot(np.arange(1, len(counts) + 1), np.cumsum(counts) / counts.sum() * 100,
             color="#54A24B")
    ax2.set_xlabel("groups included"); ax2.set_ylabel("% of rows covered")
    ax2.set_title("Cumulative coverage")
    ax2.axhline(90, ls="--", lw=0.8, color="grey")

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


def plot_empty_rate(df: pd.DataFrame, save_to: str | Path | None = None):
    """Empty-value rate per entity."""
    plt = _plt()
    table = empty_rate(df)
    table = table[table["entity_name"] != "OVERALL"]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(table))))
    ax.barh(table["entity_name"][::-1], table["pct_empty"][::-1], color="#F58518")
    ax.set_xlabel("% empty entity_value")
    ax.set_title("Empty-value rate per entity")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


def save_all_plots(df: pd.DataFrame, out_dir: str | Path | None = None) -> list[Path]:
    """Render and save every chart; returns the written paths."""
    target = Path(out_dir) if out_dir else EDA_DIR
    target.mkdir(parents=True, exist_ok=True)
    plotters = {
        "entity_distribution.png": plot_entity_distribution,
        "unit_distribution.png": plot_unit_distribution,
        "value_histograms.png": plot_value_histograms,
        "group_tail.png": plot_group_tail,
        "empty_rate.png": plot_empty_rate,
    }
    written = []
    for filename, fn in plotters.items():
        try:
            fn(df, save_to=target / filename)
            written.append(target / filename)
        except (ValueError, ImportError, IndexError) as exc:
            logger.warning("Could not render %s: %s", filename, exc)
    logger.info("Saved %d EDA charts to %s", len(written), target)
    return written


__all__ = [
    "profile_dataset", "save_profile", "save_all_plots",
    "entity_distribution", "unit_distribution", "unit_crosstab", "empty_rate",
    "value_stats", "group_distribution", "group_top", "format_audit",
    "format_examples", "image_duplication", "entities_per_image",
    "plot_entity_distribution", "plot_unit_distribution", "plot_value_histograms",
    "plot_group_tail", "plot_empty_rate",
]
