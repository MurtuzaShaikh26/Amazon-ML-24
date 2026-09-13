"""EDA pipeline: profile ``train.csv`` and write every table and chart.

CPU-only and fast. Run this before any training -- the format audit it produces
is the specification for post-processing, and post-processing is worth more here
than a bigger model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.eda import profile_dataset, save_all_plots, save_profile
from ..data.load import load_train
from ..logging_utils import setup_logging
from ..paths import EDA_DIR, describe, ensure_dirs

logger = logging.getLogger(__name__)


def run_eda(
    df: pd.DataFrame | None = None,
    save: bool = True,
    plots: bool = True,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Profile the training set.

    Returns ``{"profile": {name: DataFrame}, "figures": [...], "out_dir": Path}``
    so a notebook can display each table without recomputing anything.
    """
    setup_logging()
    ensure_dirs()
    logger.info("Environment: %s", describe())

    frame = load_train() if df is None else df
    target = Path(out_dir) if out_dir else EDA_DIR

    profile = profile_dataset(frame)

    if save:
        save_profile(profile, target)

    figures: list[Path] = []
    if plots:
        figures = save_all_plots(frame, target)

    _log_headlines(profile)

    return {"profile": profile, "figures": figures, "out_dir": target, "n_rows": len(frame)}


def _log_headlines(profile: dict[str, pd.DataFrame]) -> None:
    """Log the handful of numbers that actually change what we build next."""
    entity = profile.get("entity_distribution")
    if entity is not None and not entity.empty:
        top = entity.iloc[0]
        logger.info("Most common entity: %s (%.1f%% of rows)", top["entity_name"], top["pct"])

    empty = profile.get("empty_rate")
    if empty is not None and not empty.empty:
        overall = empty[empty["entity_name"] == "OVERALL"]
        if not overall.empty:
            logger.info("Overall empty entity_value rate: %.2f%%",
                        overall.iloc[0]["pct_empty"])

    audit = profile.get("format_audit")
    if audit is not None and not audit.empty:
        lookup = audit.set_index("property")["pct"]
        logger.info(
            "FORMAT AUDIT -> integers=%.1f%%, trailing '.0'=%.2f%%, ranges=%.2f%%, "
            "out-of-vocab units=%.2f%% (of which %.2f%% are canonicalisable). "
            "These numbers define postprocess.normalize.format_number.",
            lookup.get("integer_no_decimal_point", 0.0),
            lookup.get("trailing_dot_zero", 0.0),
            lookup.get("range_value", 0.0),
            lookup.get("unit_outside_allowed_list", 0.0),
            lookup.get("invalid_unit_but_canonicalisable", 0.0),
        )

    images = profile.get("image_duplication")
    if images is not None and not images.empty:
        row = images.iloc[0]
        logger.info(
            "Images: %d distinct for %d rows -- downloading unique URLs saves %.1f%%",
            row["n_distinct_images"], row["n_rows"], row["download_saving_pct"],
        )


__all__ = ["run_eda"]
