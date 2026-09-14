"""Environment detection and canonical path resolution.

This is the ONLY module in the repo permitted to know about concrete filesystem
locations. Everything else imports ``DATA_DIR`` / ``RESULTS_DIR`` / ``IMAGE_DIR``
/ ``REPO_ROOT`` from here.

Two environments are supported:

* **Kaggle** -- detected by the existence of ``/kaggle/input``. Input datasets are
  mounted read-only under ``/kaggle/input/<slug>/``; the slug is *not* hardcoded,
  it is discovered by globbing for a directory that looks like the competition
  data (contains ``train.csv``, or a ``dataset/`` subdir containing it). Results
  go to ``/kaggle/working/results`` because ``/kaggle/input`` is read-only.
* **Local** -- ``./data`` and ``./results`` relative to the repo root.

Paths are resolved lazily at import time but can be overridden by environment
variables (``AMLC24_DATA_DIR``, ``AMLC24_RESULTS_DIR``, ``AMLC24_IMAGE_DIR``),
which is what the tests use to avoid touching the real data directory.
"""

from __future__ import annotations

import logging
import os
from glob import glob
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo root: this file lives at <root>/src/amlc24/paths.py
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def on_kaggle() -> bool:
    """True when running inside a Kaggle notebook session."""
    return KAGGLE_INPUT.exists()


def _looks_like_data_dir(p: Path) -> bool:
    """A directory qualifies as the competition data dir if it holds train.csv."""
    return (p / "train.csv").exists()


# How deep below /kaggle/input to look for train.csv. The official archive
# unpacks as "<slug>/student_resource 3/dataset/train.csv", which is depth 3.
# 5 leaves room for an extra wrapper folder without walking image directories.
MAX_DATA_SEARCH_DEPTH = 5

# Directories that never contain the CSVs but can hold tens of thousands of
# files. Skipping them keeps the search fast on a mounted image dataset.
_SKIP_DIRS = {
    "images", "image", "img", "__pycache__", ".git", ".ipynb_checkpoints",
    "node_modules", "train_images", "test_images",
}


def _search_for_data_dir(root: Path, max_depth: int = MAX_DATA_SEARCH_DEPTH) -> Path | None:
    """Breadth-first search under ``root`` for the directory holding train.csv.

    Breadth-first so the shallowest match wins: if both ``<slug>/dataset/`` and
    some nested copy contain train.csv, we want the canonical outer one.
    Directories in ``_SKIP_DIRS`` are pruned -- an attached image dataset can
    hold 100k+ files and walking it would stall the session.
    """
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if _looks_like_data_dir(current):
            return current
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except (OSError, PermissionError):
            continue
        for child in children:
            if child.name.lower() in _SKIP_DIRS or child.name.startswith("."):
                continue
            frontier.append((child, depth + 1))
    return None


def _find_kaggle_data_dir() -> Path:
    """Locate the competition CSVs among the mounted Kaggle input datasets.

    No dataset slug is hardcoded. The official archive nests the files as
    ``<slug>/student_resource 3/dataset/train.csv`` (note the space in the
    folder name), so a recursive bounded search is used rather than a fixed
    list of candidate layouts.
    """
    mounted = sorted(Path(p) for p in glob(str(KAGGLE_INPUT / "*") + "/"))

    for cand in mounted:
        found = _search_for_data_dir(cand)
        if found is not None:
            logger.info("Found competition data at %s", found)
            return found

    # Nothing matched. Say exactly what *is* mounted, so the fix is obvious
    # instead of requiring a separate debugging round-trip.
    logger.warning(
        "No mounted Kaggle dataset contains train.csv (searched %d dataset(s) "
        "to depth %d). Mounted inputs: %s. Attach the competition dataset via "
        "'+ Add Input' in the notebook sidebar, then restart the session.",
        len(mounted), MAX_DATA_SEARCH_DEPTH,
        [p.name for p in mounted] or "(none)",
    )
    if mounted:
        return mounted[0]
    return KAGGLE_INPUT


def _find_kaggle_image_dir() -> Path | None:
    """Locate a pre-uploaded resized-image dataset, if one is mounted.

    Recognised by a mounted input directory named ``*image*`` that contains at
    least one ``.jpg``. Returning ``None`` means "no image dataset mounted", and
    the caller falls back to the writable working directory.
    """
    for p in sorted(Path(x) for x in glob(str(KAGGLE_INPUT / "*") + "/")):
        name = p.name.lower()
        if "image" not in name:
            continue
        if next(p.rglob("*.jpg"), None) is not None:
            # Images may sit one level down inside the dataset.
            direct = next(p.glob("*.jpg"), None)
            if direct is not None:
                return p
            sub = next((d for d in sorted(p.iterdir()) if d.is_dir()
                        and next(d.glob("*.jpg"), None) is not None), None)
            return sub or p
    return None


def _resolve() -> tuple[Path, Path, Path]:
    """Return ``(data_dir, results_dir, image_dir)`` for the current environment."""
    if on_kaggle():
        data_dir = _find_kaggle_data_dir()
        results_dir = KAGGLE_WORKING / "results"
        mounted_images = _find_kaggle_image_dir()
        image_dir = mounted_images or (KAGGLE_WORKING / "images")
    else:
        data_dir = REPO_ROOT / "data"
        results_dir = REPO_ROOT / "results"
        image_dir = REPO_ROOT / "images"

    data_dir = Path(os.environ.get("AMLC24_DATA_DIR", data_dir))
    results_dir = Path(os.environ.get("AMLC24_RESULTS_DIR", results_dir))
    image_dir = Path(os.environ.get("AMLC24_IMAGE_DIR", image_dir))
    return data_dir, results_dir, image_dir


DATA_DIR, RESULTS_DIR, IMAGE_DIR = _resolve()

# Derived, frequently used sub-paths.
SPLITS_DIR: Path = RESULTS_DIR / "splits"
RUNS_DIR: Path = RESULTS_DIR / "runs"
EDA_DIR: Path = RESULTS_DIR / "eda"
CACHE_DIR: Path = RESULTS_DIR / "cache"
LEADERBOARD_PATH: Path = RESULTS_DIR / "leaderboard.csv"
CONFIG_DIR: Path = REPO_ROOT / "configs"


def ensure_dirs() -> None:
    """Create every writable directory the pipeline expects. Idempotent."""
    for d in (RESULTS_DIR, SPLITS_DIR, RUNS_DIR, EDA_DIR, CACHE_DIR, IMAGE_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # read-only mount (e.g. /kaggle/input image ds)
            logger.debug("Could not create %s: %s", d, exc)


def run_dir(run_id: str) -> Path:
    """Directory holding all artefacts for ``run_id``."""
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def describe() -> dict:
    """Path summary, logged at the start of every pipeline for reproducibility.

    ``train_csv_found`` is the one that matters: if it is False, nothing
    downstream can work, and the reason is almost always that the competition
    dataset was not attached to the session.
    """
    train_csv = DATA_DIR / "train.csv"
    info = {
        "environment": "kaggle" if on_kaggle() else "local",
        "repo_root": str(REPO_ROOT),
        "data_dir": str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "image_dir": str(IMAGE_DIR),
        "data_dir_exists": DATA_DIR.exists(),
        "train_csv_found": train_csv.exists(),
        "image_dir_exists": IMAGE_DIR.exists(),
    }
    if on_kaggle():
        info["mounted_inputs"] = [
            p.name for p in sorted(KAGGLE_INPUT.iterdir())
        ] if KAGGLE_INPUT.exists() else []
    if not train_csv.exists():
        info["PROBLEM"] = (
            "train.csv not found under data_dir. Attach the competition dataset "
            "('+ Add Input' in the notebook sidebar) and restart the session."
        )
    return info


__all__ = [
    "REPO_ROOT", "DATA_DIR", "RESULTS_DIR", "IMAGE_DIR", "SPLITS_DIR",
    "RUNS_DIR", "EDA_DIR", "CACHE_DIR", "LEADERBOARD_PATH", "CONFIG_DIR",
    "on_kaggle", "ensure_dirs", "run_dir", "describe",
]
