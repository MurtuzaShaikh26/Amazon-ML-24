"""Per-run artefact directory and the cross-run leaderboard.

Every run writes ``results/runs/{run_id}/``::

    config.yaml            fully resolved config (inheritance already applied)
    metrics.json           overall F1 raw + post, counts, timings, env
    predictions_eval.csv   index, entity_name, y_true, y_pred_raw, y_pred_post
    f1_by_entity.csv       class-wise breakdown
    f1_by_unit.csv         per ground-truth unit
    error_analysis.csv     top mismatch pairs per entity
    log.txt                the run's own log
    env.json               library versions, GPU, platform

and appends one row to ``results/leaderboard.csv``. ``results/`` is committed on
purpose: the metrics *are* the research record. Only the adapter binaries under
``runs/*/checkpoints/`` are git-ignored.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..config import config_hash, dump_config
from ..paths import LEADERBOARD_PATH, RUNS_DIR

logger = logging.getLogger(__name__)

LEADERBOARD_COLUMNS = [
    "run_id", "timestamp", "description", "model", "quant_bits", "lora_r",
    "n_train", "n_eval", "epochs", "lr", "max_pixels", "f1_raw", "f1_post",
    "precision", "recall", "train_seconds", "config_hash", "notes",
]


def collect_env() -> dict[str, Any]:
    """Versions and hardware, captured so a number can be reproduced later."""
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate",
                 "trl", "pandas", "numpy", "PIL"):
        try:
            module = __import__(name)
            env[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[name] = None

    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env.update({
                "gpu_name": props.name,
                "gpu_count": torch.cuda.device_count(),
                "gpu_vram_gb": round(props.total_memory / 1024 ** 3, 2),
                "gpu_capability": f"{props.major}.{props.minor}",
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            })
    except ImportError:
        env["cuda_available"] = False

    return env


class RunTracker:
    """Owns one run's output directory and writes its artefacts."""

    def __init__(self, cfg: Any, run_id: str | None = None):
        self.cfg = cfg
        self.run_id = run_id or cfg.get("run_id", "unnamed_run")
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.dir / "checkpoints"
        self.config_hash = config_hash(cfg)
        self.started_at = datetime.now(timezone.utc)
        logger.info("Run %s -> %s (config_hash=%s)", self.run_id, self.dir, self.config_hash)

    # -- individual artefacts ------------------------------------------------
    def save_config(self) -> Path:
        return dump_config(self.cfg, self.dir / "config.yaml")

    def save_env(self) -> Path:
        path = self.dir / "env.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(collect_env(), fh, indent=2, default=str)
        return path

    def save_table(self, table: pd.DataFrame, name: str) -> Path:
        path = self.dir / (name if name.endswith(".csv") else f"{name}.csv")
        table.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path.name, len(table))
        return path

    def save_metrics(self, metrics: Mapping[str, Any]) -> Path:
        path = self.dir / "metrics.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(dict(metrics), fh, indent=2, default=str)
        logger.info("Wrote metrics.json")
        return path

    def save_predictions(self, predictions: pd.DataFrame) -> Path:
        columns = [c for c in ("index", "entity_name", "y_true", "y_pred_raw", "y_pred_post")
                   if c in predictions.columns]
        return self.save_table(predictions[columns], "predictions_eval")

    # -- leaderboard ---------------------------------------------------------
    def append_leaderboard(
        self,
        metrics: Mapping[str, Any],
        train_seconds: float | None = None,
        notes: str = "",
    ) -> Path:
        """Append this run's summary row, replacing any prior row for the run_id."""
        cfg = self.cfg
        raw = metrics.get("raw", {})
        post = metrics.get("post", {})
        headline = post if post else raw

        row = {
            "run_id": self.run_id,
            "timestamp": self.started_at.isoformat(timespec="seconds"),
            "description": cfg.get("description", ""),
            "model": cfg.get("model", {}).get("id", ""),
            "quant_bits": cfg.get("quantization", {}).get("bits", ""),
            "lora_r": cfg.get("lora", {}).get("r", ""),
            "n_train": metrics.get("n_train", ""),
            "n_eval": metrics.get("n_eval", ""),
            "epochs": cfg.get("train", {}).get("num_train_epochs", ""),
            "lr": cfg.get("train", {}).get("learning_rate", ""),
            "max_pixels": int(cfg.get("processor", {}).get("max_pixels_tokens", 0)) * 28 * 28,
            "f1_raw": round(float(raw.get("f1", 0.0)), 5),
            "f1_post": round(float(post.get("f1", 0.0)), 5) if post else "",
            "precision": round(float(headline.get("precision", 0.0)), 5),
            "recall": round(float(headline.get("recall", 0.0)), 5),
            "train_seconds": round(float(train_seconds), 1) if train_seconds else "",
            "config_hash": self.config_hash,
            "notes": notes,
        }

        LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LEADERBOARD_PATH.exists():
            board = pd.read_csv(LEADERBOARD_PATH)
            board = board[board["run_id"] != self.run_id]
        else:
            board = pd.DataFrame(columns=LEADERBOARD_COLUMNS)

        board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
        board = board.reindex(columns=LEADERBOARD_COLUMNS)
        board.to_csv(LEADERBOARD_PATH, index=False)

        logger.info(
            "Leaderboard updated: %s  f1_raw=%.4f  f1_post=%s",
            self.run_id, row["f1_raw"], row["f1_post"],
        )
        return LEADERBOARD_PATH

    # -- convenience ---------------------------------------------------------
    def zip_artifacts(self, dest: str | Path | None = None) -> Path:
        """Zip the run directory (excluding checkpoints) for download."""
        import zipfile

        target = Path(dest) if dest else self.dir.parent / f"{self.run_id}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self.dir.rglob("*")):
                if path.is_file() and "checkpoints" not in path.parts:
                    zf.write(path, path.relative_to(self.dir.parent))
        logger.info("Zipped run artefacts to %s (%.1f MB)",
                    target, target.stat().st_size / 1024 ** 2)
        return target


def read_leaderboard() -> pd.DataFrame:
    """The leaderboard, best post-processed F1 first."""
    if not LEADERBOARD_PATH.exists():
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    board = pd.read_csv(LEADERBOARD_PATH)
    sort_col = "f1_post" if "f1_post" in board.columns else "f1_raw"
    return board.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


def init_leaderboard() -> Path:
    """Create an empty leaderboard with headers, so the file is committable."""
    if not LEADERBOARD_PATH.exists():
        LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=LEADERBOARD_COLUMNS).to_csv(LEADERBOARD_PATH, index=False)
        logger.info("Initialised empty leaderboard at %s", LEADERBOARD_PATH)
    return LEADERBOARD_PATH


__all__ = ["RunTracker", "read_leaderboard", "init_leaderboard", "collect_env",
           "LEADERBOARD_COLUMNS"]
