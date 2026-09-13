"""End-to-end fine-tuning pipeline: split -> images -> train -> generate -> score.

One call, ``run_finetune(config_path)``, does everything and writes every
artefact. The notebook is a thin caller; all logic lives here.

The scoring step deliberately evaluates twice -- once on raw generations and
once after post-processing -- because the delta between them is the measurement
that tells us whether to invest in better normalisation or a better model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config, load_config
from ..data.images import download_for_frames, filter_to_available
from ..data.load import load_split_frames, load_train
from ..data.splits import get_or_create_split
from ..logging_utils import add_file_handler, remove_handler, setup_logging
from ..metrics.f1 import error_analysis, f1_by_entity, f1_by_unit, f1_score
from ..paths import IMAGE_DIR, describe, ensure_dirs
from ..postprocess.normalize import PostprocessOptions, apply_postprocess
from ..results.tracker import RunTracker
from ..seed import set_seed

logger = logging.getLogger(__name__)


def prepare_data(
    cfg: Config,
    download: bool = True,
    image_dir: Path | None = None,
) -> dict[str, Any]:
    """Load train.csv, resolve the frozen split, and fetch the needed images."""
    data_cfg = cfg.get("data", {})
    train_all = load_train()

    split = get_or_create_split(
        train_all,
        seed=int(cfg.get("seed", 42)),
        eval_size=int(data_cfg.get("eval_size", 5000)),
        train_size=int(data_cfg.get("train_size", 10000)),
        path=data_cfg.get("split_file"),
        verify=bool(data_cfg.get("verify_split", True)),
    )

    eval_df, train_df = load_split_frames(split, train_all)
    logger.info("Split resolved: %d train rows, %d eval rows", len(train_df), len(eval_df))

    target_images = Path(image_dir) if image_dir else IMAGE_DIR
    report = None
    if download:
        report = download_for_frames(
            [train_df, eval_df],
            image_dir=target_images,
            threads=int(data_cfg.get("download_threads", 32)),
        )

    if bool(data_cfg.get("drop_missing_images", True)):
        train_df = filter_to_available(train_df, target_images, drop=True)
        eval_df = filter_to_available(eval_df, target_images, drop=True)

    return {
        "split": split,
        "train_df": train_df,
        "eval_df": eval_df,
        "image_dir": target_images,
        "download_report": report.as_dict() if report else None,
    }


def score(
    predictions: pd.DataFrame,
    cfg: Config,
    tracker: RunTracker,
) -> dict[str, Any]:
    """Score raw and post-processed predictions; write all breakdown tables."""
    y_true = predictions["y_true"].fillna("").astype(str).tolist()
    entities = predictions["entity_name"].astype(str).tolist()
    raw_preds = predictions["y_pred_raw"].fillna("").astype(str).tolist()

    logger.info("--- Scoring RAW generations ---")
    raw_scores = f1_score(y_true, raw_preds)

    opts = PostprocessOptions.from_config(cfg.get("postprocess", {}))
    logger.info("--- Post-processing (%s) ---", opts)
    post_preds, rule_counts = apply_postprocess(raw_preds, entities, opts)
    predictions = predictions.copy()
    predictions["y_pred_post"] = post_preds

    logger.info("--- Scoring POST-PROCESSED predictions ---")
    post_scores = f1_score(y_true, post_preds)

    delta = post_scores["f1"] - raw_scores["f1"]
    logger.info(
        "Post-processing changed F1 by %+.4f (raw %.4f -> post %.4f). %s",
        delta, raw_scores["f1"], post_scores["f1"],
        "Worth keeping." if delta > 0 else "Review the rules -- this is a regression.",
    )

    by_entity = f1_by_entity(y_true, post_preds, entities)
    by_unit = f1_by_unit(y_true, post_preds, entities)
    errors = error_analysis(y_true, post_preds, entities, top_k=20)

    by_entity_raw = f1_by_entity(y_true, raw_preds, entities)[["entity_name", "f1"]] \
        .rename(columns={"f1": "f1_raw"})
    by_entity = by_entity.merge(by_entity_raw, on="entity_name", how="left")
    by_entity["f1_delta_from_postprocess"] = by_entity["f1"] - by_entity["f1_raw"]

    tracker.save_predictions(predictions)
    tracker.save_table(by_entity, "f1_by_entity")
    tracker.save_table(by_unit, "f1_by_unit")
    tracker.save_table(errors, "error_analysis")

    return {
        "raw": raw_scores,
        "post": post_scores,
        "f1_delta": delta,
        "postprocess_rules": dict(rule_counts),
        "by_entity": by_entity,
        "by_unit": by_unit,
        "errors": errors,
        "predictions": predictions,
    }


def run_finetune(
    config_path: str | Path,
    download_images: bool = True,
    image_dir: str | Path | None = None,
    max_eval_rows: int | None = None,
    skip_training: bool = False,
) -> dict[str, Any]:
    """Train, generate, score, and record a complete run.

    ``skip_training=True`` evaluates the base model without fine-tuning, which
    is the zero-shot baseline the fine-tune has to beat.
    """
    setup_logging()
    ensure_dirs()

    cfg = load_config(config_path)
    tracker = RunTracker(cfg)
    log_handler = add_file_handler(tracker.dir / "log.txt")

    try:
        logger.info("=" * 72)
        logger.info("RUN %s", cfg.run_id)
        logger.info("%s", cfg.description)
        logger.info("Environment: %s", describe())
        logger.info("=" * 72)

        set_seed(int(cfg.get("seed", 42)))
        tracker.save_config()

        data = prepare_data(
            cfg,
            download=download_images,
            image_dir=Path(image_dir) if image_dir else None,
        )
        train_df, eval_df = data["train_df"], data["eval_df"]

        # Imports are deferred so that split/EDA work does not require torch.
        from ..data.dataset import build_datasets
        from ..inference.generate import generate_predictions
        from ..models.qwen2vl import gpu_report, load_model
        from ..train.trainer import train as run_training

        gpu = gpu_report()
        model, processor = load_model(cfg, for_training=not skip_training)

        train_stats: dict[str, Any] = {}
        history = pd.DataFrame()
        if not skip_training:
            train_ds, eval_ds = build_datasets(
                train_df, eval_df, cfg, data["image_dir"],
                eval_loss_fraction=float(cfg.get("train", {}).get("eval_loss_fraction", 0.1)),
            )
            outcome = run_training(
                model, processor, train_ds, eval_ds, cfg,
                output_dir=tracker.checkpoints_dir,
            )
            train_stats = outcome["stats"]
            history = outcome["history"]
            model = outcome["trainer"].model
            history.to_csv(tracker.dir / "loss_history.csv", index=False)
        else:
            logger.info("skip_training=True -- evaluating the base model zero-shot")

        model.config.use_cache = True
        predictions = generate_predictions(
            model, processor, eval_df, cfg,
            image_dir=data["image_dir"], max_rows=max_eval_rows,
        )

        results = score(predictions, cfg, tracker)

        metrics = {
            "run_id": cfg.run_id,
            "description": cfg.description,
            "config_hash": tracker.config_hash,
            "n_train": len(train_df),
            "n_eval": len(predictions),
            "skip_training": skip_training,
            "raw": results["raw"],
            "post": results["post"],
            "f1_delta_from_postprocess": results["f1_delta"],
            "postprocess_rules": results["postprocess_rules"],
            "train": train_stats,
            "gpu": gpu,
            "download_report": data["download_report"],
            "split": {
                "eval_size": len(data["split"]["eval_5k"]),
                "train_size": len(data["split"].get("train_subset", [])),
                "seed": data["split"].get("seed"),
            },
        }
        tracker.save_metrics(metrics)
        tracker.save_env()
        tracker.append_leaderboard(
            metrics, train_seconds=train_stats.get("train_seconds"),
            notes="zero-shot baseline" if skip_training else "",
        )

        logger.info("=" * 72)
        logger.info(
            "RUN %s COMPLETE  |  F1 raw=%.4f  post=%.4f  (P=%.4f R=%.4f)",
            cfg.run_id, results["raw"]["f1"], results["post"]["f1"],
            results["post"]["precision"], results["post"]["recall"],
        )
        logger.info("=" * 72)

        return {
            "config": cfg,
            "tracker": tracker,
            "metrics": metrics,
            "history": history,
            **results,
        }

    finally:
        remove_handler(log_handler)


__all__ = ["run_finetune", "prepare_data", "score"]
