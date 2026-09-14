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
from ..metrics.f1 import (
    error_analysis,
    f1_by_entity,
    f1_by_group,
    f1_by_unit,
    f1_score,
    macro_f1,
)
from ..paths import IMAGE_DIR, describe, ensure_dirs
from ..postprocess.normalize import PostprocessOptions, apply_postprocess
from ..results.tracker import RunTracker
from ..seed import set_seed

logger = logging.getLogger(__name__)


def _fmt4(value: float) -> str:
    """pandas >= 2 requires float_format to be a callable."""
    return f"{value:.4f}"


def prepare_data(
    cfg: Config,
    download: bool = True,
    image_dir: Path | None = None,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
) -> dict[str, Any]:
    """Load train.csv, resolve the frozen split, and fetch the needed images.

    ``max_train_rows`` / ``max_eval_rows`` truncate the resolved split **before**
    the image download, which is what makes a smoke test cheap: they cap the
    work without touching the frozen split file. They are for plumbing checks
    only -- a truncated eval set is not a comparable score.
    """
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

    if max_train_rows or max_eval_rows:
        if max_train_rows:
            train_df = train_df.head(int(max_train_rows)).reset_index(drop=True)
        if max_eval_rows:
            eval_df = eval_df.head(int(max_eval_rows)).reset_index(drop=True)
        logger.warning(
            "TRUNCATED to %d train / %d eval rows. This is a plumbing check, "
            "NOT a comparable result -- the eval set is no longer the frozen 5k.",
            len(train_df), len(eval_df),
        )

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

    # Macro F1 weights all eight entities equally. Micro (overall) F1 is
    # dominated by item_weight at 38.95% of the data, so macro is the number
    # that actually reflects whether class-weighted training helped.
    macro_post = macro_f1(by_entity)
    macro_raw = macro_f1(by_entity_raw.rename(columns={"f1_raw": "f1"}))
    logger.info(
        "Macro F1 (unweighted mean over entities): raw=%.4f post=%.4f  |  "
        "micro F1: raw=%.4f post=%.4f",
        macro_raw, macro_post, raw_scores["f1"], post_scores["f1"],
    )

    by_group = None
    if "group_id" in predictions.columns:
        by_group = f1_by_group(y_true, post_preds, predictions["group_id"].tolist())
        tracker.save_table(by_group, "f1_by_group")

    ablation = postprocess_ablation(y_true, raw_preds, entities, opts)

    tracker.save_predictions(predictions)
    tracker.save_table(by_entity, "f1_by_entity")
    tracker.save_table(by_unit, "f1_by_unit")
    tracker.save_table(errors, "error_analysis")
    tracker.save_table(ablation, "postprocess_ablation")

    return {
        "raw": raw_scores,
        "post": post_scores,
        "f1_delta": delta,
        "macro_f1_raw": macro_raw,
        "macro_f1_post": macro_post,
        "postprocess_rules": dict(rule_counts),
        "by_entity": by_entity,
        "by_unit": by_unit,
        "by_group": by_group,
        "errors": errors,
        "ablation": ablation,
        "predictions": predictions,
    }


def postprocess_ablation(
    y_true: list[str],
    raw_preds: list[str],
    entities: list[str],
    config_opts: PostprocessOptions | None = None,
) -> pd.DataFrame:
    """Score every ``number_format`` x ``range_rule`` variant on the same outputs.

    Pure string work -- seconds of CPU, no GPU -- so every run reports which
    post-processing setting would have scored best instead of taking one config
    choice on faith. The smoke run is why this exists: forcing float-style
    numbers *lowered* F1, because the fine-tuned model had already learned the
    per-entity number convention.

    Choosing the best variant on the eval set is a mild form of selection on
    that set; with 16 discrete variants on 5,000 rows the optimism is small, but
    confirm a switch on the next run rather than reporting the maximum as-is.
    """
    import logging as _logging
    from dataclasses import replace

    from ..postprocess.normalize import NUMBER_FORMATS, RANGE_RULES

    base = config_opts or PostprocessOptions()

    def summary(scores: dict, preds: list[str]) -> dict:
        return {
            "f1": scores["f1"], "precision": scores["precision"], "recall": scores["recall"],
            "tp": scores["tp"], "fp": scores["fp"], "fn": scores["fn"],
            "macro_f1": macro_f1(f1_by_entity(y_true, preds, entities)),
        }

    # The metric and post-processing functions log at INFO on every call;
    # 17 variants would bury the run log.
    noisy = [_logging.getLogger("amlc24.metrics.f1"),
             _logging.getLogger("amlc24.postprocess.normalize")]
    previous = [lg.level for lg in noisy]
    for lg in noisy:
        lg.setLevel(_logging.WARNING)
    try:
        rows = [{"variant": "raw (no post-processing)", "number_format": "-",
                 "range_rule": "-", **summary(f1_score(y_true, raw_preds), raw_preds)}]
        for fmt in NUMBER_FORMATS:
            for rule in RANGE_RULES:
                opts = replace(base, number_format=fmt, range_rule=rule)
                preds, _ = apply_postprocess(raw_preds, entities, opts)
                rows.append({"variant": f"{fmt} / {rule}", "number_format": fmt,
                             "range_rule": rule, **summary(f1_score(y_true, preds), preds)})
    finally:
        for lg, level in zip(noisy, previous):
            lg.setLevel(level)

    table = pd.DataFrame(rows)
    table["is_config"] = (table["number_format"] == base.number_format) & (
        table["range_rule"] == base.range_rule
    )
    table = table.sort_values("f1", ascending=False, ignore_index=True)

    best = table.iloc[0]
    chosen = table[table["is_config"]].iloc[0]
    logger.info("Post-processing ablation:\n%s",
                table.to_string(index=False, float_format=_fmt4))
    logger.info(
        "Best variant: %s (F1 %.4f). Configured: %s (F1 %.4f).%s",
        best["variant"], best["f1"], chosen["variant"], chosen["f1"],
        "" if best["variant"] == chosen["variant"]
        else " Consider switching the config -- and confirm on the next run.",
    )
    return table


def run_finetune(
    config_path: str | Path,
    download_images: bool = True,
    image_dir: str | Path | None = None,
    max_eval_rows: int | None = None,
    max_train_rows: int | None = None,
    skip_training: bool = False,
) -> dict[str, Any]:
    """Train, generate, score, and record a complete run.

    ``skip_training=True`` evaluates the base model without fine-tuning, which
    is the zero-shot baseline the fine-tune has to beat.

    ``max_train_rows`` / ``max_eval_rows`` cap the work for a smoke test. They
    truncate the resolved split before images are downloaded, so a plumbing
    check costs minutes rather than hours. Any run using them is marked as
    truncated in ``metrics.json`` and on the leaderboard, because its score is
    not comparable to a full run.
    """
    setup_logging()
    ensure_dirs()

    cfg = load_config(config_path)
    # A smoke run must never share a directory or leaderboard row with the real
    # run: its checkpoints would sit beside the real ones and its row would be
    # overwritten (or worse, survive) under the real run_id.
    truncated = bool(max_train_rows or max_eval_rows)
    tracker = RunTracker(cfg, run_id=f"{cfg.run_id}_smoke" if truncated else None)
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
            max_train_rows=max_train_rows,
            max_eval_rows=max_eval_rows,
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
            model, processor, eval_df, cfg, image_dir=data["image_dir"],
        )

        results = score(predictions, cfg, tracker)

        metrics = {
            "run_id": cfg.run_id,
            "description": cfg.description,
            "config_hash": tracker.config_hash,
            "n_train": len(train_df),
            "n_eval": len(predictions),
            "skip_training": skip_training,
            "truncated": bool(max_train_rows or max_eval_rows),
            "raw": results["raw"],
            "post": results["post"],
            "macro_f1_raw": results["macro_f1_raw"],
            "macro_f1_post": results["macro_f1_post"],
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
        note_parts = []
        if skip_training:
            note_parts.append("zero-shot baseline")
        if max_train_rows or max_eval_rows:
            note_parts.append("TRUNCATED smoke run -- not comparable")
        tracker.append_leaderboard(
            metrics, train_seconds=train_stats.get("train_seconds"),
            notes="; ".join(note_parts),
        )

        logger.info("=" * 72)
        logger.info(
            "RUN %s COMPLETE  |  micro F1 raw=%.4f post=%.4f  |  macro F1 post=%.4f  "
            "(P=%.4f R=%.4f)",
            cfg.run_id, results["raw"]["f1"], results["post"]["f1"],
            results["macro_f1_post"],
            results["post"]["precision"], results["post"]["recall"],
        )
        logger.info("Per-entity F1:\n%s", results["by_entity"].to_string(
            index=False, float_format=_fmt4))
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


def run_test_predictions(
    config_path: str | Path,
    adapter_path: str | Path | None = None,
    download_images: bool = True,
    image_dir: str | Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Predict over the blind ``test.csv`` and write a valid submission.

    Loads the base model plus the trained LoRA adapter (defaulting to the run's
    ``adapter_best``), downloads the test images, generates, post-processes, and
    writes ``submission.csv`` through the validating writer.

    Kept separate from ``run_finetune`` because it is a distinct, expensive
    operation (~131k rows) that you only want at submission time, not on every
    experiment.
    """
    setup_logging()
    ensure_dirs()

    cfg = load_config(config_path)
    tracker = RunTracker(cfg)

    from ..data.load import load_test
    from ..inference.generate import generate_predictions
    from ..models.qwen2vl import load_for_inference
    from ..results.submission import write_submission

    if adapter_path is None:
        candidate = tracker.checkpoints_dir / "adapter_best"
        adapter_path = candidate if candidate.exists() else None
        if adapter_path is None:
            logger.warning(
                "No adapter found at %s; predicting with the BASE model (zero-shot).",
                candidate,
            )

    test_df = load_test()
    if max_rows:
        test_df = test_df.head(int(max_rows)).reset_index(drop=True)
    logger.info("Predicting over %d test rows", len(test_df))

    target_images = Path(image_dir) if image_dir else IMAGE_DIR
    if download_images:
        download_for_frames(
            [test_df], image_dir=target_images,
            threads=int(cfg.get("data", {}).get("download_threads", 32)),
        )

    model, processor = load_for_inference(cfg, str(adapter_path) if adapter_path else None)
    raw = generate_predictions(model, processor, test_df, cfg, image_dir=target_images)

    opts = PostprocessOptions.from_config(cfg.get("postprocess", {}))
    cleaned, rule_counts = apply_postprocess(
        raw["y_pred_raw"].tolist(), raw["entity_name"].tolist(), opts
    )
    submission = pd.DataFrame({"index": raw["index"], "prediction": cleaned})

    path = write_submission(
        submission, tracker.dir / "submission.csv", test_df=test_df
    )
    return {
        "submission": submission,
        "path": path,
        "raw": raw,
        "postprocess_rules": dict(rule_counts),
        "tracker": tracker,
    }


__all__ = ["run_finetune", "run_test_predictions", "prepare_data", "score"]
