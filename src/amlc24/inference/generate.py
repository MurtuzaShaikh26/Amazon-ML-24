"""Batched greedy generation over an evaluation frame.

Returns **raw** decoded strings, before any post-processing. That separation is
the point: the pipeline scores the raw output and the post-processed output
against the same labels, so the contribution of ``postprocess/`` is a number in
``metrics.json`` rather than an article of faith.

Decoding is deterministic (``do_sample=False``, ``num_beams=1``). Beam search
would cost several times the wall-clock for an output that is at most a number
and a unit, and sampling would make the eval score irreproducible.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import pandas as pd

from ..data.dataset import EntityExtractionDataset, QwenVLCollator

logger = logging.getLogger(__name__)


def _resolve_batch_size(cfg: Any) -> int:
    """Inference batch size. Small by default: with ~256 visual tokens per image
    plus the KV cache, a T4 that just finished training has little headroom."""
    return int(cfg.get("inference", {}).get("batch_size", 4))


def generate_predictions(
    model: Any,
    processor: Any,
    eval_df: pd.DataFrame,
    cfg: Any,
    image_dir: Any = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Generate one prediction per row of ``eval_df``.

    Returns a frame with ``index``, ``entity_name``, ``y_true`` (when present),
    and ``y_pred_raw``.
    """
    import torch

    inf_cfg = cfg.get("inference", {})
    prompt_cfg = cfg.get("prompt", {})
    batch_size = _resolve_batch_size(cfg)
    max_new_tokens = int(inf_cfg.get("max_new_tokens", 24))
    num_beams = int(inf_cfg.get("num_beams", 1))

    frame = eval_df.head(max_rows) if max_rows else eval_df
    dataset = EntityExtractionDataset(
        frame,
        image_dir=image_dir,
        template=prompt_cfg.get("template", "prompt_v1"),
        include_allowed_units=bool(prompt_cfg.get("include_allowed_units", True)),
        is_train=False,
    )
    collator = QwenVLCollator(
        processor=processor,
        template=prompt_cfg.get("template", "prompt_v1"),
        include_allowed_units=bool(prompt_cfg.get("include_allowed_units", True)),
        max_length=int(cfg.get("train", {}).get("max_length", 1024)),
        is_train=False,
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    original_side = getattr(tokenizer, "padding_side", "right")
    # Decoder-only generation requires left padding, otherwise short sequences
    # continue from pad tokens and emit garbage.
    tokenizer.padding_side = "left"

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    model.eval()
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    start = time.time()

    try:
        with torch.inference_mode():
            for batch_i, batch in enumerate(loader, start=1):
                row_index = batch.pop("row_index").tolist()
                inputs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )

                prompt_len = inputs["input_ids"].shape[1]
                completions = generated[:, prompt_len:]
                texts = processor.batch_decode(
                    completions, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for idx, text in zip(row_index, texts):
                    rows.append({"index": int(idx), "y_pred_raw": text.strip()})

                if batch_i % 25 == 0 or batch_i * batch_size >= len(dataset):
                    done = min(batch_i * batch_size, len(dataset))
                    rate = done / max(time.time() - start, 1e-6)
                    eta = (len(dataset) - done) / max(rate, 1e-6)
                    logger.info(
                        "Generated %d/%d (%.1f rows/s, ETA %.1f min)",
                        done, len(dataset), rate, eta / 60,
                    )
    finally:
        tokenizer.padding_side = original_side

    elapsed = time.time() - start
    logger.info("Generation complete: %d rows in %.1f min (%.2f rows/s)",
                len(rows), elapsed / 60, len(rows) / max(elapsed, 1e-6))

    preds = pd.DataFrame(rows)
    keep = ["index", "entity_name"] + (["entity_value"] if "entity_value" in frame.columns else [])
    out = frame[keep].merge(preds, on="index", how="left")
    out["y_pred_raw"] = out["y_pred_raw"].fillna("")

    if "entity_value" in out.columns:
        out = out.rename(columns={"entity_value": "y_true"})
        out["y_true"] = out["y_true"].fillna("").astype(str)

    n_empty = int((out["y_pred_raw"].str.strip() == "").sum())
    logger.info("Raw predictions: %d empty (%.2f%%), %d non-empty",
                n_empty, 100 * n_empty / max(len(out), 1), len(out) - n_empty)
    return out


def predict_test_set(
    model: Any,
    processor: Any,
    test_df: pd.DataFrame,
    cfg: Any,
    image_dir: Any = None,
) -> pd.DataFrame:
    """Predictions for the blind test set, in submission shape.

    Returns exactly two columns, ``index`` and ``prediction``, as the
    competition requires. Post-processing is applied because the submission is
    scored by the same exact-match metric.
    """
    from ..postprocess.normalize import PostprocessOptions, apply_postprocess

    raw = generate_predictions(model, processor, test_df, cfg, image_dir)
    opts = PostprocessOptions.from_config(cfg.get("postprocess", {}))
    cleaned, _ = apply_postprocess(
        raw["y_pred_raw"].tolist(), raw["entity_name"].tolist(), opts
    )
    return pd.DataFrame({"index": raw["index"], "prediction": cleaned})


__all__ = ["generate_predictions", "predict_test_set"]
