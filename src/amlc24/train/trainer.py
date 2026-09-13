"""HF ``Trainer`` wiring for LoRA SFT, with adapter-only checkpointing.

Notes on the choices here:

* **Adapter-only checkpoints.** A merged 7B checkpoint is ~15 GB; the LoRA
  adapter is ~40 MB. Kaggle's 20 GB working-directory limit makes merged
  checkpoints impossible to keep three of, and the adapter is all we need to
  reproduce the run. ``save_best`` is by eval loss.
* **``remove_unused_columns=False``.** The default drops any batch key the
  model's ``forward`` signature does not name, which for Qwen2-VL silently
  deletes ``pixel_values`` and ``image_grid_thw`` -- training then runs on text
  alone and quietly scores near zero.
* **``dataloader_num_workers=0`` by default.** PIL images travel through the
  collator, and worker processes have to pickle them; on Kaggle the copy
  overhead outweighs the parallel decode.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.dataset import QwenVLCollator

logger = logging.getLogger(__name__)


class LossHistoryCallback:
    """Collect train/eval loss into a DataFrame for the notebook's curves.

    Implemented against ``TrainerCallback`` at runtime (imported lazily so the
    module can be imported without transformers installed, which the tests do).
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D102
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": state.epoch}
        entry.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        self.records.append(entry)

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["step", "epoch", "loss", "eval_loss"])
        return pd.DataFrame(self.records)


def _make_callback() -> Any:
    """Build a real ``TrainerCallback`` subclass bound to our history object."""
    from transformers import TrainerCallback

    history = LossHistoryCallback()

    class _Callback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            history.on_log(args, state, control, logs=logs, **kwargs)

    callback = _Callback()
    callback.history = history  # type: ignore[attr-defined]
    return callback


def build_training_args(cfg: Any, output_dir: str | Path) -> Any:
    """``TrainingArguments`` from the ``train:`` config block.

    ``num_train_epochs``, batch size, accumulation, optimiser, scheduler, and
    fp16 all come from config so an OOM or a schedule change is a YAML edit.
    """
    from transformers import TrainingArguments

    train_cfg = cfg.get("train", {})
    output_dir = Path(output_dir)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        fp16=bool(train_cfg.get("fp16", True)),
        bf16=False,  # Turing has no bf16; never enable this on a T4.
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(train_cfg.get("logging_steps", 25)),
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=False,
        seed=int(cfg.get("seed", 42)),
        disable_tqdm=False,
        label_names=["labels"],
    )
    logger.info(
        "TrainingArguments: epochs=%s bs=%d x accum=%d (effective %d), lr=%.2e %s, "
        "optim=%s, fp16=%s, grad_ckpt=%s",
        args.num_train_epochs, args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.per_device_train_batch_size * args.gradient_accumulation_steps,
        args.learning_rate, args.lr_scheduler_type, args.optim,
        args.fp16, args.gradient_checkpointing,
    )
    return args


def build_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
    cfg: Any,
    output_dir: str | Path,
) -> tuple[Any, Any]:
    """Assemble the ``Trainer``. Returns ``(trainer, loss_history_callback)``."""
    from transformers import Trainer

    prompt_cfg = cfg.get("prompt", {})
    collator = QwenVLCollator(
        processor=processor,
        template=prompt_cfg.get("template", "prompt_v1"),
        include_allowed_units=bool(prompt_cfg.get("include_allowed_units", True)),
        max_length=int(cfg.get("train", {}).get("max_length", 1024)),
        is_train=True,
    )

    callback = _make_callback()
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[callback],
    )
    return trainer, callback


def train(
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
    cfg: Any,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run fine-tuning and save the best adapter.

    Returns training stats, the loss history, and the adapter path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer, callback = build_trainer(
        model, processor, train_dataset, eval_dataset, cfg, output_dir
    )

    logger.info(
        "Starting training: %d train rows, %d eval rows",
        len(train_dataset), len(eval_dataset),
    )
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    logger.info(
        "Training finished in %.1f min (%.2f h). Final train loss: %.4f",
        elapsed / 60, elapsed / 3600, result.training_loss,
    )

    adapter_dir = output_dir / "adapter_best"
    trainer.model.save_pretrained(str(adapter_dir))
    try:
        processor.save_pretrained(str(adapter_dir))
    except (OSError, AttributeError) as exc:
        logger.warning("Could not save processor alongside adapter: %s", exc)
    logger.info("Saved LoRA adapter to %s", adapter_dir)

    history = callback.history.to_frame()
    history_path = output_dir / "loss_history.csv"
    history.to_csv(history_path, index=False)

    peak_vram_gb = _peak_vram_gb()
    stats = {
        "train_seconds": elapsed,
        "train_minutes": elapsed / 60,
        "training_loss": float(result.training_loss),
        "global_step": int(result.global_step),
        "n_train": len(train_dataset),
        "n_eval_loss_rows": len(eval_dataset),
        "adapter_path": str(adapter_dir),
        "peak_vram_gb": peak_vram_gb,
        "best_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_eval_loss": getattr(trainer.state, "best_metric", None),
    }

    with (output_dir / "train_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)

    logger.info("Peak VRAM: %.2f GB; best eval_loss: %s",
                peak_vram_gb or 0.0, stats["best_eval_loss"])

    return {"stats": stats, "history": history, "trainer": trainer,
            "adapter_path": str(adapter_dir)}


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except ImportError:
        pass
    return None


def plot_loss_curves(history: pd.DataFrame, save_to: str | Path | None = None):
    """Train and eval loss against step."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if "loss" in history.columns:
        train_points = history.dropna(subset=["loss"])
        ax.plot(train_points["step"], train_points["loss"], label="train loss",
                color="#4C78A8", lw=1.2)
    if "eval_loss" in history.columns:
        eval_points = history.dropna(subset=["eval_loss"])
        ax.plot(eval_points["step"], eval_points["eval_loss"], label="eval loss",
                color="#E45756", marker="o", lw=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (completion tokens)")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    return fig


__all__ = ["train", "build_trainer", "build_training_args", "plot_loss_curves",
           "LossHistoryCallback"]
