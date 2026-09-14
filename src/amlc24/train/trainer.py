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

from ..data.dataset import IGNORE_INDEX, QwenVLCollator
from .weighting import describe_weights, weights_from_config

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


class TrainingTimeBudget:
    """Wall-clock guard so a run always finishes inside the Kaggle session.

    The smoke run measured ~2.7 s per training sample on a T4 at 8-bit, so a
    mis-sized config can silently need 20+ hours and die at the 12 h limit with
    nothing scored. Once ``max_hours`` elapses the trainer evaluates, saves and
    stops; ``load_best_model_at_end`` restores the best checkpoint, and
    generation and scoring still run.
    """

    def __init__(self, max_hours: float | None):
        self.max_seconds = float(max_hours) * 3600 if max_hours else None
        self.projection_logged = False

    def should_stop(self, elapsed_seconds: float) -> bool:
        return self.max_seconds is not None and elapsed_seconds >= self.max_seconds

    @staticmethod
    def project_total_seconds(elapsed_seconds: float, step: int, max_steps: int) -> float | None:
        if step <= 0 or max_steps <= 0:
            return None
        return elapsed_seconds / step * max_steps


def _make_time_budget_callback(max_hours: float | None, projection_step: int = 20) -> Any:
    """``TrainerCallback`` that logs a runtime projection and enforces the budget."""
    import time

    from transformers import TrainerCallback

    budget = TrainingTimeBudget(max_hours)

    class _TimeBudgetCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.start
            if not budget.projection_logged and state.global_step >= projection_step:
                budget.projection_logged = True
                total = budget.project_total_seconds(elapsed, state.global_step, state.max_steps)
                if total is not None:
                    logger.info(
                        "Throughput: %.1f s/step; projected training time %.2f h for %d steps",
                        elapsed / state.global_step, total / 3600, state.max_steps,
                    )
                    if budget.max_seconds and total > budget.max_seconds:
                        logger.warning(
                            "Projected %.2f h exceeds the %.2f h budget; training will stop "
                            "at the budget and keep the best checkpoint.",
                            total / 3600, budget.max_seconds / 3600,
                        )
            if budget.should_stop(elapsed):
                logger.warning(
                    "Time budget of %.2f h reached at step %d/%d; evaluating, saving, stopping.",
                    budget.max_seconds / 3600, state.global_step, state.max_steps,
                )
                control.should_evaluate = True
                control.should_save = True
                control.should_training_stop = True
            return control

    return _TimeBudgetCallback()


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
    # On T4 x2 Trainer sets train_batch_size = per_device * n_gpu = 2. It skips
    # DataParallel for 8-bit models but still doubles the batch, which OOMs.
    # The model lives on one device, so n_gpu must be 1 regardless of what is
    # visible.
    if args.n_gpu > 1:
        logger.warning("%d GPUs visible; forcing n_gpu=1 (model is on one device)",
                       args.n_gpu)
        args._n_gpu = 1
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


def _weighted_trainer_class() -> Any:
    """Build a ``Trainer`` subclass that applies per-sample class weights.

    Two paths, chosen per batch:

    * **Batch size 1** (our configuration) -- the model's own loss is already
      the mean over that single sample's completion tokens, so the weighted
      loss is just ``loss * w``. This costs nothing: no second forward, no
      extra logits copy. With ``gradient_accumulation_steps=8`` the eight
      weighted micro-batches accumulate into exactly the weighted mean.
    * **Batch size > 1** -- fall back to recomputing per-token cross-entropy
      with ``reduction="none"`` so each sample can be scaled independently.
      This materialises an fp32 view of the logits (vocab ≈ 152k), which is
      expensive; it is why batch size 1 is the configured default rather than
      merely a memory convenience.
    """
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Must be removed before forward(): the model does not accept it.
            weights = inputs.pop("sample_weight", None)

            if weights is None:
                outputs = model(**inputs)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            weights = weights.to(model.device if hasattr(model, "device") else weights.device)

            if inputs["input_ids"].shape[0] == 1:
                outputs = model(**inputs)
                loss = outputs.loss * weights.to(outputs.loss.device).reshape(())
                return (loss, outputs) if return_outputs else loss

            labels = inputs.pop("labels")
            outputs = model(**inputs)

            # Standard causal-LM shift: predict token t+1 from position t.
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(shift_labels.shape)

            mask = (shift_labels != IGNORE_INDEX).float()
            per_sample = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            loss = (per_sample * weights.to(per_sample.device)).mean()

            outputs["loss"] = loss
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def build_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
    cfg: Any,
    output_dir: str | Path,
    class_weights: dict[str, float] | None = None,
) -> tuple[Any, Any]:
    """Assemble the ``Trainer``. Returns ``(trainer, loss_history_callback)``.

    When ``class_weights`` is non-empty a weighted-loss ``Trainer`` subclass is
    used and the collator emits a per-sample weight; otherwise the plain
    ``Trainer`` and the model's own loss are used unchanged.
    """
    from transformers import Trainer

    prompt_cfg = cfg.get("prompt", {})
    collator = QwenVLCollator(
        processor=processor,
        template=prompt_cfg.get("template", "prompt_v1"),
        include_allowed_units=bool(prompt_cfg.get("include_allowed_units", True)),
        max_length=int(cfg.get("train", {}).get("max_length", 1024)),
        is_train=True,
        class_weights=class_weights,
    )

    trainer_cls = _weighted_trainer_class() if class_weights else Trainer
    if class_weights:
        logger.info("Using weighted-loss Trainer over %d classes", len(class_weights))

    callback = _make_callback()
    trainer = trainer_cls(
        model=model,
        args=build_training_args(cfg, output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            callback,
            _make_time_budget_callback(cfg.get("train", {}).get("max_train_hours")),
        ],
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

    # Class weights are derived from the *training subset's* own distribution,
    # not the full file, so they match what the optimiser actually sees.
    train_entities = train_dataset.df["entity_name"].astype(str).tolist()
    class_weights = weights_from_config(
        train_entities, cfg.get("train", {}).get("class_weights", {})
    )

    trainer, callback = build_trainer(
        model, processor, train_dataset, eval_dataset, cfg, output_dir,
        class_weights=class_weights,
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
        "class_weighted_loss": bool(class_weights),
        "class_weights": describe_weights(class_weights) if class_weights else None,
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
