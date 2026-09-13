"""Torch dataset and collator for Qwen2-VL supervised fine-tuning.

The collator is where the two non-obvious requirements live:

1. **Completion-only loss.** Labels are ``-100`` everywhere except the answer
   tokens. Training on the prompt spends capacity re-learning our own
   instructions and, worse, degrades output formatting -- and formatting is the
   metric here.
2. **Padding and image tokens.** Pad positions are masked too, and so are the
   vision placeholder ids: the image-token embeddings are replaced by patch
   features inside the model, and asking the LM head to predict them is
   meaningless.

The boundary between prompt and completion is found by tokenising the prompt
alone and using its length. That is exact for a left-to-right tokeniser with no
cross-boundary merges, which holds for Qwen2's BPE on the chat template, whose
prompt always ends on the ``<|im_start|>assistant\\n`` marker.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..prompts.templates import render_for_inference, render_for_training
from .images import load_image

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class EntityExtractionDataset(Dataset):
    """One row = one (image, entity_name) -> entity_value example.

    Holds only the DataFrame; images are decoded lazily in ``__getitem__`` so
    memory stays flat regardless of split size.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Any = None,
        template: str = "prompt_v1",
        include_allowed_units: bool = True,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.template = template
        self.include_allowed_units = include_allowed_units
        self.is_train = is_train

        if is_train and "entity_value" not in self.df.columns:
            raise ValueError("Training dataset requires an `entity_value` column")

        logger.info(
            "Dataset: %d rows, template=%s, allowed_units_in_prompt=%s, mode=%s",
            len(self.df), template, include_allowed_units,
            "train" if is_train else "inference",
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        answer = str(row["entity_value"]).strip() if self.is_train else None
        return {
            "index": int(row["index"]),
            "image": load_image(str(row["image_link"]), self.image_dir),
            "entity_name": str(row["entity_name"]),
            "answer": answer,
        }


class QwenVLCollator:
    """Batch samples into Qwen2-VL inputs with completion-only labels."""

    def __init__(
        self,
        processor: Any,
        template: str = "prompt_v1",
        include_allowed_units: bool = True,
        max_length: int = 1024,
        is_train: bool = True,
        class_weights: dict[str, float] | None = None,
    ):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.template = template
        self.include_allowed_units = include_allowed_units
        self.max_length = max_length
        self.is_train = is_train
        # Empty/None means "unweighted"; the trainer then takes the cheap path
        # and uses the model's own loss.
        self.class_weights = class_weights or {}
        self._image_token_ids = self._resolve_image_token_ids()

    def _resolve_image_token_ids(self) -> set[int]:
        """Ids of vision placeholder tokens, which must never be a loss target."""
        ids: set[int] = set()
        for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
            except (AttributeError, KeyError):
                continue
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)
        return ids

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [f["image"] for f in features]

        if self.is_train:
            rendered = [
                render_for_training(
                    self.processor, f["entity_name"], f["answer"] or "",
                    self.template, self.include_allowed_units,
                )
                for f in features
            ]
            prompts = [p for p, _ in rendered]
            texts = [t for _, t in rendered]
        else:
            prompts = [
                render_for_inference(
                    self.processor, f["entity_name"],
                    self.template, self.include_allowed_units,
                )
                for f in features
            ]
            texts = prompts

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        if self.is_train:
            batch["labels"] = self._build_labels(batch, prompts, images)
            if self.class_weights:
                batch["sample_weight"] = torch.tensor(
                    [float(self.class_weights.get(f["entity_name"], 1.0)) for f in features],
                    dtype=torch.float32,
                )
        else:
            batch["row_index"] = torch.tensor([f["index"] for f in features], dtype=torch.long)

        return batch


    def _prompt_lengths(self, prompts: Sequence[str], images: Sequence[Any]) -> list[int]:
        """Token length of each prompt, including its expanded vision tokens."""
        lengths = []
        for prompt, image in zip(prompts, images):
            encoded = self.processor(
                text=[prompt], images=[image], return_tensors="pt",
                padding=False, truncation=True, max_length=self.max_length,
            )
            lengths.append(int(encoded["input_ids"].shape[1]))
        return lengths

    def _build_labels(
        self, batch: dict[str, torch.Tensor], prompts: Sequence[str], images: Sequence[Any]
    ) -> torch.Tensor:
        """Mask everything that is not an answer token."""
        input_ids: torch.Tensor = batch["input_ids"]
        labels = input_ids.clone()

        # 1. padding
        attention = batch.get("attention_mask")
        if attention is not None:
            labels[attention == 0] = IGNORE_INDEX
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[input_ids == pad_id] = IGNORE_INDEX

        # 2. vision placeholders
        for token_id in self._image_token_ids:
            labels[input_ids == token_id] = IGNORE_INDEX

        # 3. prompt tokens -- the point of this method
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        prompt_lengths = self._prompt_lengths(prompts, images)
        seq_len = input_ids.shape[1]

        for i, prompt_len in enumerate(prompt_lengths):
            if padding_side == "left":
                # Right-aligned content: the prompt starts after the pad run.
                real_len = int(attention[i].sum()) if attention is not None else seq_len
                start = seq_len - real_len
                labels[i, : min(start + prompt_len, seq_len)] = IGNORE_INDEX
            else:
                labels[i, : min(prompt_len, seq_len)] = IGNORE_INDEX

        n_supervised = int((labels != IGNORE_INDEX).sum())
        if n_supervised == 0:
            logger.warning(
                "Batch has no supervised tokens; every label is masked. Check that "
                "answers are non-empty and max_length=%d is not truncating them.",
                self.max_length,
            )
        return labels


def build_datasets(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: Any,
    image_dir: Any = None,
    eval_loss_fraction: float | None = None,
) -> tuple["EntityExtractionDataset", "EntityExtractionDataset"]:
    """Construct the training dataset and the eval-loss dataset.

    ``eval_loss_fraction`` subsamples the eval set used for *loss* during
    training. Scoring still uses all 5,000 rows at the end; computing eval loss
    on the full set every epoch would add a large amount of GPU time for a
    number we only use to pick the best checkpoint.
    """
    prompt_cfg = cfg.get("prompt", {}) if hasattr(cfg, "get") else {}
    template = prompt_cfg.get("template", "prompt_v1")
    include_units = bool(prompt_cfg.get("include_allowed_units", True))

    eval_subset = eval_df
    if eval_loss_fraction and 0 < eval_loss_fraction < 1:
        n = max(1, int(len(eval_df) * eval_loss_fraction))
        eval_subset = eval_df.sample(n=n, random_state=int(cfg.get("seed", 42)))
        logger.info("Eval-loss subset: %d of %d rows", len(eval_subset), len(eval_df))

    train_ds = EntityExtractionDataset(
        train_df, image_dir, template, include_units, is_train=True
    )
    eval_ds = EntityExtractionDataset(
        eval_subset, image_dir, template, include_units, is_train=True
    )
    return train_ds, eval_ds


__all__ = ["EntityExtractionDataset", "QwenVLCollator", "build_datasets", "IGNORE_INDEX"]
