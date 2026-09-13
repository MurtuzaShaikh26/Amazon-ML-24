"""Amazon ML Challenge 2024 -- entity value extraction from product images.

Predict ``"<number> <unit>"`` for a given (image, entity_name) pair. Scored by
F1 with **exact string match**, which makes normalisation as important as the
model.

Layout::

    config/paths/seed/logging_utils   infrastructure
    data/                             load, EDA, frozen splits, images, dataset
    prompts/                          versioned prompt templates
    models/                           Qwen2-VL loading + quantisation + LoRA
    train/                            Trainer wiring, completion-only loss
    inference/                        batched greedy generation
    postprocess/                      unit vocabulary and normalisation rules
    metrics/                          competition F1 and its breakdowns
    results/                          per-run artefacts and the leaderboard
    pipeline/                         run_eda, run_finetune

Heavy dependencies (torch, transformers) are imported lazily inside the modules
that need them, so ``import amlc24`` works on a CPU-only box with no ML stack
installed -- which is what the tests and the EDA notebook rely on.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, config_hash, load_config
from .logging_utils import setup_logging
from .paths import DATA_DIR, IMAGE_DIR, REPO_ROOT, RESULTS_DIR
from .seed import set_seed

__all__ = [
    "__version__",
    "Config", "load_config", "config_hash",
    "setup_logging", "set_seed",
    "REPO_ROOT", "DATA_DIR", "RESULTS_DIR", "IMAGE_DIR",
]
