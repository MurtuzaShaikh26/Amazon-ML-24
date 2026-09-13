"""Global seeding for reproducible runs."""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_torch: bool = False) -> int:
    """Seed ``random``, ``numpy``, and (if installed) ``torch``.

    ``deterministic_torch`` additionally forces cuDNN into deterministic mode.
    It is off by default because it materially slows convolutions and the
    sampling we actually need to be reproducible (the data split) happens in
    numpy, not on the GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        logger.warning("numpy not installed; skipping numpy seeding")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")

    logger.info("Seeded RNGs with %d (deterministic_torch=%s)", seed, deterministic_torch)
    return seed


__all__ = ["set_seed"]
