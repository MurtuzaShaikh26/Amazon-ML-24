"""Class weights for the imbalanced `entity_name` distribution.

The EDA over all 263,859 training rows found a **31.5x** imbalance:

=============================  =======  =======
entity_name                    rows     share
=============================  =======  =======
item_weight                    102,786  38.95%
depth                           45,127  17.10%
width                           44,183  16.74%
height                          43,597  16.52%
voltage                          9,466   3.59%
wattage                          7,755   2.94%
item_volume                      7,682   2.91%
maximum_weight_recommendation    3,263   1.24%
=============================  =======  =======

Unweighted, ~72% of gradient signal comes from weight-and-dimension rows, and
the model under-fits `item_volume` (13 allowed units, the most of any entity)
and `maximum_weight_recommendation`. Since the competition metric is a *macro*
concern in practice -- we report per-entity F1 and care about all eight -- it is
worth trading a little overall accuracy for balance across classes.

Schemes
-------
``none``          every sample weight 1.0 (the unweighted baseline)
``inverse``       ``w_c  ∝ N / n_c``            -- fully balanced
``sqrt_inverse``  ``w_c  ∝ sqrt(N / n_c)``      -- **default**
``effective``     Cui et al. 2019 effective number of samples

``sqrt_inverse`` is the default because full inverse weighting hands
``maximum_weight_recommendation`` a 31x multiplier, and at batch size 1 with
8-step accumulation that makes gradient magnitude swing wildly between micro-
batches -- which with fp16 is a recipe for loss spikes and overflow. The square
root keeps the largest multiplier near 5.6x, which rebalances meaningfully
without destabilising the run.

Weights are always normalised to mean 1.0 over the *training distribution*, so
switching scheme changes the balance between classes but not the overall loss
scale -- and therefore does not implicitly change the effective learning rate.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)

WEIGHT_SCHEMES = ("none", "inverse", "sqrt_inverse", "effective")
DEFAULT_SCHEME = "sqrt_inverse"
DEFAULT_BETA = 0.999  # for the "effective" scheme


def compute_class_weights(
    entity_names: Sequence[str],
    scheme: str = DEFAULT_SCHEME,
    beta: float = DEFAULT_BETA,
    max_weight: float | None = 10.0,
) -> dict[str, float]:
    """Map ``entity_name`` -> loss multiplier, normalised to mean 1.0.

    ``max_weight`` caps any single class's multiplier before normalisation, so
    a rare class in a small training subset cannot dominate the objective.
    """
    counts = Counter(str(e) for e in entity_names)
    if not counts:
        logger.warning("No entity names supplied; class weighting disabled")
        return {}

    total = sum(counts.values())
    n_classes = len(counts)

    if scheme == "none":
        return {name: 1.0 for name in counts}

    raw: dict[str, float] = {}
    for name, count in counts.items():
        if scheme == "inverse":
            raw[name] = total / (n_classes * count)
        elif scheme == "sqrt_inverse":
            raw[name] = (total / (n_classes * count)) ** 0.5
        elif scheme == "effective":
            # Cui et al., "Class-Balanced Loss Based on Effective Number of
            # Samples": w ∝ (1 - beta) / (1 - beta^n).
            effective = (1.0 - beta ** count) / (1.0 - beta)
            raw[name] = 1.0 / effective
        else:
            raise ValueError(
                f"Unknown class_weight scheme {scheme!r}; expected one of {WEIGHT_SCHEMES}"
            )

    if max_weight is not None:
        floor = min(raw.values())
        raw = {k: min(v, floor * max_weight) for k, v in raw.items()}

    # Normalise so the *expected* weight over the training distribution is 1.0.
    mean_weight = sum(raw[name] * counts[name] for name in counts) / total
    weights = {name: value / mean_weight for name, value in raw.items()}

    logger.info("Class weights (scheme=%s, normalised to mean 1.0):", scheme)
    for name, count in counts.most_common():
        logger.info(
            "  %-32s n=%7d (%5.2f%%)  weight=%.3f",
            name, count, 100 * count / total, weights[name],
        )
    spread = max(weights.values()) / min(weights.values())
    logger.info("  weight spread (max/min) = %.2fx", spread)
    return weights


def weights_from_config(
    entity_names: Sequence[str], cfg: Mapping | None
) -> dict[str, float]:
    """Build class weights from the ``train.class_weights`` config block.

    Returns ``{}`` when weighting is disabled, which callers treat as "use the
    model's own unweighted loss" -- the cheap path.
    """
    block = dict(cfg or {})
    if not block.get("enabled", False):
        logger.info("Class weighting disabled; using unweighted loss")
        return {}

    scheme = str(block.get("scheme", DEFAULT_SCHEME)).lower()
    if scheme not in WEIGHT_SCHEMES:
        logger.warning("Unknown class weight scheme %r; using %r", scheme, DEFAULT_SCHEME)
        scheme = DEFAULT_SCHEME

    if scheme == "none":
        return {}

    return compute_class_weights(
        entity_names,
        scheme=scheme,
        beta=float(block.get("beta", DEFAULT_BETA)),
        max_weight=block.get("max_weight", 10.0),
    )


def describe_weights(weights: Mapping[str, float]) -> list[dict]:
    """JSON-serialisable summary, stored in the run's ``metrics.json``."""
    return [
        {"entity_name": name, "weight": round(float(value), 4)}
        for name, value in sorted(weights.items(), key=lambda kv: -kv[1])
    ]


__all__ = [
    "compute_class_weights", "weights_from_config", "describe_weights",
    "WEIGHT_SCHEMES", "DEFAULT_SCHEME",
]
