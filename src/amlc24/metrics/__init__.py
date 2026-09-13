"""Competition F1 with exact string match, plus per-entity and per-unit views."""

from .f1 import (
    classify,
    error_analysis,
    evaluate,
    f1_by_entity,
    f1_by_unit,
    f1_score,
)

__all__ = [
    "f1_score", "f1_by_entity", "f1_by_unit", "error_analysis", "evaluate",
    "classify",
]
