"""The competition F1 metric, implemented exactly as specified.

Classification of a single (truth, prediction) pair:

===================  ==================  ========
prediction           ground truth        outcome
===================  ==================  ========
non-empty, equal     non-empty           TP
non-empty, differs   non-empty           FP
non-empty            empty               FP
empty                non-empty           FN
empty                empty               TN
===================  ==================  ========

``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)``,
``F1 = 2PR / (P + R)``. True negatives are counted and reported but, as in the
standard formulation, do not enter F1 -- a run that predicts nothing at all
scores 0, not 1.

Comparison is exact string equality after stripping surrounding whitespace only.
No case-folding, no unit rewriting: any such leniency would make the local score
optimistic relative to the leaderboard, which is the one failure mode a local
metric must never have.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

OUTCOMES = ("TP", "FP", "FN", "TN")


def _norm(value: Any) -> str:
    """Empty-normalise one cell: NaN/None/whitespace all collapse to ``""``."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def classify(truth: Any, pred: Any) -> str:
    """Return the outcome label for one pair. See the module table."""
    t, p = _norm(truth), _norm(pred)
    if p and t:
        return "TP" if p == t else "FP"
    if p and not t:
        return "FP"
    if not p and t:
        return "FN"
    return "TN"


def classify_all(y_true: Sequence[Any], y_pred: Sequence[Any]) -> list[str]:
    """Outcome label per row. Raises if the sequences differ in length."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}"
        )
    return [classify(t, p) for t, p in zip(y_true, y_pred)]


def _scores_from_counts(counts: Mapping[str, int]) -> dict[str, float | int]:
    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    tn = int(counts.get("TN", 0))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = tp + fp + fn + tn

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "n_pred_non_empty": tp + fp,
        "n_true_non_empty": tp + fn,
    }


def f1_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float | int]:
    """Overall competition F1 with precision, recall, and raw TP/FP/FN/TN."""
    counts = Counter(classify_all(y_true, y_pred))
    scores = _scores_from_counts(counts)
    logger.info(
        "F1=%.4f  P=%.4f  R=%.4f  (TP=%d FP=%d FN=%d TN=%d, n=%d)",
        scores["f1"], scores["precision"], scores["recall"],
        scores["tp"], scores["fp"], scores["fn"], scores["tn"], scores["n"],
    )
    return scores


def _grouped_frame(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Sequence[Any],
    group_col: str,
) -> pd.DataFrame:
    """Per-group metric table, sorted by support descending, with a TOTAL row."""
    df = pd.DataFrame({
        group_col: [_norm(g) or "(unknown)" for g in groups],
        "outcome": classify_all(y_true, y_pred),
    })

    rows = []
    for name, chunk in df.groupby(group_col, sort=False):
        scores = _scores_from_counts(Counter(chunk["outcome"]))
        rows.append({group_col: name, **scores})

    out = pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)
    total = _scores_from_counts(Counter(df["outcome"]))
    out = pd.concat(
        [out, pd.DataFrame([{group_col: "TOTAL", **total}])], ignore_index=True
    )
    ordered = [group_col, "n", "f1", "precision", "recall",
               "tp", "fp", "fn", "tn", "n_true_non_empty", "n_pred_non_empty", "accuracy"]
    return out[[c for c in ordered if c in out.columns]]


def f1_by_entity(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    entity_names: Sequence[Any],
) -> pd.DataFrame:
    """Class-wise F1 broken down by ``entity_name``.

    This is the headline diagnostic: entities differ enormously in how legible
    their values are in a product photo, and an overall score hides that.
    """
    table = _grouped_frame(y_true, y_pred, entity_names, "entity_name")
    logger.info("Per-entity F1:\n%s", table.to_string(index=False, float_format="%.4f"))
    return table


def extract_unit(value: Any) -> str:
    """Trailing unit of a ``"<number> <unit>"`` label, or ``""`` when empty."""
    text = _norm(value)
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else "(no unit)"


def f1_by_unit(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    entity_names: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """F1 grouped by the *ground-truth* unit.

    Grouping by truth rather than prediction isolates unit-formatting failures
    (the model saw the right number but wrote ``g`` for ``gram``) from value
    failures. Rows whose truth is empty are grouped under ``(empty)``.
    """
    units = [extract_unit(t) or "(empty)" for t in y_true]
    table = _grouped_frame(y_true, y_pred, units, "unit")
    if entity_names is not None:
        entity_of_unit = (
            pd.DataFrame({"unit": units, "entity_name": [_norm(e) for e in entity_names]})
            .groupby("unit")["entity_name"]
            .agg(lambda s: ", ".join(sorted(set(s))[:3]))
        )
        table["entities"] = table["unit"].map(entity_of_unit).fillna("")
    logger.info("Per-unit F1 computed over %d distinct units", table["unit"].nunique() - 1)
    return table


def error_analysis(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    entity_names: Sequence[Any],
    top_k: int = 20,
) -> pd.DataFrame:
    """The ``top_k`` most frequent (predicted, actual) mismatch pairs per entity.

    This is the table that generates post-processing rules: a repeated
    ``("34 g", "34 gram")`` pair is a missing alias, whereas scattered one-off
    numeric errors are a model-capacity problem instead.
    """
    df = pd.DataFrame({
        "entity_name": [_norm(e) or "(unknown)" for e in entity_names],
        "y_true": [_norm(t) for t in y_true],
        "y_pred": [_norm(p) for p in y_pred],
    })
    df["outcome"] = classify_all(df["y_true"], df["y_pred"])
    mismatches = df[df["outcome"].isin(["FP", "FN"])]

    if mismatches.empty:
        logger.info("Error analysis: no mismatches found")
        return pd.DataFrame(
            columns=["entity_name", "y_pred", "y_true", "count", "outcome",
                     "share_of_entity_errors", "same_number_diff_unit"]
        )

    grouped = (
        mismatches.groupby(["entity_name", "y_pred", "y_true", "outcome"])
        .size().reset_index(name="count")
    )
    totals = grouped.groupby("entity_name")["count"].transform("sum")
    grouped["share_of_entity_errors"] = grouped["count"] / totals

    def _same_number(row: pd.Series) -> bool:
        """True when only the unit differs -- a pure formatting miss."""
        pn, tn = row["y_pred"].split(maxsplit=1), row["y_true"].split(maxsplit=1)
        return bool(pn and tn and pn[0] == tn[0] and row["y_pred"] != row["y_true"])

    grouped["same_number_diff_unit"] = grouped.apply(_same_number, axis=1)

    out = (
        grouped.sort_values(["entity_name", "count"], ascending=[True, False])
        .groupby("entity_name", group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    n_format_only = int(out["same_number_diff_unit"].sum())
    logger.info(
        "Error analysis: %d distinct mismatch pairs kept (top %d per entity); "
        "%d of them differ only in the unit -- recoverable by post-processing.",
        len(out), top_k, n_format_only,
    )
    return out[["entity_name", "y_pred", "y_true", "count", "outcome",
                "share_of_entity_errors", "same_number_diff_unit"]]


def evaluate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    entity_names: Sequence[Any],
    top_k: int = 20,
) -> dict[str, Any]:
    """Everything at once: overall scores plus the three breakdown tables."""
    return {
        "overall": f1_score(y_true, y_pred),
        "by_entity": f1_by_entity(y_true, y_pred, entity_names),
        "by_unit": f1_by_unit(y_true, y_pred, entity_names),
        "errors": error_analysis(y_true, y_pred, entity_names, top_k=top_k),
    }


__all__ = [
    "classify", "classify_all", "f1_score", "f1_by_entity", "f1_by_unit",
    "error_analysis", "evaluate", "extract_unit", "OUTCOMES",
]
