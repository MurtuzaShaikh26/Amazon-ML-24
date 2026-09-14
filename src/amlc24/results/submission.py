"""Write and validate the competition submission file.

The rules, from the official problem statement:

* Two columns exactly: ``index`` and ``prediction``.
* ``prediction`` is ``"x unit"`` -- a float in standard formatting, one space,
  then a unit from ``constants.py`` for that entity.
* Valid: ``"2 gram"``, ``"12.5 centimetre"``, ``"2.56 ounce"``.
  Invalid: ``"2 gms"``, ``"60 ounce/1.7 kilogram"``, ``"2.2e2 kilogram"``.
* Empty string when no value is found.
* **One row per test index, no more and no fewer.** The statement is explicit
  that a file with a different row count "won't be evaluated", and the shipped
  ``sanity.py`` does *not* check this -- so we check it here.

``validate_submission`` mirrors the dataset's ``src/sanity.py`` so a malformed
file is caught locally, before a submission is wasted on it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..postprocess.units import allowed_units, all_allowed_units

logger = logging.getLogger(__name__)

SUBMISSION_COLUMNS = ["index", "prediction"]

# Mirrors the official sanity checker: a plain decimal, one space, one unit.
# Exponent notation is deliberately excluded -- "2.2e2 kilogram" is invalid.
_PREDICTION_RE = re.compile(r"^-?\d+(\.\d+)?\s+[a-zA-Z ]+$")


class SubmissionError(ValueError):
    """Raised when a submission would be rejected by the official checker."""


def _norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def validate_prediction(prediction: Any, entity_name: str | None = None) -> list[str]:
    """Return a list of problems with one prediction. Empty list means valid."""
    text = _norm(prediction)
    if text == "":
        return []  # abstaining is always valid

    problems: list[str] = []
    if "e" in text.lower() and _PREDICTION_RE.match(text) is None:
        problems.append(f"exponent notation is invalid: {text!r}")
    if not _PREDICTION_RE.match(text):
        problems.append(f"must be '<number> <unit>': {text!r}")
        return problems

    parts = text.split(maxsplit=1)
    unit = parts[1].strip()

    if unit not in all_allowed_units():
        problems.append(f"unit {unit!r} is not in constants.py")
    elif entity_name:
        permitted = allowed_units(entity_name)
        if permitted and unit not in permitted:
            problems.append(
                f"unit {unit!r} is not allowed for entity {entity_name!r} "
                f"(allowed: {sorted(permitted)})"
            )
    return problems


def validate_submission(
    df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    entity_names: dict[Any, str] | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Check a submission frame against the competition's formatting rules.

    ``test_df`` enables the row-count and index-coverage checks that the
    official ``sanity.py`` explicitly does not perform.
    """
    problems: list[str] = []

    if list(df.columns) != SUBMISSION_COLUMNS:
        problems.append(
            f"columns must be exactly {SUBMISSION_COLUMNS}, got {list(df.columns)}"
        )

    if "index" in df.columns:
        if df["index"].isna().any():
            problems.append("`index` contains nulls")
        if df["index"].duplicated().any():
            n = int(df["index"].duplicated().sum())
            problems.append(f"`index` has {n} duplicate value(s)")

    lookup = entity_names or {}
    if test_df is not None and "entity_name" in test_df.columns:
        lookup = dict(zip(test_df["index"], test_df["entity_name"]))

    bad_rows: list[dict[str, Any]] = []
    if "prediction" in df.columns:
        for idx, pred in zip(df.get("index", []), df["prediction"]):
            issues = validate_prediction(pred, lookup.get(idx))
            if issues:
                bad_rows.append({"index": idx, "prediction": pred,
                                 "problems": "; ".join(issues)})

    if bad_rows:
        problems.append(f"{len(bad_rows)} row(s) have invalid predictions")

    if test_df is not None:
        missing = set(test_df["index"]) - set(df.get("index", []))
        extra = set(df.get("index", [])) - set(test_df["index"])
        if len(df) != len(test_df):
            problems.append(
                f"row count {len(df)} != test.csv row count {len(test_df)}; "
                "the submission will not be evaluated"
            )
        if missing:
            problems.append(f"{len(missing)} test index/indices missing from the submission")
        if extra:
            problems.append(f"{len(extra)} index/indices not present in test.csv")

    n_empty = int((df["prediction"].map(_norm) == "").sum()) if "prediction" in df else 0
    report = {
        "valid": not problems,
        "n_rows": len(df),
        "n_empty": n_empty,
        "n_non_empty": len(df) - n_empty,
        "n_invalid": len(bad_rows),
        "problems": problems,
        "bad_rows": pd.DataFrame(bad_rows).head(50) if bad_rows else pd.DataFrame(),
    }

    if problems:
        logger.error("Submission validation FAILED:\n  - %s", "\n  - ".join(problems))
        if bad_rows:
            logger.error("First invalid rows:\n%s",
                         report["bad_rows"].head(10).to_string(index=False))
        if raise_on_error:
            raise SubmissionError("; ".join(problems))
    else:
        logger.info(
            "Submission valid: %d rows (%d predictions, %d empty)",
            len(df), report["n_non_empty"], n_empty,
        )
    return report


def write_submission(
    predictions: pd.DataFrame,
    path: str | Path,
    test_df: pd.DataFrame | None = None,
    validate: bool = True,
) -> Path:
    """Write ``index,prediction`` to ``path`` after validating it.

    ``predictions`` needs an ``index`` column plus either ``prediction`` or
    ``y_pred_post``. Any test index absent from the frame is filled with an
    empty prediction, because a short file is not evaluated at all -- an empty
    row merely costs one false negative.
    """
    df = predictions.copy()
    if "prediction" not in df.columns:
        for candidate in ("y_pred_post", "y_pred_raw"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "prediction"})
                break
    if "prediction" not in df.columns:
        raise SubmissionError(
            "No prediction column found (looked for prediction / y_pred_post / y_pred_raw)"
        )

    df = df[SUBMISSION_COLUMNS].copy()
    df["prediction"] = df["prediction"].map(_norm)

    if test_df is not None:
        missing = set(test_df["index"]) - set(df["index"])
        if missing:
            logger.warning(
                "Filling %d missing test index/indices with an empty prediction "
                "so the row count matches test.csv", len(missing),
            )
            filler = pd.DataFrame({"index": sorted(missing), "prediction": ""})
            df = pd.concat([df, filler], ignore_index=True)
        order = {idx: i for i, idx in enumerate(test_df["index"])}
        df = df.sort_values("index", key=lambda s: s.map(order)).reset_index(drop=True)

    if validate:
        validate_submission(df, test_df=test_df)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote submission with %d rows to %s", len(df), out)
    return out


__all__ = [
    "write_submission", "validate_submission", "validate_prediction",
    "SubmissionError", "SUBMISSION_COLUMNS",
]
