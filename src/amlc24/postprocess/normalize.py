"""Parse and normalise ``"<number> <unit>"`` strings to competition format.

The metric is exact string match, so this module is where most of the score
lives. A model that reads ``34 g`` off the packaging is *correct about the
world* and *wrong on the leaderboard* unless we rewrite it to ``34 gram``.

Number formatting follows the convention the EDA format audit **measured** over
all 263,859 training labels (see ``data/eda.py`` -> ``format_audit``):

* 64.12% carry a trailing ``.0``  (``500.0 gram``)
* 27.89% are real decimals        (``3.53 ounce``)
*  7.99% are bare integers        (``50 gram``)

So 92.01% of labels are exactly ``str(float(x))``, and ``format_number``
defaults to that. This contradicted the initial assumption that labels were
written without ``.0`` -- had we shipped that guess, we would have emitted
``500 gram`` for a label of ``500.0 gram`` and scored a false positive on the
large majority of otherwise-correct predictions. Run ``run_eda`` to re-check
the audit whenever the dataset changes.

There are also **zero empty labels** in ``train.csv``. Since
``F1 = 2TP/(2TP+FP+FN)`` charges a false negative and a false positive
identically, an empty prediction is never better than a guess -- which is why
``range_rule`` defaults to ``bracket`` (reproducing the labels' own
``"[100.0, 240.0] volt"`` notation) rather than blanking.

Every transformation is individually toggleable through ``PostprocessOptions``
so a later ablation can attribute score to each rule. ``apply_postprocess``
returns a counter of which rules fired, which is written into ``metrics.json``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping, Sequence

from .units import allowed_units, canonicalise_unit

logger = logging.getLogger(__name__)

RANGE_RULES = ("bracket", "blank", "max", "min")

# A number: optional sign, digits with optional thousands separators, optional
# decimal part; also matches a bare ".5".
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\d*\.\d+)(?:\.\d+)?"
_UNIT = r"[A-Za-zµμ][A-Za-zµμ\.\s/]*"

_VALUE_RE = re.compile(rf"(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT})?", re.UNICODE)

# The dataset's own range form, which is by far the most common one in the
# labels: "[100.0, 240.0] volt" (1.24% of training rows, 3,276 of them).
_BRACKET_RANGE_RE = re.compile(
    rf"\[\s*(?P<lo>{_NUMBER})\s*,\s*(?P<hi>{_NUMBER})\s*\]\s*(?P<unit>{_UNIT})?",
    re.UNICODE,
)

# "10 to 20 gram", "10-20 gram", "10 ~ 20 gram", "between 10 and 20 gram", and
# the dataset's "10 kilogram to 15 kilogram" (unit repeated after each number).
_RANGE_RE = re.compile(
    rf"(?P<lo>{_NUMBER})\s*(?P<lounit>{_UNIT})?\s*(?:to|-|–|—|~|and)\s*"
    rf"(?P<hi>{_NUMBER})\s*(?P<unit>{_UNIT})?",
    re.IGNORECASE | re.UNICODE,
)

# Chat models like to wrap answers; strip the usual scaffolding before parsing.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:the\s+)?(?:answer|value|result|output|prediction)\s*(?:is)?\s*[:\-]?\s*",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"```[a-zA-Z]*|```")


@dataclass(frozen=True)
class PostprocessOptions:
    """Per-rule switches, mirrored from the ``postprocess:`` config block."""

    enabled: bool = True
    strip_scaffolding: bool = True
    normalize_units: bool = True
    normalize_numbers: bool = True
    reject_invalid_units: bool = True
    range_rule: str = "bracket"
    infer_missing_unit: bool = True
    number_format: str = "float"

    @classmethod
    def from_config(cls, cfg: Mapping | None) -> "PostprocessOptions":
        """Build from the config's ``postprocess`` mapping, ignoring extra keys."""
        if cfg is None:
            return cls()
        raw = dict(cfg)

        rule = str(raw.get("range_rule", "bracket")).lower()
        if rule not in RANGE_RULES:
            logger.warning("Unknown range_rule %r; falling back to 'bracket'", rule)
            rule = "bracket"

        style = str(raw.get("number_format", "float")).lower()
        if style not in NUMBER_FORMATS:
            logger.warning("Unknown number_format %r; falling back to 'float'", style)
            style = "float"

        strings = {"range_rule", "number_format"}
        known = {f for f in cls.__dataclass_fields__ if f not in strings}
        kwargs = {k: bool(v) for k, v in raw.items() if k in known}
        return cls(range_rule=rule, number_format=style, **kwargs)


@dataclass
class ParsedValue:
    """Outcome of parsing one generated string."""

    number: Decimal | None = None
    unit: str | None = None
    is_range: bool = False
    range_lo: Decimal | None = None
    range_hi: Decimal | None = None
    raw: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.number is not None and bool(self.unit)


def _to_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "").strip())
    except (InvalidOperation, ValueError, ArithmeticError):
        return None


NUMBER_FORMATS = ("float", "int", "strip")


def format_number(value: Decimal | float | int | str, style: str = "float") -> str:
    """Render a number the way the ground-truth labels are actually written.

    The convention was **measured, not assumed** -- see ``data/eda.py``'s format
    audit, run over all 263,859 training labels:

    =========================  =======  ===================
    number shape               share    example
    =========================  =======  ===================
    ``d.0`` (trailing ``.0``)  64.12%   ``500.0 gram``
    real decimal               27.89%   ``3.53 ounce``
    bare integer                7.99%   ``50 gram``
    =========================  =======  ===================

    So **92.01% of labels are exactly ``str(float(x))``** -- Python float
    repr, which always keeps at least one decimal place. Six of the eight
    entities (depth, height, item_volume, voltage, wattage, width) are 100%
    float-style with not a single bare integer; only ``item_weight`` (18.9%
    bare) and ``maximum_weight_recommendation`` (49.9% bare) mix the two.

    ``style="float"`` is therefore the default and reproduces 92.01% of labels.
    The earlier ``"int"`` behaviour (strip ``.0``) matches only 7.99% and would
    turn a correct ``500.0 gram`` into ``500 gram`` -- a false positive on the
    large majority of otherwise-correct predictions.

    * ``float``  -- ``str(float(x))``: ``500`` -> ``500.0``, ``3.53`` -> ``3.53``
    * ``int``    -- drop a trailing ``.0``: ``500.0`` -> ``500``
    * ``strip``  -- trim trailing zeros but keep real decimals
    """
    dec = value if isinstance(value, Decimal) else _to_decimal(str(value))
    if dec is None:
        return ""

    if style == "float":
        try:
            # str(float(...)) is exactly the labels' convention, including its
            # exponent form for the handful of extreme values that use it.
            return str(float(dec))
        except (ValueError, OverflowError, ArithmeticError):
            return ""

    if style == "strip":
        normalised = dec.normalize()
        text = format(normalised, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    # style == "int"
    if dec == dec.to_integral_value():
        # normalize() would give 2E+3 for 2000; quantize to a plain integer.
        return str(dec.quantize(Decimal(1)))
    normalised = dec.normalize()
    text = format(normalised, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _clean(text: str) -> str:
    text = _FENCE_RE.sub(" ", str(text))
    text = text.replace(" ", " ")
    text = _PREAMBLE_RE.sub("", text.strip())
    return " ".join(text.split()).strip(" .;,:\"'")


def parse_value(
    text: str | None,
    entity_name: str | None = None,
    opts: PostprocessOptions | None = None,
) -> ParsedValue:
    """Extract a number and a canonical unit from a free-form model output.

    Ranges are detected first, because ``"10 to 20 gram"`` would otherwise parse
    as the single value ``10``, silently inventing a confident wrong answer.
    """
    opts = opts or PostprocessOptions()
    if text is None:
        return ParsedValue(raw="", notes=["empty_input"])

    raw = str(text)
    cleaned = _clean(raw) if opts.strip_scaffolding else " ".join(raw.split())
    if not cleaned:
        return ParsedValue(raw=raw, notes=["empty_input"])

    parsed = ParsedValue(raw=raw)

    # Bracket form first -- it is the dataset's own notation and would otherwise
    # be mangled by the looser "X to Y" pattern.
    for pattern, label in ((_BRACKET_RANGE_RE, "bracket"), (_RANGE_RE, "textual")):
        match = pattern.search(cleaned)
        if not match:
            continue
        lo, hi = _to_decimal(match.group("lo")), _to_decimal(match.group("hi"))
        if lo is None or hi is None:
            continue

        parsed.is_range = True
        parsed.range_lo, parsed.range_hi = min(lo, hi), max(lo, hi)
        parsed.notes.extend(["range_detected", f"range_form_{label}"])
        # "10 kilogram to 15 kilogram" repeats the unit; either capture works.
        unit_text = match.group("unit") or (
            match.groupdict().get("lounit") if label == "textual" else None
        )
        parsed.unit = _resolve_unit(unit_text, entity_name, opts, parsed)

        if opts.range_rule == "max":
            parsed.number = max(lo, hi)
        elif opts.range_rule == "min":
            parsed.number = min(lo, hi)
        elif opts.range_rule == "bracket":
            # Emitted by normalize_prediction as the dataset's own notation.
            parsed.number = None
        else:
            parsed.number = None
            parsed.notes.append("range_blanked")
        return parsed

    match = _VALUE_RE.search(cleaned)
    if not match:
        parsed.notes.append("no_number_found")
        return parsed

    parsed.number = _to_decimal(match.group("number"))
    if parsed.number is None:
        parsed.notes.append("unparseable_number")
        return parsed

    unit_text = match.group("unit")
    if not unit_text:
        # The unit may follow the number after other tokens, or precede it
        # ("gram 34"); fall back to scanning the whole string for a known unit.
        unit_text = _scan_for_unit(cleaned)
        if unit_text:
            parsed.notes.append("unit_recovered_by_scan")
    parsed.unit = _resolve_unit(unit_text, entity_name, opts, parsed)
    return parsed


def _scan_for_unit(text: str) -> str | None:
    """Find the longest token sequence in ``text`` that names a known unit."""
    tokens = re.findall(r"[A-Za-zµμ]+", text)
    for width in (3, 2, 1):  # "imperial gallon" is two words; allow three.
        for i in range(len(tokens) - width + 1):
            candidate = " ".join(tokens[i:i + width])
            if canonicalise_unit(candidate):
                return candidate
    return None


def _resolve_unit(
    unit_text: str | None,
    entity_name: str | None,
    opts: PostprocessOptions,
    parsed: ParsedValue,
) -> str | None:
    """Canonicalise a unit string and record why it failed if it did."""
    if not unit_text:
        parsed.notes.append("no_unit_found")
        return None

    if opts.normalize_units:
        canonical = canonicalise_unit(unit_text)
    else:
        stripped = " ".join(str(unit_text).strip().lower().split())
        canonical = stripped or None

    if canonical is None:
        parsed.notes.append("unknown_unit")
        return None
    if canonical != " ".join(str(unit_text).strip().lower().split()):
        parsed.notes.append("unit_normalised")
    return canonical


def normalize_prediction(
    text: str | None,
    entity_name: str | None = None,
    opts: PostprocessOptions | None = None,
) -> tuple[str, list[str]]:
    """Full pipeline for one prediction. Returns ``(prediction, notes)``.

    The prediction is ``""`` whenever no confident, valid value could be
    produced -- an empty answer is a false negative, whereas a wrong non-empty
    answer is a false positive *and* costs precision, so blanking is the
    conservative choice.
    """
    opts = opts or PostprocessOptions()
    if not opts.enabled:
        return ("" if text is None else str(text).strip()), ["postprocess_disabled"]

    parsed = parse_value(text, entity_name, opts)
    notes = parsed.notes

    # Reproduce the dataset's own range notation, "[100.0, 240.0] volt".
    # There are no empty labels in train.csv, so an empty prediction is a
    # guaranteed false negative; emitting the bracket form can at least score a
    # true positive when the label really is a range (1.24% of rows).
    if parsed.is_range and opts.range_rule == "bracket" and parsed.unit:
        if opts.reject_invalid_units and entity_name:
            permitted = allowed_units(entity_name)
            if permitted and parsed.unit not in permitted:
                return "", notes + ["blanked_invalid_unit_for_entity"]
        lo = format_number(parsed.range_lo, opts.number_format)
        hi = format_number(parsed.range_hi, opts.number_format)
        if lo and hi:
            return f"[{lo}, {hi}] {parsed.unit}", notes + ["range_emitted_as_bracket"]

    if parsed.number is None or not parsed.unit:
        return "", notes + ["blanked_incomplete"]

    if opts.reject_invalid_units and entity_name:
        permitted = allowed_units(entity_name)
        if permitted and parsed.unit not in permitted:
            return "", notes + ["blanked_invalid_unit_for_entity"]

    number = (
        format_number(parsed.number, opts.number_format)
        if opts.normalize_numbers else str(parsed.number)
    )
    if not number:
        return "", notes + ["blanked_unformattable_number"]
    return f"{number} {parsed.unit}", notes


def apply_postprocess(
    raw_predictions: Sequence[str | None],
    entity_names: Sequence[str] | None = None,
    opts: PostprocessOptions | None = None,
) -> tuple[list[str], Counter]:
    """Vectorised ``normalize_prediction`` over a run's outputs.

    Returns the cleaned predictions and a counter of how often each rule fired,
    which is logged and stored so the effect of post-processing is auditable.
    """
    opts = opts or PostprocessOptions()
    entities: Iterable[str | None] = entity_names if entity_names is not None \
        else [None] * len(raw_predictions)

    cleaned: list[str] = []
    counter: Counter = Counter()
    for raw, entity in zip(raw_predictions, entities):
        pred, notes = normalize_prediction(raw, entity, opts)
        cleaned.append(pred)
        counter.update(notes)
        counter["total"] += 1
        if pred:
            counter["non_empty_out"] += 1
        if raw is not None and str(raw).strip() and not pred:
            counter["blanked_from_non_empty"] += 1

    logger.info(
        "Post-processing: %d rows, %d non-empty out, %d blanked from non-empty. "
        "Rules fired: %s",
        counter["total"], counter["non_empty_out"], counter["blanked_from_non_empty"],
        dict(counter.most_common(12)),
    )
    if counter.get("range_detected"):
        logger.info(
            "Range rule %r fired on %d prediction(s)",
            opts.range_rule, counter["range_detected"],
        )
    return cleaned, counter


def normalize_ground_truth(value: str | None) -> str:
    """Canonicalise a *label* for comparison and for the EDA audit.

    Applied to ground truth only when explicitly requested; the metric itself
    always compares against the untouched label, since that is what the
    leaderboard does.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return " ".join(str(value).strip().split())


__all__ = [
    "PostprocessOptions", "ParsedValue", "RANGE_RULES", "parse_value",
    "normalize_prediction", "apply_postprocess", "format_number",
    "normalize_ground_truth",
]
