"""Parse and normalise ``"<number> <unit>"`` strings to competition format.

The metric is exact string match, so this module is where most of the score
lives. A model that reads ``34 g`` off the packaging is *correct about the
world* and *wrong on the leaderboard* unless we rewrite it to ``34 gram``.

Number formatting follows the convention the EDA format audit reports for the
training labels (see ``data/eda.py`` -> ``format_audit``): values are written
with no thousands separators, no trailing zeros beyond what is significant, and
integers carry no ``.0``. ``format_number`` implements exactly that, and
``run_eda`` prints the audit so the assumption is checkable rather than assumed.

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

RANGE_RULES = ("blank", "max", "min")

# A number: optional sign, digits with optional thousands separators, optional
# decimal part; also matches a bare ".5".
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\d*\.\d+)(?:\.\d+)?"
_UNIT = r"[A-Za-zµμ][A-Za-zµμ\.\s/]*"

_VALUE_RE = re.compile(rf"(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT})?", re.UNICODE)

# "10 to 20 gram", "10-20 gram", "10 ~ 20 gram", "between 10 and 20 gram"
_RANGE_RE = re.compile(
    rf"(?P<lo>{_NUMBER})\s*(?:to|-|–|—|~|and)\s*(?P<hi>{_NUMBER})\s*(?P<unit>{_UNIT})?",
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
    range_rule: str = "blank"
    infer_missing_unit: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping | None) -> "PostprocessOptions":
        """Build from the config's ``postprocess`` mapping, ignoring extra keys."""
        if cfg is None:
            return cls()
        raw = dict(cfg)
        rule = str(raw.get("range_rule", "blank")).lower()
        if rule not in RANGE_RULES:
            logger.warning("Unknown range_rule %r; falling back to 'blank'", rule)
            rule = "blank"
        known = {f for f in cls.__dataclass_fields__ if f != "range_rule"}
        kwargs = {k: bool(v) for k, v in raw.items() if k in known}
        return cls(range_rule=rule, **kwargs)


@dataclass
class ParsedValue:
    """Outcome of parsing one generated string."""

    number: Decimal | None = None
    unit: str | None = None
    is_range: bool = False
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


def format_number(value: Decimal | float | int | str) -> str:
    """Render a number the way the ground-truth labels are written.

    * integers lose any ``.0``  -> ``2``
    * decimals keep significant digits only -> ``12.50`` becomes ``12.5``
    * no thousands separators, no exponent notation
    """
    dec = value if isinstance(value, Decimal) else _to_decimal(str(value))
    if dec is None:
        return ""
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

    range_match = _RANGE_RE.search(cleaned)
    if range_match and _to_decimal(range_match.group("lo")) is not None \
            and _to_decimal(range_match.group("hi")) is not None:
        lo = _to_decimal(range_match.group("lo"))
        hi = _to_decimal(range_match.group("hi"))
        parsed.is_range = True
        parsed.notes.append("range_detected")
        parsed.unit = _resolve_unit(range_match.group("unit"), entity_name, opts, parsed)
        if opts.range_rule == "max":
            parsed.number = max(lo, hi)
        elif opts.range_rule == "min":
            parsed.number = min(lo, hi)
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

    if parsed.number is None or not parsed.unit:
        return "", notes + ["blanked_incomplete"]

    if opts.reject_invalid_units and entity_name:
        permitted = allowed_units(entity_name)
        if permitted and parsed.unit not in permitted:
            return "", notes + ["blanked_invalid_unit_for_entity"]

    number = format_number(parsed.number) if opts.normalize_numbers else str(parsed.number)
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
