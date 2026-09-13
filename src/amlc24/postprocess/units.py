"""Allowed units per entity, plus the variant -> canonical unit vocabulary.

The competition ships a ``src/constants.py`` inside the dataset archive defining
``entity_unit_map``. We parse that file when it is present so the repo tracks the
official list rather than a copy that can silently drift. When it is absent
(tests, a fresh clone with no data, a Kaggle session where only the images are
mounted) we fall back to ``FALLBACK_ENTITY_UNIT_MAP`` below, which is the 2024
list transcribed verbatim.

Parsing is done with ``ast.literal_eval`` on the assignment node, never
``exec``/``import``, so a tampered dataset file cannot run code in our process.

A prediction whose unit is not in the allowed set for its entity is *invalid*
and scores as a false positive, so ``reject_invalid_units`` in post-processing
blanks it instead -- an empty prediction is only a false negative, which costs
recall but not precision.
"""

from __future__ import annotations

import ast
import logging
from functools import lru_cache
from pathlib import Path

from ..paths import DATA_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Official 2024 entity -> allowed-unit map (fallback copy).
# ---------------------------------------------------------------------------
FALLBACK_ENTITY_UNIT_MAP: dict[str, set[str]] = {
    "width": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "depth": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "height": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "item_weight": {
        "gram", "kilogram", "microgram", "milligram", "ounce", "pound", "ton",
    },
    "maximum_weight_recommendation": {
        "gram", "kilogram", "microgram", "milligram", "ounce", "pound", "ton",
    },
    "voltage": {"kilovolt", "millivolt", "volt"},
    "wattage": {"kilowatt", "watt"},
    "item_volume": {
        "centilitre", "cubic foot", "cubic inch", "cup", "decilitre",
        "fluid ounce", "gallon", "imperial gallon", "litre", "microlitre",
        "millilitre", "pint", "quart",
    },
}

# Locations to search for the dataset's own constants.py, in priority order.
_CONSTANTS_CANDIDATES = (
    DATA_DIR / "src" / "constants.py",
    DATA_DIR / "constants.py",
    DATA_DIR.parent / "src" / "constants.py",
    REPO_ROOT / "data" / "src" / "constants.py",
)


def _parse_constants_file(path: Path) -> dict[str, set[str]] | None:
    """Extract ``entity_unit_map`` from a constants.py without executing it."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "entity_unit_map" not in names:
            continue
        try:
            raw = ast.literal_eval(node.value)
        except ValueError as exc:
            logger.warning("entity_unit_map in %s is not a literal: %s", path, exc)
            return None
        return {str(k): {str(u) for u in v} for k, v in dict(raw).items()}
    return None


@lru_cache(maxsize=1)
def entity_unit_map() -> dict[str, set[str]]:
    """Entity -> set of allowed unit strings, from the dataset when available."""
    for path in _CONSTANTS_CANDIDATES:
        if not path.exists():
            continue
        parsed = _parse_constants_file(path)
        if parsed:
            logger.info("Loaded entity_unit_map from %s (%d entities)", path, len(parsed))
            _warn_on_drift(parsed)
            return parsed

    logger.info(
        "No dataset constants.py found; using the built-in 2024 entity_unit_map "
        "(%d entities).", len(FALLBACK_ENTITY_UNIT_MAP)
    )
    return {k: set(v) for k, v in FALLBACK_ENTITY_UNIT_MAP.items()}


def _warn_on_drift(parsed: dict[str, set[str]]) -> None:
    """Loudly flag any divergence between the dataset file and our fallback."""
    for entity in set(parsed) | set(FALLBACK_ENTITY_UNIT_MAP):
        ours = FALLBACK_ENTITY_UNIT_MAP.get(entity, set())
        theirs = parsed.get(entity, set())
        if ours != theirs:
            logger.warning(
                "entity_unit_map drift for %r: dataset-only=%s fallback-only=%s",
                entity, sorted(theirs - ours), sorted(ours - theirs),
            )


def allowed_units(entity_name: str) -> set[str]:
    """Allowed units for one entity; empty set for an unknown entity."""
    return entity_unit_map().get(entity_name, set())


def all_allowed_units() -> set[str]:
    """Union of every allowed unit across all entities."""
    return {u for units in entity_unit_map().values() for u in units}


def is_valid_unit(entity_name: str, unit: str) -> bool:
    """True when ``unit`` is permitted for ``entity_name``."""
    return unit in allowed_units(entity_name)


# ---------------------------------------------------------------------------
# Variant vocabulary: what a model might emit -> the canonical competition unit.
#
# Keys are matched case-insensitively against the whitespace-normalised unit
# token. Every canonical unit also maps to itself (added programmatically) so
# lookup is a single dict hit. Plural forms are generated automatically, so only
# genuinely irregular variants need listing here.
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, tuple[str, ...]] = {
    # length
    "centimetre": ("cm", "cms", "centimeter", "centi metre", "centi meter", "c.m.", "cm.", "centimetres", "centimeters"),
    "millimetre": ("mm", "mms", "millimeter", "milli metre", "milli meter", "m.m.", "mm.", "millimetres", "millimeters"),
    "metre": ("m", "mtr", "mtrs", "meter", "mts", "metres", "meters"),
    "foot": ("ft", "feet", "foot", "'", "ft.", "fts"),
    "inch": ("in", "inches", "inch", '"', "in.", "ins", "''"),
    "yard": ("yd", "yds", "yards", "yard"),
    # mass
    "gram": ("g", "gm", "gms", "grams", "gr", "grm", "g.", "gramme", "grammes"),
    "kilogram": ("kg", "kgs", "kilograms", "kilo", "kilos", "kgm", "kg.", "kilogramme", "kilogrammes"),
    "milligram": ("mg", "mgs", "milligrams", "mg.", "milligramme", "milligrammes"),
    "microgram": ("mcg", "ug", "µg", "micrograms", "mcgs", "microgramme", "microgrammes"),
    # "fl oz" deliberately absent here: it belongs to "fluid ounce".
    "ounce": ("oz", "ozs", "ounces", "oz."),
    "pound": ("lb", "lbs", "pounds", "lb.", "#"),
    "ton": ("tons", "tonne", "tonnes", "t", "mt", "metric ton"),
    # electrical
    "volt": ("v", "volts", "v.", "vdc", "vac"),
    "kilovolt": ("kv", "kilovolts", "kvs"),
    "millivolt": ("mv", "millivolts", "mvs"),
    "watt": ("w", "watts", "w.", "wattage"),
    "kilowatt": ("kw", "kilowatts", "kws"),
    # volume
    "litre": ("l", "liter", "liters", "ltr", "ltrs", "lt", "litres", "l."),
    "millilitre": ("ml", "milliliter", "milliliters", "mls", "millilitres", "ml.", "cc"),
    "centilitre": ("cl", "centiliter", "centiliters", "centilitres"),
    "decilitre": ("dl", "deciliter", "deciliters", "decilitres"),
    "microlitre": ("ul", "µl", "microliter", "microliters", "mcl", "microlitres"),
    "cubic foot": ("cu ft", "cuft", "ft3", "ft^3", "cubic feet", "cu. ft.", "cubic ft"),
    "cubic inch": ("cu in", "cuin", "in3", "in^3", "cubic inches", "cu. in.", "cubic in"),
    "cup": ("cups", "cup"),
    "fluid ounce": ("fl oz", "floz", "fl. oz.", "fluid ounces", "fl ounce", "fluid oz", "oz fl"),
    "gallon": ("gal", "gals", "gallons", "us gallon", "us gallons"),
    "imperial gallon": ("imp gallon", "imperial gallons", "imp gal", "uk gallon", "uk gallons"),
    "pint": ("pt", "pts", "pints"),
    "quart": ("qt", "qts", "quarts"),
}


@lru_cache(maxsize=1)
def unit_alias_table() -> dict[str, str]:
    """Lowercased variant -> canonical unit, including identity mappings.

    Built once. Canonical names and their auto-pluralised forms are registered
    first; explicit variants are registered after and are allowed to overwrite
    only entries that do not already point at a different canonical unit, so an
    ambiguous alias never silently steals another unit's mapping.
    """
    table: dict[str, str] = {}
    canonical = all_allowed_units() | set(_VARIANTS)

    for unit in canonical:
        table[unit.lower()] = unit
        if not unit.endswith("s"):
            table.setdefault(f"{unit}s".lower(), unit)

    for unit, variants in _VARIANTS.items():
        if unit not in canonical:
            continue
        for variant in variants:
            key = variant.strip().lower()
            existing = table.get(key)
            if existing is not None and existing != unit:
                # e.g. "oz" must not be captured by "fluid ounce".
                logger.debug("Alias %r already maps to %r; not remapping to %r",
                             key, existing, unit)
                continue
            table[key] = unit

    # "t" is ambiguous (ton vs nothing else here) but harmless; "m" must stay
    # metre rather than being pulled to millimetre by pluralisation.
    table["m"] = "metre"
    table["in"] = "inch"
    return table


def canonicalise_unit(raw: str | None) -> str | None:
    """Map an arbitrary unit spelling to its canonical form, or ``None``.

    Punctuation and internal whitespace are normalised before lookup, so
    ``"Fl. Oz."`` and ``"fl oz"`` both resolve to ``"fluid ounce"``.
    """
    if not raw:
        return None
    key = " ".join(str(raw).strip().lower().replace(".", " ").split())
    table = unit_alias_table()
    if key in table:
        return table[key]
    # Retry without internal spaces: "kilo gram" -> "kilogram".
    squashed = key.replace(" ", "")
    return table.get(squashed)


__all__ = [
    "entity_unit_map", "allowed_units", "all_allowed_units", "is_valid_unit",
    "canonicalise_unit", "unit_alias_table", "FALLBACK_ENTITY_UNIT_MAP",
]
