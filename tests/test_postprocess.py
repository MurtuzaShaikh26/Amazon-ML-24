"""Unit canonicalisation, number formatting, and normalisation round-trips."""

from __future__ import annotations

import pytest

from amlc24.postprocess.normalize import (
    PostprocessOptions,
    apply_postprocess,
    format_number,
    normalize_prediction,
    parse_value,
)
from amlc24.postprocess.units import (
    allowed_units,
    all_allowed_units,
    canonicalise_unit,
    entity_unit_map,
    is_valid_unit,
)


# --- allowed units ---------------------------------------------------------
def test_entity_unit_map_covers_the_eight_competition_entities():
    expected = {
        "width", "depth", "height", "item_weight",
        "maximum_weight_recommendation", "voltage", "wattage", "item_volume",
    }
    assert expected <= set(entity_unit_map())


def test_allowed_units_are_correct_for_known_entities():
    assert "gram" in allowed_units("item_weight")
    assert "kilogram" in allowed_units("item_weight")
    assert "centimetre" not in allowed_units("item_weight")
    assert "volt" in allowed_units("voltage")
    assert "fluid ounce" in allowed_units("item_volume")


def test_is_valid_unit_rejects_cross_entity_units():
    assert is_valid_unit("item_weight", "gram")
    assert not is_valid_unit("item_weight", "volt")
    assert not is_valid_unit("voltage", "gram")


def test_unknown_entity_has_no_allowed_units():
    assert allowed_units("not_a_real_entity") == set()


# --- unit canonicalisation round-trips -------------------------------------
@pytest.mark.parametrize("variant,expected", [
    ("g", "gram"), ("gm", "gram"), ("gms", "gram"), ("grams", "gram"),
    ("kg", "kilogram"), ("kgs", "kilogram"), ("KG", "kilogram"),
    ("mg", "milligram"), ("mcg", "microgram"),
    ("cm", "centimetre"), ("CM", "centimetre"), ("centimeter", "centimetre"),
    ("mm", "millimetre"), ("m", "metre"), ("meters", "metre"),
    ("in", "inch"), ("inches", "inch"), ("ft", "foot"), ("feet", "foot"),
    ("yd", "yard"),
    ("oz", "ounce"), ("ounces", "ounce"), ("lb", "pound"), ("lbs", "pound"),
    ("v", "volt"), ("volts", "volt"), ("kv", "kilovolt"), ("mv", "millivolt"),
    ("w", "watt"), ("kw", "kilowatt"),
    ("l", "litre"), ("liter", "litre"), ("ltr", "litre"),
    ("ml", "millilitre"), ("milliliters", "millilitre"),
    ("fl oz", "fluid ounce"), ("fl. oz.", "fluid ounce"), ("floz", "fluid ounce"),
    ("gal", "gallon"), ("pt", "pint"), ("qt", "quart"),
    ("cu ft", "cubic foot"), ("cubic feet", "cubic foot"),
])
def test_variants_canonicalise(variant, expected):
    assert canonicalise_unit(variant) == expected


def test_every_canonical_unit_maps_to_itself():
    """The round-trip property: canonicalise(u) == u for all allowed units."""
    for unit in all_allowed_units():
        assert canonicalise_unit(unit) == unit, f"{unit} is not a fixed point"


def test_canonicalise_is_idempotent():
    for variant in ("g", "cm", "fl oz", "lbs", "KG"):
        once = canonicalise_unit(variant)
        assert canonicalise_unit(once) == once


def test_fluid_ounce_does_not_steal_the_ounce_alias():
    assert canonicalise_unit("oz") == "ounce"
    assert canonicalise_unit("fl oz") == "fluid ounce"


def test_unknown_units_return_none():
    for junk in ("banana", "", None, "xyz"):
        assert canonicalise_unit(junk) is None


# --- number formatting -----------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("2", "2"), ("2.0", "2"), ("2.00", "2"), ("2.50", "2.5"), ("2.5", "2.5"),
    ("0.5", "0.5"), (".5", "0.5"), ("1000", "1000"), ("1,000", "1000"),
    ("12.345", "12.345"), ("0", "0"), ("2000", "2000"),
])
def test_format_number_matches_label_conventions(raw, expected):
    assert format_number(raw) == expected


def test_format_number_never_uses_exponent_notation():
    assert "E" not in format_number("1000000") and "e" not in format_number("1000000")


# --- parsing ---------------------------------------------------------------
def test_parse_simple_value():
    parsed = parse_value("34 gram", "item_weight")
    assert str(parsed.number) == "34" and parsed.unit == "gram"
    assert parsed.ok and not parsed.is_range


def test_parse_recovers_unit_from_variant():
    assert parse_value("34 g", "item_weight").unit == "gram"


def test_parse_strips_chat_scaffolding():
    for text in ("The answer is 34 gram", "Answer: 34 gram", "```\n34 gram\n```"):
        parsed = parse_value(text, "item_weight")
        assert str(parsed.number) == "34" and parsed.unit == "gram", text


def test_parse_detects_range_before_taking_the_first_number():
    parsed = parse_value("10 to 20 gram", "item_weight")
    assert parsed.is_range
    assert "range_detected" in parsed.notes


def test_parse_handles_missing_number():
    assert parse_value("gram", "item_weight").number is None


# --- full normalisation ----------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("34 gram", "34 gram"),
    ("34 g", "34 gram"),
    ("34 gms", "34 gram"),
    ("34.0 gram", "34 gram"),
    ("34.50 g", "34.5 gram"),
    ("1,000 g", "1000 gram"),
    ("The item weighs 34 grams", "34 gram"),
    ("34GRAM", "34 gram"),
])
def test_normalize_prediction_produces_exact_match_format(raw, expected):
    assert normalize_prediction(raw, "item_weight")[0] == expected


def test_normalize_blanks_units_invalid_for_the_entity():
    """A volt reading for item_weight is invalid; blanking costs recall only."""
    result, notes = normalize_prediction("34 volt", "item_weight")
    assert result == ""
    assert "blanked_invalid_unit_for_entity" in notes


def test_normalize_blanks_unparseable_output():
    for junk in ("", None, "I don't know", "n/a"):
        assert normalize_prediction(junk, "item_weight")[0] == ""


def test_range_rule_blank_is_the_default():
    assert normalize_prediction("10 to 20 gram", "item_weight")[0] == ""


def test_range_rule_max_and_min():
    max_opts = PostprocessOptions(range_rule="max")
    min_opts = PostprocessOptions(range_rule="min")
    assert normalize_prediction("10 to 20 gram", "item_weight", max_opts)[0] == "20 gram"
    assert normalize_prediction("10 to 20 gram", "item_weight", min_opts)[0] == "10 gram"


def test_disabling_postprocess_passes_raw_text_through():
    opts = PostprocessOptions(enabled=False)
    assert normalize_prediction("34 g", "item_weight", opts)[0] == "34 g"


def test_reject_invalid_units_is_individually_toggleable():
    opts = PostprocessOptions(reject_invalid_units=False)
    assert normalize_prediction("34 volt", "item_weight", opts)[0] == "34 volt"


def test_ground_truth_labels_survive_normalisation_unchanged():
    """Correctly formatted values must be fixed points, or we would corrupt
    predictions that were already right."""
    labels = [
        ("34 gram", "item_weight"), ("12.5 centimetre", "width"),
        ("2.56 ounce", "item_weight"), ("110 volt", "voltage"),
        ("1.5 litre", "item_volume"), ("8 fluid ounce", "item_volume"),
        ("3 cubic foot", "item_volume"), ("1000 watt", "wattage"),
    ]
    for label, entity in labels:
        assert normalize_prediction(label, entity)[0] == label, label


# --- batch -----------------------------------------------------------------
def test_apply_postprocess_returns_counts_of_fired_rules():
    raws = ["34 g", "10 to 20 gram", "34 volt", "", "12 gram"]
    entities = ["item_weight"] * 5

    cleaned, counts = apply_postprocess(raws, entities)

    assert cleaned == ["34 gram", "", "", "", "12 gram"]
    assert counts["total"] == 5
    assert counts["non_empty_out"] == 2
    assert counts["range_detected"] == 1
    assert counts["blanked_invalid_unit_for_entity"] == 1


def test_options_from_config_ignores_unknown_keys():
    opts = PostprocessOptions.from_config(
        {"enabled": True, "range_rule": "max", "some_future_flag": True}
    )
    assert opts.range_rule == "max" and opts.enabled is True


def test_options_from_config_falls_back_on_bad_range_rule():
    assert PostprocessOptions.from_config({"range_rule": "nonsense"}).range_rule == "blank"
