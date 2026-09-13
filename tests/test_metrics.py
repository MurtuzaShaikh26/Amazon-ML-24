"""Competition F1, with hand-computed cases for all four outcome classes."""

from __future__ import annotations

import pandas as pd
import pytest

from amlc24.metrics.f1 import (
    classify,
    error_analysis,
    evaluate,
    f1_by_entity,
    f1_by_group,
    f1_by_unit,
    f1_score,
    macro_f1,
)


# --- the four outcome classes ---------------------------------------------
def test_true_positive_requires_exact_equality():
    assert classify("34 gram", "34 gram") == "TP"


def test_false_positive_when_non_empty_prediction_differs():
    assert classify("34 gram", "35 gram") == "FP"


def test_false_positive_when_truth_is_empty():
    assert classify("", "34 gram") == "FP"


def test_false_negative_when_prediction_is_empty():
    assert classify("34 gram", "") == "FN"


def test_true_negative_when_both_empty():
    assert classify("", "") == "TN"


@pytest.mark.parametrize("empty", [None, "", "   ", float("nan")])
def test_all_empty_representations_are_equivalent(empty):
    assert classify(empty, empty) == "TN"
    assert classify("2 gram", empty) == "FN"
    assert classify(empty, "2 gram") == "FP"


# --- exactness -------------------------------------------------------------
@pytest.mark.parametrize("pred", ["2.0 gram", "2 gms", "2 g", "2 Gram", "2gram", "2  gram"])
def test_near_misses_are_false_positives_not_true_positives(pred):
    """The metric is exact string match: formatting counts as much as the number."""
    assert classify("2 gram", pred) == "FP"


def test_surrounding_whitespace_is_the_only_leniency():
    assert classify("2 gram", "  2 gram  ") == "TP"


# --- aggregate -------------------------------------------------------------
def test_f1_hand_computed():
    """3 TP, 2 FP, 1 FN, 2 TN.

    precision = 3/5 = 0.6, recall = 3/4 = 0.75,
    F1 = 2*0.6*0.75 / 1.35 = 0.9/1.35 = 0.666666...
    """
    y_true = ["1 gram", "2 gram", "3 gram", "4 gram", "",       "5 gram", "", ""]
    y_pred = ["1 gram", "2 gram", "3 gram", "9 gram", "7 gram", "",       "", ""]

    scores = f1_score(y_true, y_pred)
    assert (scores["tp"], scores["fp"], scores["fn"], scores["tn"]) == (3, 2, 1, 2)
    assert scores["precision"] == pytest.approx(0.6)
    assert scores["recall"] == pytest.approx(0.75)
    assert scores["f1"] == pytest.approx(2 / 3)
    assert scores["n"] == 8


def test_perfect_predictions_score_one():
    y = ["1 gram", "2 gram", ""]
    assert f1_score(y, y)["f1"] == pytest.approx(1.0)


def test_all_empty_predictions_score_zero_not_one():
    """Abstaining everywhere must not be rewarded, even though TN count is high."""
    scores = f1_score(["1 gram", "2 gram", ""], ["", "", ""])
    assert scores["f1"] == 0.0
    assert scores["tn"] == 1 and scores["fn"] == 2


def test_no_true_or_predicted_positives_is_zero_not_nan():
    scores = f1_score(["", ""], ["", ""])
    assert scores["f1"] == 0.0
    assert scores["precision"] == 0.0 and scores["recall"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        f1_score(["a"], ["a", "b"])


# --- breakdowns ------------------------------------------------------------
def test_f1_by_entity_splits_and_totals_correctly():
    y_true = ["1 gram", "2 gram", "3 centimetre", "4 centimetre"]
    y_pred = ["1 gram", "9 gram", "3 centimetre", "4 centimetre"]
    entities = ["item_weight", "item_weight", "width", "width"]

    table = f1_by_entity(y_true, y_pred, entities).set_index("entity_name")

    assert table.loc["width", "f1"] == pytest.approx(1.0)
    assert table.loc["item_weight", "tp"] == 1
    assert table.loc["item_weight", "fp"] == 1
    assert table.loc["TOTAL", "n"] == 4
    assert table.loc["TOTAL", "tp"] == 3


def test_f1_by_entity_covers_every_entity_present():
    entities = ["item_weight", "width", "voltage"]
    table = f1_by_entity(["1 gram", "", "5 volt"], ["1 gram", "", "5 volt"], entities)
    assert set(entities) <= set(table["entity_name"])


def test_f1_by_unit_groups_on_ground_truth_unit():
    y_true = ["1 gram", "2 kilogram", "3 gram"]
    y_pred = ["1 gram", "9 kilogram", "3 gram"]
    entities = ["item_weight"] * 3

    table = f1_by_unit(y_true, y_pred, entities).set_index("unit")
    assert table.loc["gram", "tp"] == 2
    assert table.loc["kilogram", "fp"] == 1
    assert "entities" in table.columns


def test_f1_by_unit_buckets_empty_truth_separately():
    table = f1_by_unit(["", "1 gram"], ["5 gram", "1 gram"], ["item_weight"] * 2)
    assert "(empty)" in set(table["unit"])


# --- error analysis --------------------------------------------------------
def test_error_analysis_ranks_mismatch_pairs_by_frequency():
    y_true = ["34 gram"] * 5 + ["10 gram"]
    y_pred = ["34 g"] * 5 + ["11 gram"]
    entities = ["item_weight"] * 6

    errors = error_analysis(y_true, y_pred, entities, top_k=20)
    top = errors.iloc[0]
    assert top["y_pred"] == "34 g" and top["y_true"] == "34 gram"
    assert top["count"] == 5
    assert bool(top["same_number_diff_unit"]) is True, (
        "a pure unit-formatting miss must be flagged as recoverable"
    )


def test_error_analysis_respects_top_k():
    y_true = [f"{i} gram" for i in range(50)]
    y_pred = [f"{i + 1} gram" for i in range(50)]
    errors = error_analysis(y_true, y_pred, ["item_weight"] * 50, top_k=20)
    assert len(errors) == 20


def test_error_analysis_returns_empty_frame_when_perfect():
    y = ["1 gram", "2 gram"]
    errors = error_analysis(y, y, ["item_weight"] * 2)
    assert errors.empty
    assert "y_pred" in errors.columns  # shape preserved for downstream writers


def test_error_analysis_marks_value_errors_as_not_recoverable():
    errors = error_analysis(["10 gram"], ["11 gram"], ["item_weight"])
    assert bool(errors.iloc[0]["same_number_diff_unit"]) is False


# --- macro F1 (class-balanced view) ----------------------------------------
def test_macro_f1_weights_every_entity_equally():
    """Micro F1 is dominated by item_weight (38.95% of the data); macro is the
    number that moves when class weighting helps a rare entity."""
    y_true = ["1 gram"] * 9 + ["5 volt"]
    y_pred = ["1 gram"] * 9 + ["9 volt"]          # the rare class is wrong
    entities = ["item_weight"] * 9 + ["voltage"]

    micro = f1_score(y_true, y_pred)["f1"]
    macro = macro_f1(f1_by_entity(y_true, y_pred, entities))

    assert micro == pytest.approx(0.9473684, abs=1e-6)
    assert macro == pytest.approx(0.5), "one perfect + one zero entity -> 0.5"
    assert macro < micro, "macro must expose the rare-class failure micro hides"


def test_macro_f1_equals_micro_when_entities_are_balanced_and_equal():
    y = ["1 gram", "5 volt"]
    entities = ["item_weight", "voltage"]
    assert macro_f1(f1_by_entity(y, y, entities)) == pytest.approx(1.0)


def test_macro_f1_excludes_the_total_row():
    table = f1_by_entity(["1 gram", "5 volt"], ["1 gram", "5 volt"],
                         ["item_weight", "voltage"])
    assert "TOTAL" in set(table["entity_name"])
    assert macro_f1(table) == pytest.approx(1.0)


# --- per-group (category-wise) ---------------------------------------------
def test_f1_by_group_breaks_down_by_product_category():
    y_true = ["1 gram"] * 4
    y_pred = ["1 gram", "1 gram", "9 gram", "9 gram"]
    groups = [100, 100, 200, 200]

    table = f1_by_group(y_true, y_pred, groups, min_support=1).set_index("group_id")
    assert table.loc["100", "f1"] == pytest.approx(1.0)
    assert table.loc["200", "f1"] == pytest.approx(0.0)
    assert table.loc["TOTAL", "n"] == 4


def test_f1_by_group_pools_small_categories():
    """750 groups with a long tail would otherwise produce mostly noise."""
    y_true = ["1 gram"] * 30
    y_pred = ["1 gram"] * 30
    groups = [1] * 25 + list(range(100, 105))     # one big group, five singletons

    table = f1_by_group(y_true, y_pred, groups, min_support=10)
    labels = set(table["group_id"])
    assert "1" in labels
    pooled = [l for l in labels if l.startswith("(small groups")]
    assert len(pooled) == 1, "the five singleton groups must be pooled into one row"
    assert table[table["group_id"].isin(pooled)]["n"].iloc[0] == 5


def test_f1_by_group_totals_match_the_overall_score():
    y_true = ["1 gram", "2 gram", "3 gram"]
    y_pred = ["1 gram", "9 gram", "3 gram"]
    groups = [1, 2, 3]

    table = f1_by_group(y_true, y_pred, groups, min_support=1)
    total = table[table["group_id"] == "TOTAL"].iloc[0]
    assert total["f1"] == pytest.approx(f1_score(y_true, y_pred)["f1"])


# --- evaluate() ------------------------------------------------------------
def test_evaluate_returns_every_breakdown():
    y_true = ["1 gram", "5 volt"]
    y_pred = ["1 gram", "9 volt"]
    entities = ["item_weight", "voltage"]

    result = evaluate(y_true, y_pred, entities, group_ids=[1, 2])
    assert {"overall", "macro_f1", "by_entity", "by_unit", "errors", "by_group"} <= set(result)
    assert isinstance(result["macro_f1"], float)


def test_evaluate_omits_group_table_when_no_group_ids_given():
    result = evaluate(["1 gram"], ["1 gram"], ["item_weight"])
    assert "by_group" not in result
