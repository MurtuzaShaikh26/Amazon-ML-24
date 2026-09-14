"""`preserve` number format, the post-processing ablation, and the time budget."""

from __future__ import annotations

import pytest

from amlc24.pipeline.run_finetune import postprocess_ablation
from amlc24.postprocess.normalize import (
    PostprocessOptions,
    format_number,
    normalize_prediction,
)
from amlc24.train.trainer import TrainingTimeBudget


# --- preserve ---------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("50", "50"),            # bare integer kept: ~50% of max_weight labels
    ("500.0", "500.0"),      # float style kept
    ("3.53", "3.53"),
    ("2.50", "2.5"),         # redundant trailing zero: 0% of labels
    ("2.00", "2.0"),
    (".5", "0.5"),
    ("1,000", "1000"),
    ("2.2e2", "220.0"),      # exponent is invalid output -> plain float
    ("1e20", "100000000000000000000.0"),
    ("1e-3", "0.001"),
])
def test_preserve_keeps_plain_forms_and_fixes_invalid_ones(raw, expected):
    assert format_number(raw, "preserve") == expected


def test_preserve_is_the_default():
    assert PostprocessOptions().number_format == "preserve"
    assert PostprocessOptions.from_config({}).number_format == "preserve"


@pytest.mark.parametrize("raw,entity,expected", [
    # The smoke-run regression: float turned a correct "50 pound" into "50.0 pound".
    ("50 pound", "maximum_weight_recommendation", "50 pound"),
    ("500.0 gram", "item_weight", "500.0 gram"),
    ("50 lbs", "maximum_weight_recommendation", "50 pound"),
    ("[100.0, 240.0] volt", "voltage", "[100.0, 240.0] volt"),
])
def test_preserve_keeps_a_fine_tuned_models_format(raw, entity, expected):
    assert normalize_prediction(raw, entity)[0] == expected


# --- ablation ---------------------------------------------------------------
Y = ["50 pound", "500.0 gram", "[100.0, 240.0] volt"]
RAW = ["50 pound", "500 gram", "100 to 240 volt"]
ENT = ["maximum_weight_recommendation", "item_weight", "voltage"]


def test_ablation_scores_raw_plus_every_variant():
    table = postprocess_ablation(Y, RAW, ENT, PostprocessOptions())
    assert len(table) == 1 + 4 * 4
    assert "raw (no post-processing)" in set(table["variant"])
    assert {"f1", "precision", "recall", "macro_f1", "is_config"} <= set(table.columns)


def test_ablation_flags_exactly_the_configured_variant():
    table = postprocess_ablation(Y, RAW, ENT, PostprocessOptions(number_format="float",
                                                                 range_rule="blank"))
    flagged = table[table["is_config"]]
    assert len(flagged) == 1 and flagged.iloc[0]["variant"] == "float / blank"


def test_ablation_is_sorted_best_first():
    table = postprocess_ablation(Y, RAW, ENT)
    assert table["f1"].tolist() == sorted(table["f1"], reverse=True)


def test_ablation_exposes_the_smoke_run_regression():
    y = ["50 pound", "60 pound"]
    table = postprocess_ablation(y, list(y), ["maximum_weight_recommendation"] * 2)
    table = table.set_index("variant")
    assert table.loc["preserve / bracket", "f1"] == pytest.approx(1.0)
    assert table.loc["float / bracket", "f1"] == pytest.approx(0.0)


# --- time budget ------------------------------------------------------------
def test_budget_disabled_never_stops():
    assert not TrainingTimeBudget(None).should_stop(1e9)


def test_budget_stops_at_the_limit():
    budget = TrainingTimeBudget(1.0)
    assert not budget.should_stop(3599)
    assert budget.should_stop(3600)


def test_projection_extrapolates_linearly():
    assert TrainingTimeBudget.project_total_seconds(100, 10, 50) == pytest.approx(500)


def test_projection_is_none_before_the_first_step():
    assert TrainingTimeBudget.project_total_seconds(5, 0, 50) is None


def test_smoke_throughput_shows_three_epochs_cannot_fit():
    """2.7 s/sample x 8 samples/step = 21.6 s/step; 10k x 3 epochs = 3,750 steps."""
    total = TrainingTimeBudget.project_total_seconds(21.6 * 20, 20, 3750)
    assert total / 3600 > 12
    one_epoch = TrainingTimeBudget.project_total_seconds(21.6 * 20, 20, 1250)
    assert one_epoch / 3600 < 9
