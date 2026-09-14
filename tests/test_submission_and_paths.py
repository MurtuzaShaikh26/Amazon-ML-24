"""Submission formatting rules, and Kaggle data-directory discovery."""

from __future__ import annotations

import pandas as pd
import pytest

from amlc24.paths import _search_for_data_dir
from amlc24.results.submission import (
    SUBMISSION_COLUMNS,
    SubmissionError,
    validate_prediction,
    validate_submission,
    write_submission,
)


# ---------------------------------------------------------------------------
# Data directory discovery
# ---------------------------------------------------------------------------
def _make_tree(root, *relative_dirs, with_train=True):
    for rel in relative_dirs:
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        if with_train:
            (d / "train.csv").write_text("index\n", encoding="utf-8")
    return root


def test_finds_the_official_nested_layout(tmp_path):
    """The real archive is '<slug>/student_resource 3/dataset/train.csv' --
    note the space in the folder name and the extra nesting level."""
    slug = tmp_path / "amazon-ml-challenge-2024"
    _make_tree(slug, "student_resource 3/dataset")

    found = _search_for_data_dir(slug)
    assert found == slug / "student_resource 3" / "dataset"
    assert (found / "train.csv").exists()


def test_finds_train_csv_at_the_dataset_root(tmp_path):
    slug = _make_tree(tmp_path / "flat", ".")
    assert _search_for_data_dir(slug) == slug


def test_finds_a_single_nested_dataset_folder(tmp_path):
    slug = _make_tree(tmp_path / "nested", "dataset")
    assert _search_for_data_dir(slug) == slug / "dataset"


def test_prefers_the_shallowest_match(tmp_path):
    """Breadth-first: the canonical outer copy wins over a nested duplicate."""
    slug = tmp_path / "both"
    _make_tree(slug, "dataset", "dataset/backup/dataset")
    assert _search_for_data_dir(slug) == slug / "dataset"


def test_returns_none_when_train_csv_is_absent(tmp_path):
    empty = tmp_path / "no-data"
    (empty / "src").mkdir(parents=True)
    assert _search_for_data_dir(empty) is None


def test_does_not_descend_into_image_directories(tmp_path):
    """An attached image dataset can hold 100k+ files; walking it would stall."""
    slug = tmp_path / "imgs"
    _make_tree(slug, "images/deep")
    assert _search_for_data_dir(slug) is None


def test_respects_the_depth_limit(tmp_path):
    slug = tmp_path / "deep"
    _make_tree(slug, "a/b/c/d/e/f/g")
    assert _search_for_data_dir(slug, max_depth=2) is None
    assert _search_for_data_dir(slug, max_depth=8) is not None


def test_constants_py_is_reachable_from_the_official_layout(tmp_path):
    """units.py looks for DATA_DIR.parent/src/constants.py -- confirm that is
    where the real archive puts it."""
    slug = tmp_path / "amazon-ml-challenge-2024"
    _make_tree(slug, "student_resource 3/dataset")
    (slug / "student_resource 3" / "src").mkdir(parents=True)
    (slug / "student_resource 3" / "src" / "constants.py").write_text("x=1", encoding="utf-8")

    data_dir = _search_for_data_dir(slug)
    assert (data_dir.parent / "src" / "constants.py").exists()


# ---------------------------------------------------------------------------
# Prediction validity (mirrors the official sanity checker)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pred", [
    "2 gram", "12.5 centimetre", "2.56 ounce", "500.0 gram",
    "8.0 fluid ounce", "3.0 cubic foot", "",
])
def test_valid_predictions_pass(pred):
    assert validate_prediction(pred) == []


@pytest.mark.parametrize("pred", [
    "2 gms",                      # unit not in constants.py
    "60 ounce/1.7 kilogram",      # compound
    "2.2e2 kilogram",             # exponent notation
    "2",                          # no unit
    "gram",                       # no number
    "2gram",                      # missing space
    "[10.0, 20.0] gram",          # bracket range is not a valid *submission*
])
def test_invalid_predictions_are_rejected(pred):
    assert validate_prediction(pred), f"{pred!r} should have been rejected"


def test_exponent_notation_is_called_out_explicitly():
    problems = validate_prediction("2.2e2 kilogram")
    assert any("exponent" in p for p in problems)


def test_unit_must_be_allowed_for_that_entity():
    assert validate_prediction("2 gram", "item_weight") == []
    problems = validate_prediction("2 volt", "item_weight")
    assert any("not allowed for entity" in p for p in problems)


def test_empty_prediction_is_always_valid():
    assert validate_prediction("", "item_weight") == []
    assert validate_prediction(None, "item_weight") == []


# ---------------------------------------------------------------------------
# Submission frame
# ---------------------------------------------------------------------------
@pytest.fixture
def test_df():
    return pd.DataFrame({
        "index": [0, 1, 2],
        "image_link": ["https://a.invalid/x.jpg"] * 3,
        "group_id": [1, 1, 2],
        "entity_name": ["item_weight", "width", "voltage"],
    })


def test_valid_submission_passes(test_df):
    sub = pd.DataFrame({"index": [0, 1, 2],
                        "prediction": ["2.0 gram", "12.5 centimetre", ""]})
    report = validate_submission(sub, test_df)
    assert report["valid"] and report["n_non_empty"] == 2 and report["n_empty"] == 1


def test_wrong_column_names_are_rejected(test_df):
    sub = pd.DataFrame({"idx": [0, 1, 2], "pred": ["", "", ""]})
    with pytest.raises(SubmissionError, match="columns"):
        validate_submission(sub, test_df)


def test_row_count_mismatch_is_rejected(test_df):
    """The statement says a short/long file is not evaluated at all, and the
    shipped sanity.py does not check this -- so we must."""
    sub = pd.DataFrame({"index": [0, 1], "prediction": ["2.0 gram", ""]})
    with pytest.raises(SubmissionError, match="row count"):
        validate_submission(sub, test_df)


def test_duplicate_indices_are_rejected(test_df):
    sub = pd.DataFrame({"index": [0, 0, 2], "prediction": ["", "", ""]})
    with pytest.raises(SubmissionError, match="duplicate"):
        validate_submission(sub, test_df)


def test_cross_entity_unit_is_rejected(test_df):
    sub = pd.DataFrame({"index": [0, 1, 2],
                        "prediction": ["2.0 volt", "12.5 centimetre", ""]})
    with pytest.raises(SubmissionError, match="invalid predictions"):
        validate_submission(sub, test_df)


def test_report_without_raising_lists_the_bad_rows(test_df):
    sub = pd.DataFrame({"index": [0, 1, 2],
                        "prediction": ["2 gms", "12.5 centimetre", ""]})
    report = validate_submission(sub, test_df, raise_on_error=False)
    assert not report["valid"] and report["n_invalid"] == 1
    assert "2 gms" in report["bad_rows"]["prediction"].tolist()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_write_submission_round_trips(tmp_path, test_df):
    preds = pd.DataFrame({"index": [0, 1, 2],
                          "prediction": ["2.0 gram", "12.5 centimetre", ""]})
    path = write_submission(preds, tmp_path / "sub.csv", test_df)

    written = pd.read_csv(path, keep_default_na=False)
    assert list(written.columns) == SUBMISSION_COLUMNS
    assert len(written) == 3
    assert written.loc[0, "prediction"] == "2.0 gram"
    assert written.loc[2, "prediction"] == ""


def test_write_submission_accepts_the_pipeline_column_name(tmp_path, test_df):
    preds = pd.DataFrame({"index": [0, 1, 2],
                          "y_pred_post": ["2.0 gram", "12.5 centimetre", ""]})
    path = write_submission(preds, tmp_path / "sub.csv", test_df)
    assert pd.read_csv(path, keep_default_na=False).loc[0, "prediction"] == "2.0 gram"


def test_missing_indices_are_filled_rather_than_dropped(tmp_path, test_df):
    """A short file is not evaluated at all; an empty row costs only one FN."""
    preds = pd.DataFrame({"index": [0], "prediction": ["2.0 gram"]})
    path = write_submission(preds, tmp_path / "sub.csv", test_df)

    written = pd.read_csv(path, keep_default_na=False)
    assert len(written) == 3
    assert set(written["index"]) == {0, 1, 2}
    assert written[written["index"] == 1]["prediction"].iloc[0] == ""


def test_written_rows_follow_test_csv_order(tmp_path, test_df):
    preds = pd.DataFrame({"index": [2, 0, 1], "prediction": ["", "2.0 gram", "1.0 centimetre"]})
    path = write_submission(preds, tmp_path / "sub.csv", test_df)
    assert pd.read_csv(path)["index"].tolist() == [0, 1, 2]


def test_write_rejects_a_frame_with_no_prediction_column(tmp_path, test_df):
    with pytest.raises(SubmissionError, match="No prediction column"):
        write_submission(pd.DataFrame({"index": [0]}), tmp_path / "s.csv", test_df)
