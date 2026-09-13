"""EDA profiling tables, path resolution, and the leaderboard writer."""

from __future__ import annotations

import pandas as pd
import pytest

from amlc24.data.eda import (
    empty_rate,
    entity_distribution,
    format_audit,
    group_distribution,
    image_duplication,
    profile_dataset,
    unit_distribution,
    value_stats,
)
from amlc24.data.images import image_filename, image_path
from amlc24.paths import DATA_DIR, IMAGE_DIR, REPO_ROOT, RESULTS_DIR
from amlc24.results.tracker import LEADERBOARD_COLUMNS


# --- paths -----------------------------------------------------------------
def test_repo_root_points_at_the_project():
    assert (REPO_ROOT / "src" / "amlc24").is_dir()
    assert (REPO_ROOT / "configs" / "base.yaml").exists()


def test_env_overrides_are_respected(tmp_root):
    """conftest redirects writes to a temp dir; if this fails, tests would
    scribble into the real results/ directory."""
    assert str(tmp_root) in str(RESULTS_DIR)
    assert str(tmp_root) in str(IMAGE_DIR)
    assert str(tmp_root) in str(DATA_DIR)


def test_image_filenames_are_stable_and_safe():
    url = "https://m.media-amazon.com/images/I/abc123.jpg"
    name = image_filename(url)
    assert name == image_filename(url), "must be deterministic across calls"
    assert name.endswith(".jpg") and len(name) == 44
    assert not set(name) & set('<>:"/\\|?*'), "must be Windows-safe"


def test_different_urls_get_different_filenames():
    assert image_filename("https://a.invalid/1.jpg") != image_filename("https://a.invalid/2.jpg")


def test_image_path_lands_under_the_image_dir():
    assert image_path("https://a.invalid/1.jpg").parent == IMAGE_DIR


# --- EDA tables ------------------------------------------------------------
def test_entity_distribution_percentages_sum_to_100(synthetic_train_df):
    table = entity_distribution(synthetic_train_df)
    assert table["count"].sum() == len(synthetic_train_df)
    assert table["pct"].sum() == pytest.approx(100.0)
    assert table["cumulative_pct"].iloc[-1] == pytest.approx(100.0)


def test_entity_distribution_is_sorted_descending(synthetic_train_df):
    counts = entity_distribution(synthetic_train_df)["count"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_unit_distribution_flags_allowed_units(synthetic_train_df):
    table = unit_distribution(synthetic_train_df)
    weights = table[table["entity_name"] == "item_weight"]
    assert set(weights["unit"]) <= {"gram", "kilogram"}
    assert weights["is_allowed"].all()


def test_empty_rate_reports_overall_and_per_entity(synthetic_train_df):
    table = empty_rate(synthetic_train_df)
    overall = table[table["entity_name"] == "OVERALL"].iloc[0]
    expected = (synthetic_train_df["entity_value"].str.strip() == "").mean() * 100
    assert overall["pct_empty"] == pytest.approx(expected)
    assert overall["n"] == len(synthetic_train_df)


def test_value_stats_reports_range_per_entity(synthetic_train_df):
    table = value_stats(synthetic_train_df)
    assert not table.empty
    assert (table["min"] <= table["median"]).all()
    assert (table["median"] <= table["max"]).all()


def test_group_distribution_summarises_the_tail(synthetic_train_df):
    row = group_distribution(synthetic_train_df).iloc[0]
    assert row["n_distinct_groups"] == synthetic_train_df["group_id"].nunique()
    assert row["largest_group_size"] >= row["median_group_size"]


def test_image_duplication_counts_shared_images(synthetic_train_df):
    """The fixture reuses each URL across three rows."""
    row = image_duplication(synthetic_train_df).iloc[0]
    assert row["n_distinct_images"] == synthetic_train_df["image_link"].nunique()
    assert row["n_distinct_images"] < row["n_rows"]
    assert row["download_saving_pct"] > 0


# --- format audit ----------------------------------------------------------
def test_format_audit_reports_every_expected_property(synthetic_train_df):
    audit = format_audit(synthetic_train_df)
    required = {
        "integer_no_decimal_point", "trailing_dot_zero", "range_value",
        "unit_outside_allowed_list", "multiword_unit", "thousands_separator",
        "invalid_unit_but_canonicalisable",
    }
    assert required <= set(audit["property"])


def test_format_audit_detects_known_bad_formats():
    df = pd.DataFrame({
        "index": range(4),
        "image_link": ["https://a.invalid/x.jpg"] * 4,
        "group_id": [1] * 4,
        "entity_name": ["item_weight"] * 4,
        "entity_value": ["2.0 gram", "1,000 gram", "10 to 20 gram", "5 gms"],
    })
    audit = df.pipe(format_audit).set_index("property")["count"]

    assert audit["trailing_dot_zero"] == 1
    assert audit["thousands_separator"] == 1
    assert audit["range_value"] == 1
    # "gms" is not in constants.py but our alias table rescues it.
    assert audit["unit_outside_allowed_list"] >= 1
    assert audit["invalid_unit_but_canonicalisable"] >= 1


def test_format_audit_handles_an_all_empty_frame():
    df = pd.DataFrame({
        "index": [0], "image_link": ["https://a.invalid/x.jpg"],
        "group_id": [1], "entity_name": ["item_weight"], "entity_value": [""],
    })
    assert format_audit(df).empty


# --- full profile ----------------------------------------------------------
def test_profile_returns_every_required_table(synthetic_train_df):
    profile = profile_dataset(synthetic_train_df)
    required = {
        "entity_distribution", "unit_distribution", "empty_rate", "value_stats",
        "group_distribution", "format_audit", "image_duplication",
    }
    assert required <= set(profile)
    assert all(isinstance(v, pd.DataFrame) for v in profile.values())


def test_profile_tables_are_saved_as_csv(synthetic_train_df, tmp_path):
    from amlc24.data.eda import save_profile

    profile = profile_dataset(synthetic_train_df)
    out = save_profile(profile, tmp_path)
    assert (out / "entity_distribution.csv").exists()
    assert (out / "format_audit.csv").exists()


# --- leaderboard -----------------------------------------------------------
def test_leaderboard_columns_match_the_spec():
    expected = [
        "run_id", "timestamp", "description", "model", "quant_bits", "lora_r",
        "n_train", "n_eval", "epochs", "lr", "max_pixels", "f1_raw", "f1_post",
        "precision", "recall", "train_seconds", "config_hash", "notes",
    ]
    assert LEADERBOARD_COLUMNS == expected
