"""Shared fixtures. Redirects all writes to a tmp dir before importing amlc24."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Point results/images at a throwaway directory before amlc24.paths resolves
# them at import time, so the test suite can never touch real run artefacts.
_TMP = Path(tempfile.mkdtemp(prefix="amlc24_tests_"))
os.environ.setdefault("AMLC24_RESULTS_DIR", str(_TMP / "results"))
os.environ.setdefault("AMLC24_IMAGE_DIR", str(_TMP / "images"))
os.environ.setdefault("AMLC24_DATA_DIR", str(_TMP / "data"))


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture
def synthetic_train_df():
    """A small synthetic train.csv-shaped frame.

    Deliberately includes the messy cases the pipeline must survive: empty
    values, shared image URLs, multi-word units, and a decimal value.
    """
    import pandas as pd

    entities = [
        ("item_weight", "gram"), ("item_weight", "kilogram"),
        ("width", "centimetre"), ("height", "inch"), ("depth", "millimetre"),
        ("voltage", "volt"), ("wattage", "watt"),
        ("item_volume", "litre"), ("item_volume", "fluid ounce"),
        ("maximum_weight_recommendation", "pound"),
    ]

    rows = []
    for i in range(600):
        entity, unit = entities[i % len(entities)]
        if i % 17 == 0:
            value = ""  # empty labels are a real and scored category
        elif i % 23 == 0:
            value = f"{(i % 50) + 0.5} {unit}"
        else:
            value = f"{(i % 900) + 1} {unit}"
        rows.append({
            "index": i,
            # Every third row reuses an image, as in the real data.
            "image_link": f"https://example.invalid/img{i // 3}.jpg",
            "group_id": 100 + (i % 12),
            "entity_name": entity,
            "entity_value": value,
        })
    return pd.DataFrame(rows)
