"""Class weights for the 31.5x entity imbalance found by the EDA."""

from __future__ import annotations

import pytest

from amlc24.train.weighting import (
    WEIGHT_SCHEMES,
    compute_class_weights,
    describe_weights,
    weights_from_config,
)

# Proportions mirroring the real train.csv distribution.
REAL_SHARES = {
    "item_weight": 3895,
    "depth": 1710,
    "width": 1674,
    "height": 1652,
    "voltage": 359,
    "wattage": 294,
    "item_volume": 291,
    "maximum_weight_recommendation": 124,
}


@pytest.fixture
def entities():
    return [name for name, count in REAL_SHARES.items() for _ in range(count)]


# --- normalisation ---------------------------------------------------------
def test_weights_normalise_to_mean_one_over_the_training_distribution(entities):
    """Keeps the loss scale (and so the effective LR) unchanged across schemes."""
    weights = compute_class_weights(entities, scheme="sqrt_inverse")
    mean = sum(weights[e] for e in entities) / len(entities)
    assert mean == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("scheme", ["inverse", "sqrt_inverse", "effective"])
def test_every_scheme_normalises_to_mean_one(entities, scheme):
    weights = compute_class_weights(entities, scheme=scheme)
    mean = sum(weights[e] for e in entities) / len(entities)
    assert mean == pytest.approx(1.0, abs=1e-9)


# --- direction -------------------------------------------------------------
def test_rare_classes_get_larger_weights(entities):
    weights = compute_class_weights(entities, scheme="sqrt_inverse")
    assert weights["maximum_weight_recommendation"] > weights["item_weight"]
    assert weights["item_volume"] > weights["depth"]


def test_weight_order_is_the_inverse_of_frequency_order(entities):
    weights = compute_class_weights(entities, scheme="sqrt_inverse")
    by_freq = sorted(REAL_SHARES, key=lambda k: -REAL_SHARES[k])
    by_weight = sorted(weights, key=lambda k: weights[k])
    assert by_freq == by_weight


def test_inverse_fully_balances_classes(entities):
    """Under `inverse`, weight x count is the same for every class."""
    weights = compute_class_weights(entities, scheme="inverse", max_weight=None)
    mass = {name: weights[name] * count for name, count in REAL_SHARES.items()}
    assert max(mass.values()) == pytest.approx(min(mass.values()))


def test_sqrt_inverse_is_gentler_than_inverse(entities):
    """The reason it is the default: full inverse gives the rarest class a ~31x
    multiplier, which destabilises fp16 training at batch size 1."""
    sqrt_w = compute_class_weights(entities, scheme="sqrt_inverse", max_weight=None)
    inv_w = compute_class_weights(entities, scheme="inverse", max_weight=None)

    sqrt_spread = max(sqrt_w.values()) / min(sqrt_w.values())
    inv_spread = max(inv_w.values()) / min(inv_w.values())

    assert sqrt_spread < inv_spread
    assert inv_spread == pytest.approx(sqrt_spread ** 2, rel=1e-6)
    assert sqrt_spread < 7.0


# --- guards ----------------------------------------------------------------
def test_max_weight_caps_the_spread(entities):
    weights = compute_class_weights(entities, scheme="inverse", max_weight=3.0)
    assert max(weights.values()) / min(weights.values()) <= 3.0 + 1e-9


def test_scheme_none_gives_uniform_weights(entities):
    assert set(compute_class_weights(entities, scheme="none").values()) == {1.0}


def test_balanced_input_gives_near_uniform_weights():
    entities = ["a"] * 100 + ["b"] * 100
    weights = compute_class_weights(entities, scheme="sqrt_inverse")
    assert weights["a"] == pytest.approx(weights["b"])


def test_unknown_scheme_raises(entities):
    with pytest.raises(ValueError, match="Unknown class_weight scheme"):
        compute_class_weights(entities, scheme="nope")


def test_empty_input_returns_empty_mapping():
    assert compute_class_weights([]) == {}


# --- config plumbing -------------------------------------------------------
def test_disabled_config_returns_empty_so_trainer_uses_plain_loss(entities):
    assert weights_from_config(entities, {"enabled": False}) == {}
    assert weights_from_config(entities, None) == {}


def test_enabled_config_returns_weights(entities):
    weights = weights_from_config(entities, {"enabled": True, "scheme": "sqrt_inverse"})
    assert len(weights) == len(REAL_SHARES)
    assert weights["maximum_weight_recommendation"] > 1.0


def test_scheme_none_via_config_disables_weighting(entities):
    assert weights_from_config(entities, {"enabled": True, "scheme": "none"}) == {}


def test_unknown_scheme_in_config_falls_back_instead_of_raising(entities):
    weights = weights_from_config(entities, {"enabled": True, "scheme": "bogus"})
    assert weights, "should fall back to the default scheme, not fail the run"


def test_describe_weights_is_json_serialisable_and_sorted(entities):
    described = describe_weights(compute_class_weights(entities, scheme="sqrt_inverse"))
    assert [d["entity_name"] for d in described][0] == "maximum_weight_recommendation"
    assert all(isinstance(d["weight"], float) for d in described)


def test_all_declared_schemes_are_usable(entities):
    for scheme in WEIGHT_SCHEMES:
        assert isinstance(compute_class_weights(entities, scheme=scheme), dict)
