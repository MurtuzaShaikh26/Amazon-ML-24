"""Config inheritance, dot access, and hashing."""

from __future__ import annotations

import re

import pytest

from amlc24.config import (
    HASH_LENGTH,
    Config,
    ConfigError,
    config_hash,
    deep_merge,
    load_config,
)
from amlc24.paths import CONFIG_DIR

HEX_RE = re.compile(r"^[0-9a-f]+$")


# --- hashing ---------------------------------------------------------------
def test_hash_is_hex_of_expected_length():
    cfg = Config({"run_id": "a", "description": "d", "seed": 42})
    digest = config_hash(cfg)
    assert len(digest) == HASH_LENGTH
    assert HEX_RE.match(digest), f"{digest!r} is not lowercase hex"


def test_hash_is_stable_across_calls_and_key_order():
    a = Config({"run_id": "x", "description": "d", "train": {"lr": 1e-4, "epochs": 3}})
    b = Config({"description": "d", "train": {"epochs": 3, "lr": 1e-4}, "run_id": "x"})
    assert config_hash(a) == config_hash(a)
    assert config_hash(a) == config_hash(b), "key order must not change the hash"


def test_different_configs_hash_differently():
    base = {"run_id": "x", "description": "d", "train": {"lr": 2e-4}}
    variants = [
        {**base, "train": {"lr": 1e-4}},
        {**base, "run_id": "y"},
        {**base, "train": {"lr": 2e-4, "epochs": 3}},
        {**base, "description": "other"},
    ]
    digests = {config_hash(Config(base))} | {config_hash(Config(v)) for v in variants}
    assert len(digests) == len(variants) + 1, "each distinct config needs its own hash"


def test_hash_length_is_configurable():
    cfg = Config({"run_id": "a", "description": "d"})
    assert len(config_hash(cfg, length=16)) == 16


def test_extends_key_does_not_affect_hash():
    a = Config({"run_id": "x", "description": "d", "extends": "base.yaml"})
    b = Config({"run_id": "x", "description": "d"})
    assert config_hash(a) == config_hash(b)


# --- merging ---------------------------------------------------------------
def test_deep_merge_recurses_into_dicts():
    merged = deep_merge(
        {"train": {"lr": 1e-4, "epochs": 3}, "seed": 1},
        {"train": {"lr": 2e-4}},
    )
    assert merged == {"train": {"lr": 2e-4, "epochs": 3}, "seed": 1}


def test_deep_merge_replaces_lists_rather_than_concatenating():
    merged = deep_merge({"t": ["q_proj", "k_proj"]}, {"t": ["v_proj"]})
    assert merged["t"] == ["v_proj"], "a child must be able to shorten a list"


def test_deep_merge_does_not_mutate_inputs():
    base = {"train": {"lr": 1e-4}}
    deep_merge(base, {"train": {"lr": 9e-9}})
    assert base["train"]["lr"] == 1e-4


# --- access ----------------------------------------------------------------
def test_dot_and_item_access_and_nesting():
    cfg = Config({"run_id": "x", "description": "d", "lora": {"r": 16}})
    assert cfg.run_id == "x"
    assert cfg["lora"]["r"] == 16
    assert cfg.lora.r == 16
    assert cfg.get_path("lora.r") == 16
    assert cfg.get_path("lora.missing", "fallback") == "fallback"


def test_missing_attribute_raises_attribute_error_naming_keys():
    cfg = Config({"run_id": "x", "description": "d"})
    with pytest.raises(AttributeError, match="nope"):
        _ = cfg.nope


def test_config_is_read_only():
    cfg = Config({"run_id": "x", "description": "d"})
    with pytest.raises(TypeError):
        cfg.run_id = "y"


def test_config_supports_mapping_protocol():
    cfg = Config({"run_id": "x", "description": "d"})
    assert dict(**cfg) == {"run_id": "x", "description": "d"}
    assert "run_id" in cfg and len(cfg) == 2


# --- real files ------------------------------------------------------------
def test_run001_inherits_from_base():
    cfg = load_config(CONFIG_DIR / "run001_qwen2vl_8bit_10k.yaml")
    assert cfg.run_id == "run001_qwen2vl_8bit_10k"
    # Overridden in the child:
    assert cfg.quantization.bits == 8
    # 1, not 3: the smoke run measured ~2.7 s/sample, so 3 epochs needs ~22 h.
    assert cfg.train.num_train_epochs == 1
    assert cfg.train.max_train_hours == 9.0
    # Inherited from base.yaml only:
    assert cfg.train.max_length == 1024
    assert cfg.inference.batch_size == 4
    assert cfg.postprocess.strip_scaffolding is True


def test_config_loadable_by_bare_name():
    assert load_config("run001_qwen2vl_8bit_10k").run_id == "run001_qwen2vl_8bit_10k"


def test_run001_memory_critical_settings():
    """These specific values are what make a 7B model fit on a 16 GB T4."""
    cfg = load_config("run001_qwen2vl_8bit_10k")
    assert cfg.train.per_device_train_batch_size == 1
    assert cfg.train.gradient_accumulation_steps == 8
    assert cfg.train.gradient_checkpointing is True
    assert cfg.train.fp16 is True
    assert cfg.train.optim == "paged_adamw_8bit"
    assert cfg.processor.max_pixels_tokens == 256
    assert cfg.model.attn_implementation == "sdpa"
    assert cfg.lora.freeze_vision_tower is True
    assert cfg.quantization.bits in (4, 8)


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="run_id"):
        load_config(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_folded_description_is_collapsed_to_one_line():
    cfg = load_config("run001_qwen2vl_8bit_10k")
    assert "\n" not in cfg.description
    assert cfg.description.startswith("First fine-tune.")
