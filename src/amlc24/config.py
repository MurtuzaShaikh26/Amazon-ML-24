"""YAML configuration with ``extends:`` inheritance, deep merge, dot access.

A config file may declare ``extends: base.yaml`` (resolved relative to the file
itself, then to ``configs/``). Parents are loaded recursively and deep-merged,
with the child winning on scalar conflicts and dicts merged key-by-key. Lists
are *replaced*, not concatenated -- concatenation semantics make it impossible
to shorten a list in a child config.

Every resolved config must carry ``run_id`` and ``description``.

``config_hash`` gives a short, stable sha1 over the canonical JSON form so a
leaderboard row can be traced back to the exact settings that produced it. The
``extends`` key is stripped before hashing: two configs that resolve to the same
settings by different inheritance routes are the same experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml

from .paths import CONFIG_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

HASH_LENGTH = 10
REQUIRED_KEYS = ("run_id", "description")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or incomplete."""


class Config(Mapping):
    """Immutable-ish, dot-accessible view over a nested dict.

    Nested dicts are wrapped on access so ``cfg.train.learning_rate`` works.
    Implements ``Mapping`` so ``**cfg``, ``cfg["k"]``, ``in``, and ``dict(cfg)``
    all behave as expected.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- dot access ---------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key {key!r}. Available: {sorted(self._data)}"
            ) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError(
            "Config is read-only; build a new one with Config(dict(cfg, **changes))"
        )

    # -- helpers ------------------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. ``cfg.get_path('lora.r')``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict:
        """Deep copy as a plain dict, safe to mutate or serialise."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, indent=2, default=str, sort_keys=True)})"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` onto ``base``. Dicts merge; everything
    else (including lists) is replaced wholesale."""
    out = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_parent(ref: str, child_path: Path) -> Path:
    """Resolve an ``extends:`` reference relative to the child, then configs/,
    then the repo root."""
    for candidate in (child_path.parent / ref, CONFIG_DIR / ref, REPO_ROOT / ref):
        if candidate.exists():
            return candidate.resolve()
    raise ConfigError(
        f"{child_path.name} extends {ref!r}, which was not found near the child, "
        f"in {CONFIG_DIR}, or at the repo root."
    )


def _load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load one YAML file and merge it onto its resolved parent chain."""
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ConfigError(f"Circular extends chain: {chain}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    refs: Iterable[str] = [parent_ref] if isinstance(parent_ref, str) else parent_ref
    merged: dict = {}
    for ref in refs:
        parent = _load_raw(_resolve_parent(ref, path), (*_seen, path))
        merged = deep_merge(merged, parent)
    return deep_merge(merged, data)


def load_config(path: str | Path) -> Config:
    """Load, resolve inheritance for, and validate a config file.

    A bare name (``run001_qwen2vl_8bit_10k`` or ``...yaml``) is looked up in
    ``configs/``, so notebooks need not know where the repo lives on disk.
    """
    p = Path(path)
    if not p.exists():
        for candidate in (CONFIG_DIR / p.name, CONFIG_DIR / f"{p.name}.yaml"):
            if candidate.exists():
                p = candidate
                break
    data = _load_raw(p)

    missing = [k for k in REQUIRED_KEYS if not data.get(k)]
    if missing:
        raise ConfigError(f"{p} is missing required key(s): {missing}")

    # Descriptions written with YAML folded scalars pick up a trailing newline.
    data["description"] = " ".join(str(data["description"]).split())
    cfg = Config(data)
    logger.info("Loaded config %s (hash=%s) from %s", cfg.run_id, config_hash(cfg), p)
    return cfg


def _canonical(obj: Any) -> Any:
    """Recursively convert to JSON-safe primitives with deterministic ordering."""
    if isinstance(obj, Config):
        obj = obj.to_dict()
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def config_hash(cfg: Config | Mapping[str, Any], length: int = HASH_LENGTH) -> str:
    """Short sha1 over the canonical JSON form of the resolved config."""
    payload = _canonical(cfg)
    if isinstance(payload, dict):
        payload.pop("extends", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def dump_config(cfg: Config, path: str | Path) -> Path:
    """Write the fully resolved config to ``path`` for run reproducibility."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=True, default_flow_style=False)
    return out


__all__ = [
    "Config", "ConfigError", "load_config", "config_hash", "dump_config",
    "deep_merge", "HASH_LENGTH",
]
