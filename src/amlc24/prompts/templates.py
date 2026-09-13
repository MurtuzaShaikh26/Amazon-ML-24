"""Versioned prompt templates for Qwen2-VL.

Templates are *named functions* registered in ``TEMPLATES`` and selected by
``cfg.prompt.template``. Changing the prompt is therefore a config edit that
shows up in the leaderboard's ``config_hash``, which makes prompt variants a
measurable ablation instead of an untracked code change.

Every template returns the Qwen2-VL chat message structure::

    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ...}]}]

The image placeholder carries no data; ``processor.apply_chat_template`` expands
it into the vision tokens and the real PIL image is passed separately to the
processor call. Assistant turns are appended only for training targets.

Listing the entity's allowed units in the prompt measurably reduces invalid-unit
outputs -- the model otherwise writes whatever the packaging says (``g``, ``fl.
oz.``), which the exact-match metric scores as wrong.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..postprocess.units import allowed_units

logger = logging.getLogger(__name__)

PromptFn = Callable[..., list[dict[str, Any]]]

SYSTEM_TEXT = (
    "You read product images and extract measurement values exactly as printed."
)


def _entity_label(entity_name: str) -> str:
    """``item_weight`` -> ``item weight`` for more natural prompting."""
    return str(entity_name).replace("_", " ").strip()


def _units_clause(entity_name: str, include: bool) -> str:
    units = sorted(allowed_units(entity_name))
    if not include or not units:
        return ""
    return (
        "\nThe unit must be exactly one of: "
        + ", ".join(units)
        + "."
    )


# Representative unit per entity, used to build the format example. A weight
# example inside a voltage prompt would both confuse the instruction and bias
# the model toward an invalid unit, so the example always uses a unit that is
# actually allowed for the entity being asked about.
_EXAMPLE_UNIT = {
    "item_weight": "gram",
    "maximum_weight_recommendation": "kilogram",
    "width": "centimetre",
    "height": "centimetre",
    "depth": "centimetre",
    "voltage": "volt",
    "wattage": "watt",
    "item_volume": "litre",
}


def _example_value(entity_name: str) -> str:
    """A format example using a unit valid for this entity."""
    unit = _EXAMPLE_UNIT.get(entity_name)
    if unit is None or unit not in allowed_units(entity_name):
        permitted = sorted(allowed_units(entity_name))
        unit = permitted[0] if permitted else "unit"
    return f"34 {unit}"


def prompt_v1(
    entity_name: str,
    include_allowed_units: bool = True,
    **_: Any,
) -> list[dict[str, Any]]:
    """Baseline template: direct instruction, explicit format, allowed units.

    The format demand is stated twice (once as a rule, once as a negative
    example) because instruction-tuned models reliably add prose like "The
    weight is 34 grams." otherwise, and every such wrapper is a false positive.
    """
    label = _entity_label(entity_name)
    text = (
        f"Look at this product image and extract the {label}.\n"
        f"Answer with only the value and its unit, in the form: "
        f"<number> <unit>\n"
        f"For example: {_example_value(entity_name)}\n"
        f"{_units_clause(entity_name, include_allowed_units)}\n"
        f"Write the number using digits, with no thousands separators and no "
        f"trailing zeros. Do not write any other words, units, symbols, or "
        f"explanation. If the {label} is not visible in the image, answer with "
        f"nothing at all."
    )
    return [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": text.strip()}],
    }]


def prompt_v2_terse(
    entity_name: str,
    include_allowed_units: bool = True,
    **_: Any,
) -> list[dict[str, Any]]:
    """Minimal-token variant, for an ablation against ``prompt_v1``.

    Shorter prompts leave more of the (tight) context budget for vision tokens
    and train marginally faster. Whether the lost instruction detail costs
    accuracy is an empirical question -- hence the variant.
    """
    label = _entity_label(entity_name)
    units = sorted(allowed_units(entity_name))
    units_text = f" Unit from: {', '.join(units)}." if include_allowed_units and units else ""
    text = f"{label}? Answer as '<number> <unit>' only.{units_text} Empty if absent."
    return [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": text}],
    }]


def prompt_v3_system(
    entity_name: str,
    include_allowed_units: bool = True,
    **_: Any,
) -> list[dict[str, Any]]:
    """``prompt_v1`` preceded by a system turn, for an ablation."""
    messages = [{"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_TEXT}]}]
    messages.extend(prompt_v1(entity_name, include_allowed_units))
    return messages


TEMPLATES: dict[str, PromptFn] = {
    "prompt_v1": prompt_v1,
    "prompt_v2_terse": prompt_v2_terse,
    "prompt_v3_system": prompt_v3_system,
}


def get_template(name: str) -> PromptFn:
    """Look up a template by name, failing loudly on a typo in the config."""
    try:
        return TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt template {name!r}. Available: {sorted(TEMPLATES)}"
        ) from exc


def build_messages(
    entity_name: str,
    template: str = "prompt_v1",
    include_allowed_units: bool = True,
    answer: str | None = None,
) -> list[dict[str, Any]]:
    """Build the message list, optionally appending the assistant target.

    ``answer`` is the raw ``entity_value`` (unmodified -- the metric compares
    against the raw label, so training on anything else would teach the model
    the wrong surface form). Pass ``None`` at inference time.
    """
    messages = get_template(template)(entity_name, include_allowed_units=include_allowed_units)
    if answer is not None:
        messages = messages + [{
            "role": "assistant",
            "content": [{"type": "text", "text": str(answer).strip()}],
        }]
    return messages


def render_for_training(
    processor: Any,
    entity_name: str,
    answer: str,
    template: str = "prompt_v1",
    include_allowed_units: bool = True,
) -> tuple[str, str]:
    """Return ``(prompt_text, full_text)`` for completion-only loss masking.

    ``prompt_text`` ends with the assistant generation header; ``full_text``
    continues into the target. Tokenising both and diffing their lengths gives
    the boundary at which labels switch from ``-100`` to real token ids, which
    is how the loss is restricted to completion tokens.
    """
    prompt_messages = build_messages(entity_name, template, include_allowed_units)
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + str(answer).strip() + _eos_suffix(processor)
    return prompt_text, full_text


def render_for_inference(
    processor: Any,
    entity_name: str,
    template: str = "prompt_v1",
    include_allowed_units: bool = True,
) -> str:
    """Prompt text ending with the assistant generation header."""
    messages = build_messages(entity_name, template, include_allowed_units)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _eos_suffix(processor: Any) -> str:
    """Chat-template end-of-turn marker, so the model learns to stop.

    Without this the model never emits a stop token and generation runs to
    ``max_new_tokens``, trailing junk after an otherwise correct answer.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    eos = getattr(tokenizer, "eos_token", None)
    return eos if eos else "<|im_end|>"


__all__ = [
    "TEMPLATES", "get_template", "build_messages", "render_for_training",
    "render_for_inference", "prompt_v1", "prompt_v2_terse", "prompt_v3_system",
]
