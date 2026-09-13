"""Prompt templates produce the expected Qwen2-VL message structure."""

from __future__ import annotations

import pytest

from amlc24.postprocess.units import allowed_units
from amlc24.prompts.templates import (
    TEMPLATES,
    build_messages,
    get_template,
    prompt_v1,
    render_for_inference,
    render_for_training,
)


class FakeProcessor:
    """Minimal stand-in for ``AutoProcessor``'s chat-template behaviour."""

    class _Tok:
        eos_token = "<|im_end|>"

    tokenizer = _Tok()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            body = "".join(
                "<|vision_start|><|image_pad|><|vision_end|>" if c["type"] == "image"
                else c["text"]
                for c in message["content"]
            )
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


# --- structure -------------------------------------------------------------
def test_prompt_v1_returns_a_single_user_message():
    messages = prompt_v1("item_weight")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_content_is_an_image_placeholder_followed_by_text():
    content = prompt_v1("item_weight")[0]["content"]
    assert [c["type"] for c in content] == ["image", "text"]
    assert content[0] == {"type": "image"}, "image placeholder carries no data"
    assert isinstance(content[1]["text"], str) and content[1]["text"]


def test_prompt_names_the_entity_in_readable_form():
    text = prompt_v1("item_weight")[0]["content"][1]["text"]
    assert "item weight" in text
    assert "item_weight" not in text, "underscores should be humanised"


def test_prompt_demands_the_exact_output_format():
    text = prompt_v1("item_weight")[0]["content"][1]["text"]
    assert "<number> <unit>" in text
    assert "34 gram" in text, "an example anchors the format"
    assert "nothing" in text.lower(), "must instruct abstention when not visible"


# --- allowed units in the prompt -------------------------------------------
def test_allowed_units_are_listed_when_enabled():
    text = prompt_v1("item_weight", include_allowed_units=True)[0]["content"][1]["text"]
    for unit in allowed_units("item_weight"):
        assert unit in text, f"{unit} missing from the prompt"


def test_allowed_units_are_omitted_when_disabled():
    text = prompt_v1("item_weight", include_allowed_units=False)[0]["content"][1]["text"]
    assert "microgram" not in text


def test_prompt_lists_only_that_entitys_units():
    text = prompt_v1("voltage", include_allowed_units=True)[0]["content"][1]["text"]
    assert "volt" in text
    assert "gram" not in text, "units from other entities must not leak in"


@pytest.mark.parametrize("entity", [
    "item_weight", "width", "height", "depth", "voltage", "wattage",
    "item_volume", "maximum_weight_recommendation",
])
def test_every_entity_builds_a_valid_prompt(entity):
    content = prompt_v1(entity)[0]["content"]
    assert content[0]["type"] == "image" and content[1]["text"].strip()


# --- registry --------------------------------------------------------------
def test_registry_contains_prompt_v1():
    assert "prompt_v1" in TEMPLATES
    assert get_template("prompt_v1") is prompt_v1


def test_unknown_template_raises_and_lists_options():
    with pytest.raises(KeyError, match="prompt_v1"):
        get_template("prompt_v999")


def test_all_registered_templates_share_the_message_contract():
    for name, fn in TEMPLATES.items():
        messages = fn("item_weight")
        assert messages[-1]["role"] == "user", name
        types = [c["type"] for c in messages[-1]["content"]]
        assert "image" in types and "text" in types, name


def test_v3_prepends_a_system_turn():
    messages = TEMPLATES["prompt_v3_system"]("item_weight")
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"


# --- training vs inference -------------------------------------------------
def test_build_messages_appends_the_assistant_target():
    messages = build_messages("item_weight", answer="34 gram")
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"][0]["text"] == "34 gram"


def test_build_messages_omits_the_assistant_turn_at_inference():
    assert build_messages("item_weight", answer=None)[-1]["role"] == "user"


def test_target_is_the_raw_entity_value_unmodified():
    """The metric compares against the raw label, so training must too."""
    for label in ("2.56 ounce", "1000 watt", "8 fluid ounce"):
        messages = build_messages("item_weight", answer=label)
        assert messages[-1]["content"][0]["text"] == label


def test_render_for_training_splits_prompt_from_completion():
    # A value distinct from the template's own format example, so the assertion
    # below really tests for answer leakage rather than matching the example.
    answer = "782.25 kilogram"
    prompt, full = render_for_training(FakeProcessor(), "item_weight", answer)
    assert full.startswith(prompt), "the completion must extend the prompt exactly"
    assert prompt.endswith("<|im_start|>assistant\n")
    assert answer in full[len(prompt):]
    assert answer not in prompt, "the answer must not leak into the prompt"


def test_render_for_training_appends_an_end_of_turn_marker():
    """Without it the model never stops and trails junk after the answer."""
    _, full = render_for_training(FakeProcessor(), "item_weight", "34 gram")
    assert full.rstrip().endswith("<|im_end|>")


def test_render_for_inference_ends_at_the_generation_header():
    text = render_for_inference(FakeProcessor(), "item_weight")
    assert text.endswith("<|im_start|>assistant\n")


def test_rendered_prompt_contains_the_vision_placeholder():
    text = render_for_inference(FakeProcessor(), "item_weight")
    assert "<|image_pad|>" in text
