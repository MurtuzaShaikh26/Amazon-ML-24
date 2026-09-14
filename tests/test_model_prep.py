"""Regression tests for the T4 OOM: frozen weights must NOT be upcast to fp32."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from amlc24.models.qwen2vl import prepare_quantized_for_training  # noqa: E402


class FakeVLM(nn.Module):
    """fp16 stand-in exposing the two HF hooks the prep function calls."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 16).half()
        self.visual = nn.Linear(16, 16).half()
        self.lm_head = nn.Linear(16, 50).half()
        self.ckpt_kwargs = None
        self.input_grads = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.ckpt_kwargs = gradient_checkpointing_kwargs

    def enable_input_require_grads(self):
        self.input_grads = True


def test_frozen_weights_stay_fp16():
    """peft's prepare_model_for_kbit_training cast these to fp32 and OOM'd."""
    model = prepare_quantized_for_training(FakeVLM())
    assert {p.dtype for p in model.parameters()} == {torch.float16}


def test_all_base_parameters_are_frozen():
    model = prepare_quantized_for_training(FakeVLM())
    assert not any(p.requires_grad for p in model.parameters())


def test_gradient_checkpointing_is_non_reentrant_and_inputs_require_grad():
    model = prepare_quantized_for_training(FakeVLM(), use_gradient_checkpointing=True)
    assert model.ckpt_kwargs == {"use_reentrant": False}
    assert model.input_grads, "LoRA gets no gradient through checkpointing otherwise"


def test_checkpointing_can_be_disabled():
    model = prepare_quantized_for_training(FakeVLM(), use_gradient_checkpointing=False)
    assert model.ckpt_kwargs is None and not model.input_grads
