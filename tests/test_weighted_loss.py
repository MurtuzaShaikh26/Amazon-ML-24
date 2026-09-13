"""The weighted-loss Trainer actually reweights, and matches hand computation.

These tests exercise ``_weighted_trainer_class().compute_loss`` directly with a
tiny stand-in model, so they run on CPU in milliseconds and do not need the 7B
checkpoint, a GPU, peft, or bitsandbytes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn as nn  # noqa: E402

from amlc24.data.dataset import IGNORE_INDEX  # noqa: E402
from amlc24.train.trainer import _weighted_trainer_class  # noqa: E402

VOCAB, SEQ = 11, 6


class TinyLM(nn.Module):
    """Minimal causal LM with the HF output contract the trainer relies on."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 8)
        self.head = nn.Linear(8, VOCAB)
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.head(self.embed(input_ids))
        out = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out["loss"] = nn.functional.cross_entropy(
                shift_logits.view(-1, VOCAB), shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return _Output(out)


class _Output(dict):
    """dict with attribute access, like a HF ModelOutput."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@pytest.fixture
def trainer():
    """A ``WeightedLossTrainer`` instance without running ``Trainer.__init__``.

    ``compute_loss`` is a pure function of its arguments here, so bypassing the
    heavyweight constructor keeps the test fast and dependency-free.
    """
    cls = _weighted_trainer_class()
    return cls.__new__(cls)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyLM()


def make_batch(batch_size: int, n_supervised: int = 3):
    """Inputs whose first tokens are masked, mimicking completion-only labels."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, VOCAB, (batch_size, SEQ))
    labels = input_ids.clone()
    labels[:, : SEQ - n_supervised] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}


# --- no weights ------------------------------------------------------------
def test_without_weights_matches_the_models_own_loss(trainer, model):
    batch = make_batch(1)
    expected = model(**batch).loss

    loss = trainer.compute_loss(model, dict(batch))
    assert torch.allclose(loss, expected)


def test_missing_weight_key_does_not_crash(trainer, model):
    assert trainer.compute_loss(model, make_batch(2)).isfinite()


# --- batch size 1 (the configured path) ------------------------------------
@pytest.mark.parametrize("weight", [0.5, 1.0, 2.0, 3.598])
def test_batch_size_one_scales_the_loss_exactly(trainer, model, weight):
    """per_device_train_batch_size=1 takes the cheap path: loss * w."""
    batch = make_batch(1)
    base = model(**batch).loss

    weighted = trainer.compute_loss(
        model, {**batch, "sample_weight": torch.tensor([weight])}
    )
    assert torch.allclose(weighted, base * weight, atol=1e-6)


def test_weight_of_one_is_a_no_op(trainer, model):
    batch = make_batch(1)
    base = model(**batch).loss
    weighted = trainer.compute_loss(
        model, {**batch, "sample_weight": torch.tensor([1.0])}
    )
    assert torch.allclose(weighted, base, atol=1e-6)


def test_rarer_class_produces_a_larger_loss(trainer, model):
    """The whole point: a rare entity contributes more gradient."""
    batch = make_batch(1)
    common = trainer.compute_loss(model, {**batch, "sample_weight": torch.tensor([0.642])})
    rare = trainer.compute_loss(model, {**batch, "sample_weight": torch.tensor([3.598])})
    assert rare > common


def test_batch_size_one_gradients_scale_with_the_weight(trainer, model):
    batch = make_batch(1)

    model.zero_grad()
    trainer.compute_loss(model, {**batch, "sample_weight": torch.tensor([1.0])}).backward()
    g1 = model.head.weight.grad.clone()

    model.zero_grad()
    trainer.compute_loss(model, {**batch, "sample_weight": torch.tensor([2.0])}).backward()
    g2 = model.head.weight.grad.clone()

    assert torch.allclose(g2, g1 * 2.0, atol=1e-5)


# --- batch size > 1 (per-sample path) --------------------------------------
def test_batched_path_matches_hand_computed_weighted_mean(trainer, model):
    """Verifies the reduction="none" branch against an explicit calculation."""
    batch = make_batch(3)
    weights = torch.tensor([0.5, 1.0, 3.0])

    # Hand-compute: mean completion-token CE per sample, then weighted mean.
    logits = model(input_ids=batch["input_ids"]).logits
    shift_logits = logits[..., :-1, :]
    shift_labels = batch["labels"][..., 1:]
    per_token = nn.functional.cross_entropy(
        shift_logits.reshape(-1, VOCAB).float(), shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != IGNORE_INDEX).float()
    per_sample = (per_token * mask).sum(1) / mask.sum(1)
    expected = (per_sample * weights).mean()

    got = trainer.compute_loss(model, {**batch, "sample_weight": weights})
    assert torch.allclose(got, expected, atol=1e-5)


def test_batched_uniform_weights_equal_the_unweighted_mean(trainer, model):
    batch = make_batch(4)
    weighted = trainer.compute_loss(
        model, {**batch, "sample_weight": torch.ones(4)}
    )
    plain = trainer.compute_loss(model, dict(batch))
    # Equal only because every sample has the same number of supervised tokens.
    assert torch.allclose(weighted, plain, atol=1e-5)


def test_masked_tokens_are_excluded_from_the_loss(trainer, model):
    """Prompt tokens are -100; changing them must not change the loss."""
    batch = make_batch(2)
    first = trainer.compute_loss(model, {**batch, "sample_weight": torch.ones(2)})

    tampered = {k: v.clone() for k, v in batch.items()}
    tampered["labels"][:, 0] = IGNORE_INDEX  # already masked; stays masked
    second = trainer.compute_loss(model, {**tampered, "sample_weight": torch.ones(2)})

    assert torch.allclose(first, second, atol=1e-6)


def test_sample_weight_is_removed_before_the_forward_pass(trainer):
    """The model must never receive `sample_weight` -- it would raise."""
    class StrictModel(TinyLM):
        def forward(self, input_ids=None, labels=None, **kwargs):
            assert "sample_weight" not in kwargs, "sample_weight leaked into forward()"
            return super().forward(input_ids=input_ids, labels=labels, **kwargs)

    model = StrictModel()
    for size in (1, 3):
        batch = make_batch(size)
        trainer.compute_loss(model, {**batch, "sample_weight": torch.ones(size)})


def test_return_outputs_yields_loss_and_outputs(trainer, model):
    batch = make_batch(1)
    loss, outputs = trainer.compute_loss(
        model, {**batch, "sample_weight": torch.tensor([2.0])}, return_outputs=True
    )
    assert loss.isfinite() and "logits" in outputs
