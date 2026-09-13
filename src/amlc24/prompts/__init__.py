"""Versioned prompt templates, selected by config for ablation."""

from .templates import (
    TEMPLATES,
    build_messages,
    get_template,
    render_for_inference,
    render_for_training,
)

__all__ = [
    "TEMPLATES", "get_template", "build_messages",
    "render_for_training", "render_for_inference",
]
