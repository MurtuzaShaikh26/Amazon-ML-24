"""Top-level entry points. Notebooks call these and nothing else.

``run_finetune`` is imported lazily because it pulls in torch; ``run_eda`` is
CPU-only and safe to import eagerly.
"""

from .run_eda import run_eda

__all__ = ["run_eda", "run_finetune"]


def __getattr__(name):
    if name == "run_finetune":
        from .run_finetune import run_finetune

        return run_finetune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
