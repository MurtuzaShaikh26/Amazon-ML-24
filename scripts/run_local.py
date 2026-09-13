#!/usr/bin/env python
"""Local CLI for the pipelines -- the same entry points the notebooks call.

Examples::

    python scripts/run_local.py eda
    python scripts/run_local.py splits
    python scripts/run_local.py train --config run001_qwen2vl_8bit_10k
    python scripts/run_local.py train --smoke      # 32 rows, 1 epoch
    python scripts/run_local.py leaderboard

``train`` needs a GPU. On a machine without one, ``eda`` and ``splits`` still
work and are the useful things to do locally before pushing to Kaggle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlc24.config import load_config  # noqa: E402
from amlc24.data.load import load_train  # noqa: E402
from amlc24.data.splits import get_or_create_split  # noqa: E402
from amlc24.logging_utils import get_logger, setup_logging  # noqa: E402
from amlc24.paths import describe, ensure_dirs  # noqa: E402
from amlc24.pipeline.run_eda import run_eda  # noqa: E402
from amlc24.results.tracker import init_leaderboard, read_leaderboard  # noqa: E402


def _fmt3(value: float) -> str:
    """pandas >= 2 requires float_format to be a callable."""
    return f"{value:.3f}"


def _fmt4(value: float) -> str:
    """pandas >= 2 requires float_format to be a callable."""
    return f"{value:.4f}"

logger = get_logger(__name__)


def cmd_eda(args: argparse.Namespace) -> int:
    result = run_eda(plots=not args.no_plots)
    profile = result["profile"]
    print(f"\nProfiled {result['n_rows']:,} rows -> {result['out_dir']}\n")
    print("=== entity_name distribution ===")
    print(profile["entity_distribution"].to_string(index=False))
    print("\n=== format audit (drives post-processing) ===")
    print(profile["format_audit"].to_string(index=False))
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    train_all = load_train()
    split = get_or_create_split(
        train_all,
        seed=int(cfg.get("seed", 42)),
        eval_size=int(cfg.get("data", {}).get("eval_size", 5000)),
        train_size=int(cfg.get("data", {}).get("train_size", 10000)),
        path=cfg.get("data", {}).get("split_file"),
    )
    print(f"\neval_5k:      {len(split['eval_5k']):,} rows (FROZEN)")
    print(f"train_subset: {len(split.get('train_subset', [])):,} rows")
    print(f"pool:         {len(split.get('pool', [])):,} rows")
    table = split.get("proportion_table")
    if table is not None and hasattr(table, "to_string"):
        print("\n=== entity_name proportions ===")
        print(table.to_string(index=False, float_format=_fmt3))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from amlc24.pipeline.run_finetune import run_finetune

    result = run_finetune(
        args.config,
        download_images=not args.no_download,
        image_dir=args.image_dir,
        max_eval_rows=32 if args.smoke else args.max_eval_rows,
        skip_training=args.zero_shot,
    )
    print("\n=== F1 ===")
    print(f"  raw  : {result['raw']['f1']:.4f}")
    print(f"  post : {result['post']['f1']:.4f}  ({result['f1_delta']:+.4f})")
    print("\n=== per entity ===")
    print(result["by_entity"].to_string(index=False, float_format=_fmt4))
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    init_leaderboard()
    board = read_leaderboard()
    if board.empty:
        print("No runs recorded yet.")
    else:
        print(board.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eda = sub.add_parser("eda", help="Profile train.csv and save tables/charts")
    p_eda.add_argument("--no-plots", action="store_true")
    p_eda.set_defaults(func=cmd_eda)

    p_split = sub.add_parser("splits", help="Create or verify the frozen split")
    p_split.add_argument("--config", default="run001_qwen2vl_8bit_10k")
    p_split.set_defaults(func=cmd_splits)

    p_train = sub.add_parser("train", help="Run the full fine-tune pipeline (needs a GPU)")
    p_train.add_argument("--config", default="run001_qwen2vl_8bit_10k")
    p_train.add_argument("--image-dir", default=None)
    p_train.add_argument("--no-download", action="store_true")
    p_train.add_argument("--max-eval-rows", type=int, default=None)
    p_train.add_argument("--zero-shot", action="store_true",
                         help="Skip training; evaluate the base model as a baseline")
    p_train.add_argument("--smoke", action="store_true",
                         help="Tiny run to check the plumbing end to end")
    p_train.set_defaults(func=cmd_train)

    p_board = sub.add_parser("leaderboard", help="Print results/leaderboard.csv")
    p_board.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args(argv)
    setup_logging()
    ensure_dirs()
    logger.info("Environment: %s", describe())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
