#!/usr/bin/env python
"""Download and resize only the images the frozen split actually needs.

Typical use, once, locally::

    python scripts/download_images.py

That resolves the frozen 5k eval split plus the 10k training subset, collapses
duplicate URLs, and writes ~448px JPEGs into ``images/``. Upload that folder to
Kaggle as a private Dataset afterwards -- Kaggle sessions are ephemeral, and
re-downloading 15k images at the start of every run wastes an hour of the
12-hour budget.

The download is resumable: existing files are skipped, so interrupting and
re-running is free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlc24.config import load_config  # noqa: E402
from amlc24.data.images import (  # noqa: E402
    DEFAULT_THREADS,
    TARGET_LONG_SIDE,
    available_mask,
    download_for_frames,
)
from amlc24.data.load import load_split_frames, load_test, load_train  # noqa: E402
from amlc24.data.splits import get_or_create_split  # noqa: E402
from amlc24.logging_utils import get_logger, setup_logging  # noqa: E402
from amlc24.paths import IMAGE_DIR, describe, ensure_dirs  # noqa: E402

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="run001_qwen2vl_8bit_10k",
                        help="Config naming the split to download for")
    parser.add_argument("--image-dir", default=None, help="Destination (default: paths.IMAGE_DIR)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--long-side", type=int, default=TARGET_LONG_SIDE,
                        help="Longest side in pixels after resize")
    parser.add_argument("--split", choices=("train_eval", "test", "all"), default="train_eval",
                        help="Which images to fetch (default: the frozen split only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows per frame, for a quick smoke test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    ensure_dirs()
    logger.info("Environment: %s", describe())

    image_dir = Path(args.image_dir) if args.image_dir else IMAGE_DIR
    cfg = load_config(args.config)
    frames = []

    if args.split in ("train_eval", "all"):
        train_all = load_train()
        split = get_or_create_split(
            train_all,
            seed=int(cfg.get("seed", 42)),
            eval_size=int(cfg.get("data", {}).get("eval_size", 5000)),
            train_size=int(cfg.get("data", {}).get("train_size", 10000)),
            path=cfg.get("data", {}).get("split_file"),
        )
        eval_df, train_df = load_split_frames(split, train_all)
        frames.extend([train_df, eval_df])
        logger.info("Need images for %d train + %d eval rows", len(train_df), len(eval_df))

    if args.split in ("test", "all"):
        test_df = load_test()
        frames.append(test_df)
        logger.info("Need images for %d test rows", len(test_df))

    if args.limit:
        frames = [f.head(args.limit) for f in frames]

    report = download_for_frames(
        frames, image_dir=image_dir, threads=args.threads,
        timeout=args.timeout, retries=args.retries, long_side=args.long_side,
    )

    total_rows = sum(len(f) for f in frames)
    all_urls = [u for f in frames for u in f["image_link"].astype(str)]
    have = sum(available_mask(all_urls, image_dir))

    logger.info("=" * 64)
    logger.info("Rows needing an image: %d", total_rows)
    logger.info("Unique URLs:           %d", report.requested)
    logger.info("Available locally:     %d (%.2f%% of rows)",
                have, 100 * have / max(total_rows, 1))
    logger.info("Failed:                %d", report.failed)
    logger.info("Image dir:             %s", image_dir)
    logger.info("=" * 64)
    logger.info(
        "Next: upload %s to Kaggle as a private Dataset and attach it to the "
        "notebook, so future runs skip this step entirely.", image_dir,
    )

    return 0 if report.failed < max(1, report.requested // 10) else 1


if __name__ == "__main__":
    raise SystemExit(main())
