"""Threaded image download with resize-on-save, plus resilient local loading.

Design decisions that matter:

* **Resize during download.** Images are written with the longest side at
  ``TARGET_LONG_SIDE`` (448 px) as JPEG q90. 448 px is the smallest size that
  still exceeds what the processor consumes at ``max_pixels = 256*28*28``
  (~448x448 worth of 28 px patches), so the cap -- not the file -- is what limits
  resolution. This cuts the image dataset roughly 10-20x and removes a large
  CPU decode cost from every training step.
* **Resumable.** Existing files are skipped, so a session that dies halfway
  costs nothing. This is essential on Kaggle, where sessions are ephemeral.
* **Never crash on a missing image.** A dead CDN URL skips the row with a
  warning. Losing a handful of rows is survivable; losing a 9-hour run is not.

Filenames are the sha1 of the URL, so they are stable across machines and
sessions and contain no characters that upset Windows or Kaggle.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..paths import IMAGE_DIR

logger = logging.getLogger(__name__)

TARGET_LONG_SIDE = 448
JPEG_QUALITY = 90
DEFAULT_THREADS = 32
DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (compatible; amlc24-research/1.0)"

# Placeholder used when an image is missing at train/eval time. Grey rather than
# black so it is visually distinct from a genuinely dark product photo.
PLACEHOLDER_SIZE = (64, 64)
PLACEHOLDER_COLOUR = (127, 127, 127)


@dataclass
class DownloadReport:
    """Outcome summary for one download sweep."""

    requested: int = 0
    already_present: int = 0
    downloaded: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def available(self) -> int:
        return self.already_present + self.downloaded

    def as_dict(self) -> dict:
        return {**asdict(self), "available": self.available}


def image_filename(url: str) -> str:
    """Stable, filesystem-safe filename for a URL."""
    digest = hashlib.sha1(str(url).strip().encode("utf-8")).hexdigest()
    return f"{digest}.jpg"


def image_path(url: str, image_dir: Path | None = None) -> Path:
    """Local path where ``url`` is (or would be) cached."""
    return (image_dir or IMAGE_DIR) / image_filename(url)


def _resize_and_save(payload: bytes, dest: Path,
                     long_side: int = TARGET_LONG_SIDE) -> bool:
    """Decode, downscale to ``long_side``, convert to RGB, save as JPEG."""
    from PIL import Image

    try:
        with Image.open(BytesIO(payload)) as img:
            img = img.convert("RGB")
            width, height = img.size
            if max(width, height) > long_side:
                scale = long_side / max(width, height)
                img = img.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.LANCZOS,
                )
            tmp = dest.with_suffix(".tmp")
            img.save(tmp, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        # Atomic-ish rename so an interrupted write never leaves a partial JPEG
        # that a later resumed run would happily skip.
        tmp.replace(dest)
        return True
    except (OSError, ValueError) as exc:
        logger.debug("Decode/resize failed for %s: %s", dest.name, exc)
        dest.with_suffix(".tmp").unlink(missing_ok=True)
        return False


def download_one(
    url: str,
    image_dir: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    long_side: int = TARGET_LONG_SIDE,
) -> tuple[str, str]:
    """Download and cache a single URL. Returns ``(url, status)``.

    Status is one of ``skipped`` (already on disk), ``ok``, or ``failed``.
    """
    dest = image_path(url, image_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return url, "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            request = Request(str(url), headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if _resize_and_save(payload, dest, long_side):
                return url, "ok"
            last_error = ValueError("could not decode image payload")
        except (HTTPError, URLError, OSError, ValueError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code in (403, 404, 410):
                break  # permanent; retrying wastes time
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1) * 0.5, 4.0))

    logger.warning("Download failed after %d attempt(s): %s (%s)", retries, url, last_error)
    return url, "failed"


def download_images(
    urls: Iterable[str],
    image_dir: Path | None = None,
    threads: int = DEFAULT_THREADS,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    long_side: int = TARGET_LONG_SIDE,
    log_every: int = 500,
    failure_log: Path | str | None = None,
) -> DownloadReport:
    """Download many URLs concurrently, skipping files already on disk.

    Duplicate URLs are collapsed first -- one product image commonly backs
    several rows, so the unique count is well below the row count.
    """
    target_dir = Path(image_dir) if image_dir else IMAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    unique = sorted({str(u).strip() for u in urls if u and str(u).strip()})
    report = DownloadReport(requested=len(unique))
    if not unique:
        logger.warning("No URLs to download")
        return report

    logger.info("Downloading %d unique image(s) to %s with %d threads (resize to %dpx)",
                len(unique), target_dir, threads, long_side)
    start = time.time()
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(download_one, url, target_dir, timeout, retries, long_side): url
            for url in unique
        }
        for done, future in enumerate(as_completed(futures), start=1):
            _, status = future.result()
            if status == "ok":
                report.downloaded += 1
            elif status == "skipped":
                report.already_present += 1
            else:
                report.failed += 1
                failures.append(futures[future])

            if done % log_every == 0 or done == len(unique):
                logger.info(
                    "  %d/%d  (new=%d skipped=%d failed=%d)",
                    done, len(unique), report.downloaded,
                    report.already_present, report.failed,
                )

    report.elapsed_seconds = time.time() - start
    logger.info(
        "Download finished in %.1fs: %d new, %d already present, %d failed (%.2f%% miss rate)",
        report.elapsed_seconds, report.downloaded, report.already_present,
        report.failed, 100 * report.failed / max(len(unique), 1),
    )

    if failures:
        log_path = Path(failure_log) if failure_log else target_dir.parent / "download_failures.txt"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(failures), encoding="utf-8")
            logger.warning("Wrote %d failed URL(s) to %s", len(failures), log_path)
        except OSError as exc:
            logger.warning("Could not write failure log: %s", exc)

    return report


def download_for_frames(
    frames: Sequence, image_dir: Path | None = None, **kwargs
) -> DownloadReport:
    """Download only the images referenced by the given DataFrames.

    The eval 5k plus a 10k training subset reference ~15k rows but fewer unique
    images, and that is all we ever need -- the full competition dataset is far
    larger and downloading it would waste hours for no benefit.
    """
    urls: list[str] = []
    for frame in frames:
        if frame is not None and len(frame):
            urls.extend(frame["image_link"].dropna().astype(str).tolist())
    return download_images(urls, image_dir=image_dir, **kwargs)


def load_image(url: str, image_dir: Path | None = None, allow_placeholder: bool = True):
    """Open a cached image as a PIL RGB image.

    Returns a grey placeholder (never ``None``) when the file is missing or
    corrupt and ``allow_placeholder`` is set, so a single bad file cannot abort
    a long training run.
    """
    from PIL import Image

    path = image_path(url, image_dir)
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Missing/corrupt image %s (%s): %s", path.name, url, exc)
        if not allow_placeholder:
            raise
        return Image.new("RGB", PLACEHOLDER_SIZE, PLACEHOLDER_COLOUR)


def available_mask(urls: Sequence[str], image_dir: Path | None = None) -> list[bool]:
    """Per-URL flag for whether a usable cached file exists."""
    target = Path(image_dir) if image_dir else IMAGE_DIR
    return [
        (p := target / image_filename(u)).exists() and p.stat().st_size > 0
        for u in urls
    ]


def filter_to_available(df, image_dir: Path | None = None, drop: bool = True):
    """Drop rows whose image failed to download, logging how many were lost."""
    mask = available_mask(df["image_link"].astype(str).tolist(), image_dir)
    n_missing = mask.count(False)
    if n_missing:
        logger.warning(
            "%d/%d row(s) have no local image (%.2f%%)%s",
            n_missing, len(df), 100 * n_missing / max(len(df), 1),
            "; dropping them" if drop else "; keeping them with placeholders",
        )
    return df[mask].reset_index(drop=True) if drop else df


__all__ = [
    "download_images", "download_one", "download_for_frames", "load_image",
    "image_path", "image_filename", "available_mask", "filter_to_available",
    "DownloadReport", "TARGET_LONG_SIDE", "DEFAULT_THREADS",
]
