"""Sample data utilities for detr_geo quick-start demos.

Downloads a small (~10 MB) public-domain NAIP GeoTIFF bundle to a local
cache directory so users can try the library without sourcing their own data.

Example::

    from detr_geo.datasets import download_sample

    paths = download_sample()
    print(paths["image"])   # Path to the cached GeoTIFF
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Asset registry
# ---------------------------------------------------------------------------

# Key: logical asset name → {url, sha256, filename}
# SHA256 is verified after download to ensure data integrity.
#
# NOTE: The sample data is hosted as a GitHub Release asset. When v0.1.0-data
# is published, update _SAMPLE_ASSETS with the actual SHA256 digest of the
# uploaded file.  Run:
#   sha256sum sample_naip.tif
# and paste the result into the "sha256" field below.
_SAMPLE_ASSETS: dict[str, dict[str, str]] = {
    "image": {
        "url": ("https://github.com/gpriceless/detr-geo/releases/download/v0.1.0-data/sample_naip.tif"),
        # 3-band RGB NAIP, ~1m GSD, public domain (USDA)
        "sha256": "PLACEHOLDER_UPDATE_AFTER_RELEASE",
        "filename": "sample_naip.tif",
    },
}

# Default cache location: ~/.cache/detr-geo/sample/
# Override by passing *dest* to download_sample().
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "detr-geo" / "sample"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _TqdmUpTo(tqdm):  # type: ignore[type-arg]
    """tqdm subclass for urllib.request.urlretrieve progress hook."""

    def update_to(self, n_blocks: int = 1, block_size: int = 1, total_size: int = -1) -> None:
        if total_size not in (None, -1):
            self.total = total_size
        self.update(n_blocks * block_size - self.n)


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, show_progress: bool = True) -> None:
    """Download *url* to *dest*, optionally showing a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        with _TqdmUpTo(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=dest.name,
        ) as progress:
            urllib.request.urlretrieve(url, dest, reporthook=progress.update_to)
    else:
        urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_sample(
    dest: str | Path | None = None,
    *,
    force: bool = False,
    show_progress: bool = True,
    _assets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Download a small public-domain NAIP sample for quick-start demos.

    Files are cached in ``~/.cache/detr-geo/sample/`` (or *dest*) and only
    re-downloaded when they are missing or *force=True*.  A SHA-256 checksum
    is verified after every download to guarantee data integrity.

    Args:
        dest: Directory to store downloaded files.  Defaults to
            ``~/.cache/detr-geo/sample/``.
        force: Re-download even if the cached file already exists.
        show_progress: Show a tqdm download progress bar.
        _assets: Override the default asset registry (intended for tests).

    Returns:
        A dict mapping asset names to their local ``Path`` objects.
        Currently contains a single key ``"image"`` pointing to the
        downloaded GeoTIFF.

    Raises:
        urllib.error.URLError: If the download fails (network error, 404, …).
        ValueError: If the downloaded file fails checksum verification.

    Example::

        from detr_geo.datasets import download_sample

        paths = download_sample()
        # {'image': PosixPath('/home/user/.cache/detr-geo/sample/sample_naip.tif')}
    """
    cache_dir = Path(dest) if dest is not None else _DEFAULT_CACHE_DIR
    assets = _assets if _assets is not None else _SAMPLE_ASSETS

    result: dict[str, Path] = {}

    for name, meta in assets.items():
        url: str = meta["url"]
        expected_sha256: str = meta["sha256"]
        filename: str = meta["filename"]
        local_path = cache_dir / filename

        needs_download = force or not local_path.exists()

        if needs_download:
            _download(url, local_path, show_progress=show_progress)

            # Verify checksum only when the value is not a placeholder.
            if not expected_sha256.startswith("PLACEHOLDER"):
                actual = _sha256(local_path)
                if actual != expected_sha256:
                    local_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"Checksum mismatch for {filename}: "
                        f"expected {expected_sha256}, got {actual}. "
                        "The cached file has been removed. Try again or "
                        "file a bug at https://github.com/gpriceless/detr-geo/issues"
                    )

        result[name] = local_path

    return result
