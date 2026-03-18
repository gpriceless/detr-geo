"""Tests for detr_geo.datasets.download_sample().

All network I/O is mocked — these tests never hit the internet.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from detr_geo.datasets import _sha256, download_sample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_tif(path: Path, content: bytes = b"FAKE_GEOTIFF_DATA") -> None:
    """Write *content* to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_assets(tmp_path: Path, sha256: str | None = None) -> dict:
    """Return a minimal asset registry pointing at a local tmp path."""
    fake_content = b"FAKE_GEOTIFF_DATA"
    computed = _sha256_of(fake_content)
    return {
        "image": {
            "url": "https://example.com/sample.tif",
            "sha256": sha256 if sha256 is not None else computed,
            "filename": "sample.tif",
        }
    }


# ---------------------------------------------------------------------------
# _sha256 utility
# ---------------------------------------------------------------------------


class TestSha256:
    def test_correct_digest(self, tmp_path: Path) -> None:
        data = b"hello world"
        p = tmp_path / "file.bin"
        p.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256(p) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert _sha256(p) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# download_sample — happy path
# ---------------------------------------------------------------------------


class TestDownloadSampleHappyPath:
    def test_returns_dict_with_image_key(self, tmp_path: Path) -> None:
        fake_content = b"FAKE_GEOTIFF_DATA"
        assets = _make_assets(tmp_path)

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, fake_content)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            result = download_sample(dest=tmp_path, _assets=assets)

        assert "image" in result
        assert isinstance(result["image"], Path)

    def test_file_written_to_dest(self, tmp_path: Path) -> None:
        fake_content = b"FAKE_GEOTIFF_DATA"
        assets = _make_assets(tmp_path)

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, fake_content)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            result = download_sample(dest=tmp_path, _assets=assets)

        assert result["image"].exists()
        assert result["image"].read_bytes() == fake_content

    def test_path_uses_dest_directory(self, tmp_path: Path) -> None:
        assets = _make_assets(tmp_path)

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            result = download_sample(dest=tmp_path, _assets=assets)

        assert result["image"].parent == tmp_path

    def test_filename_matches_asset_registry(self, tmp_path: Path) -> None:
        assets = _make_assets(tmp_path)

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            result = download_sample(dest=tmp_path, _assets=assets)

        assert result["image"].name == "sample.tif"


# ---------------------------------------------------------------------------
# Cache behaviour (no re-download on second call)
# ---------------------------------------------------------------------------


class TestCacheBehaviour:
    def test_no_redownload_when_file_exists(self, tmp_path: Path) -> None:
        fake_content = b"FAKE_GEOTIFF_DATA"
        assets = _make_assets(tmp_path)

        # Pre-populate cache
        cached = tmp_path / "sample.tif"
        _write_fake_tif(cached, fake_content)

        with patch("detr_geo.datasets._download") as mock_dl:
            result = download_sample(dest=tmp_path, _assets=assets)

        mock_dl.assert_not_called()
        assert result["image"] == cached

    def test_force_redownload_even_when_cached(self, tmp_path: Path) -> None:
        fake_content = b"FAKE_GEOTIFF_DATA"
        assets = _make_assets(tmp_path)

        # Pre-populate cache
        cached = tmp_path / "sample.tif"
        _write_fake_tif(cached, fake_content)

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, fake_content)

        with patch("detr_geo.datasets._download", side_effect=fake_download) as mock_dl:
            download_sample(dest=tmp_path, force=True, _assets=assets)

        mock_dl.assert_called_once()

    def test_second_call_uses_cache(self, tmp_path: Path) -> None:
        fake_content = b"FAKE_GEOTIFF_DATA"
        assets = _make_assets(tmp_path)
        call_count = 0

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            nonlocal call_count
            call_count += 1
            _write_fake_tif(dest, fake_content)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            download_sample(dest=tmp_path, _assets=assets)
            download_sample(dest=tmp_path, _assets=assets)

        assert call_count == 1


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------


class TestChecksumVerification:
    def test_valid_checksum_passes(self, tmp_path: Path) -> None:
        fake_content = b"EXACT_CONTENT"
        correct_sha = _sha256_of(fake_content)
        assets = _make_assets(tmp_path, sha256=correct_sha)

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, fake_content)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            result = download_sample(dest=tmp_path, _assets=assets)

        assert result["image"].exists()

    def test_bad_checksum_raises_value_error(self, tmp_path: Path) -> None:
        assets = _make_assets(tmp_path, sha256="a" * 64)  # wrong digest

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, b"DIFFERENT_CONTENT")

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            with pytest.raises(ValueError, match="Checksum mismatch"):
                download_sample(dest=tmp_path, _assets=assets)

    def test_bad_checksum_removes_corrupted_file(self, tmp_path: Path) -> None:
        assets = _make_assets(tmp_path, sha256="b" * 64)  # wrong digest
        corrupt_file = tmp_path / "sample.tif"

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, b"CORRUPTED")

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            with pytest.raises(ValueError):
                download_sample(dest=tmp_path, _assets=assets)

        assert not corrupt_file.exists()

    def test_placeholder_sha256_skips_verification(self, tmp_path: Path) -> None:
        """PLACEHOLDER sha256 should not trigger checksum verification."""
        assets = _make_assets(tmp_path, sha256="PLACEHOLDER_UPDATE_AFTER_RELEASE")

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, b"ANY_CONTENT")

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            result = download_sample(dest=tmp_path, _assets=assets)

        assert result["image"].exists()


# ---------------------------------------------------------------------------
# Progress bar flag
# ---------------------------------------------------------------------------


class TestProgressBarFlag:
    def test_show_progress_passed_to_download(self, tmp_path: Path) -> None:
        fake_content = b"FAKE_GEOTIFF_DATA"
        assets = _make_assets(tmp_path)

        calls: list[bool] = []

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            calls.append(show_progress)
            _write_fake_tif(dest, fake_content)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            download_sample(dest=tmp_path, show_progress=False, _assets=assets)

        assert calls == [False]

    def test_show_progress_true_by_default(self, tmp_path: Path) -> None:
        fake_content = b"FAKE_GEOTIFF_DATA"
        assets = _make_assets(tmp_path)

        calls: list[bool] = []

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            calls.append(show_progress)
            _write_fake_tif(dest, fake_content)

        with patch("detr_geo.datasets._download", side_effect=fake_download):
            download_sample(dest=tmp_path, _assets=assets)

        assert calls == [True]


# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------


class TestDefaultCacheDir:
    def test_default_dest_is_home_cache(self, tmp_path: Path) -> None:
        """When dest is not specified, files go to ~/.cache/detr-geo/sample/."""
        fake_content = b"FAKE_GEOTIFF_DATA"
        fake_cache = tmp_path / ".cache" / "detr-geo" / "sample"
        assets = _make_assets(fake_cache)

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            _write_fake_tif(dest, fake_content)

        with (
            patch("detr_geo.datasets._download", side_effect=fake_download),
            patch("detr_geo.datasets._DEFAULT_CACHE_DIR", fake_cache),
        ):
            result = download_sample(_assets=assets)

        assert result["image"].parent == fake_cache


# ---------------------------------------------------------------------------
# Public API surface — importable from top-level package
# ---------------------------------------------------------------------------


class TestPublicApiSurface:
    def test_importable_from_datasets_module(self) -> None:
        from detr_geo.datasets import download_sample as ds

        assert callable(ds)

    def test_importable_from_top_level_package(self) -> None:
        from detr_geo import download_sample as ds

        assert callable(ds)

    def test_in_all(self) -> None:
        import detr_geo

        assert "download_sample" in detr_geo.__all__
