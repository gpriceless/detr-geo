# Contributing to detr-geo

Thanks for your interest in contributing to detr-geo! This is an early-stage project and contributions of all kinds are welcome -- bug reports, feature suggestions, documentation improvements, and code.

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/detr-geo.git
cd detr-geo
```

### 2. Set up the development environment

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with all dependencies + dev tools
pip install -e ".[all]"

# Install linting tools
pip install ruff
```

Note: detr-geo requires GDAL/rasterio. See the [README](README.md#installation) for system-level prerequisites.

### 3. Verify your setup

```bash
# Run the test suite
pytest

# Run linting
ruff check src/ tests/
ruff format --check src/ tests/
```

---

## How to Contribute

### Reporting Bugs

Use the [bug report template](https://github.com/gpriceless/detr-geo/issues/new?template=bug_report.md). Include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (Python version, OS, GPU if relevant)

### Suggesting Features

Use the [feature request template](https://github.com/gpriceless/detr-geo/issues/new?template=feature_request.md). Describe:

- The problem you are trying to solve
- Your proposed solution (if you have one)
- Why this would be useful to other users

### Submitting Code

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b my-feature
   ```

2. **Write your code.** Follow the style guidelines below.

3. **Add tests** for new functionality. Tests live in `tests/` and use pytest:
   ```bash
   pytest tests/
   ```

4. **Run linting** before committing:
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```

5. **Commit** with a clear message describing what and why:
   ```bash
   git commit -m "add support for GeoParquet export"
   ```

6. **Push** and open a pull request against `main`.

---

## Code Style

- **Linting**: [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. The project configuration is in `pyproject.toml`.
- **Type hints**: Encouraged on all public functions. We use `mypy` in CI.
- **Docstrings**: Use Google-style docstrings for public API methods.
- **Imports**: Use absolute imports (`from detr_geo.io import ...`), not relative.
- **Tests**: Use `pytest`. Tests that require optional dependencies (rfdetr, matplotlib, pystac) should use `pytest.importorskip()` to skip gracefully.

---

## Project Structure

```
src/detr_geo/
    core.py          # DetrGeo main class
    io.py            # Raster I/O, band selection, normalization
    crs.py           # CRS utilities
    tiling.py        # Tile generation and NMS
    export.py        # GeoJSON, GeoPackage, Shapefile export
    viz.py           # Visualization (matplotlib, leafmap)
    training.py      # Dataset preparation and training
    exceptions.py    # Exception hierarchy
tests/
    test_core.py     # Core workflow tests
    test_io.py       # I/O tests
    ...
```

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold a welcoming, inclusive environment for everyone.

---

## Questions?

Open a [discussion](https://github.com/gpriceless/detr-geo/discussions) or file an issue. We are happy to help.
