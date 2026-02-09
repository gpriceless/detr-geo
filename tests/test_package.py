"""Tests for package installation and basic imports."""

from __future__ import annotations

import re


def test_import_detr_geo():
    """Importing detr_geo should succeed."""
    import detr_geo

    assert detr_geo is not None


def test_version_is_semver():
    """detr_geo.__version__ should be a valid semver string."""
    import detr_geo

    version = detr_geo.__version__
    assert isinstance(version, str)
    # Match semver pattern: MAJOR.MINOR.PATCH with optional pre-release
    assert re.match(r"^\d+\.\d+\.\d+", version), f"Version '{version}' is not a valid semver string"


def test_import_detr_geo_class():
    """from detr_geo import DetrGeo should work."""
    from detr_geo import DetrGeo

    assert DetrGeo is not None


def test_import_exceptions():
    """from detr_geo.exceptions import DetrGeoError should work."""
    from detr_geo.exceptions import DetrGeoError

    assert DetrGeoError is not None


def test_no_star_imports():
    """__init__.py should not use star imports."""
    import inspect

    import detr_geo

    source = inspect.getsource(detr_geo)
    assert "import *" not in source, "__init__.py contains star imports"
