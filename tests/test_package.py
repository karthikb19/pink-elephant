"""Smoke tests for the package layout."""

import pink_elephant


def test_package_imports() -> None:
    assert pink_elephant.__doc__ is not None
