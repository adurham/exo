"""Dashboard path resolution utilities.

This module provides functions for locating the dashboard build directory
in various deployment scenarios (development, production, bundled).
"""

import os
import sys
from pathlib import Path
from typing import cast


def find_dashboard() -> Path:
    """Find the dashboard build directory.

    Searches for dashboard assets in the following order:
    1. DASHBOARD_DIR environment variable
    2. Repository root (dashboard/build)
    3. PyInstaller bundle directory

    Returns:
        Path to dashboard build directory.

    Raises:
        FileNotFoundError: If dashboard assets cannot be located.
    """
    dashboard = (
        _find_dashboard_in_env()
        or _find_dashboard_in_repo()
        or _find_dashboard_in_bundle()
    )
    if not dashboard:
        raise FileNotFoundError(
            "Unable to locate dashboard assets - make sure the dashboard has been built, or export DASHBOARD_DIR if you've built the dashboard elsewhere."
        )
    return dashboard


def _find_dashboard_in_env() -> Path | None:
    """Find dashboard via DASHBOARD_DIR environment variable.

    Returns:
        Path if DASHBOARD_DIR is set and valid, None otherwise.
    """
    env = os.environ.get("DASHBOARD_DIR")
    if not env:
        return None
    resolved_env = Path(env).expanduser().resolve()

    return resolved_env


def _find_dashboard_in_repo() -> Path | None:
    """Find dashboard in repository structure.

    Searches parent directories for dashboard/build directory.

    Returns:
        Path to dashboard/build if found, None otherwise.
    """
    current_module = Path(__file__).resolve()
    for parent in current_module.parents:
        build = parent / "dashboard" / "build"
        if build.is_dir() and (build / "index.html").exists():
            return build
    return None


def _find_dashboard_in_bundle() -> Path | None:
    """Find dashboard in PyInstaller bundle.

    Checks if running from a frozen executable (PyInstaller) and
    looks for dashboard in the bundle directory.

    Returns:
        Path to dashboard in bundle if found, None otherwise.
    """
    frozen_root = cast(str | None, getattr(sys, "_MEIPASS", None))
    if frozen_root is None:
        return None
    candidate = Path(frozen_root) / "dashboard"
    if candidate.is_dir():
        return candidate
    return None
