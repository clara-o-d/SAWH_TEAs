"""Shared visual styling for salt types in plots and maps."""

from __future__ import annotations

SALT_MARKERS: dict[str, str] = {
    "LiCl": "o",
    "NaCl": "s",
    "CaCl2": "^",
    "MgCl2": "D",
    "none": "x",
}

DEFAULT_SALT_MARKER = "P"
