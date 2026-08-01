"""sys.path wiring for the four SAWH device packages. All four are pip install -e'd, so the
package imports need no path surgery; this adds src/ fallbacks plus the scripts/ dirs the
adapters import run_solar_sim / run_waste_heat_sim from."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

PACKAGE_DIRS: dict[str, str] = {
    "solar_lumped": "solar_lumped",
    "waste_heat_lumped": "waste-heat/lumped",
    "waste_heat_cycle_lumped": "waste-heat/cycle_lumped",
    "waste_heat_cycle_lumped_no_loop": "waste-heat/cycle_lumped_no_loop",
}


def _ensure_src_on_path(package_import_name: str, repo_dir_name: str) -> None:
    """Fall back to manual ``src/`` path insertion if the editable install is missing."""
    try:
        importlib.import_module(package_import_name)
        return
    except ImportError:
        pass
    src = REPO_ROOT / repo_dir_name / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def ensure_scripts_on_path(repo_dir_name: str) -> None:
    """Add one package's ``scripts/`` dir to sys.path. Deliberately not done for all four in
    :func:`bootstrap` -- the cycle packages ship identically-named scripts, so blanket-adding
    them risks importing the wrong module. Request only the packages you import from."""
    scripts_dir = REPO_ROOT / repo_dir_name / "scripts"
    if scripts_dir.is_dir() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def bootstrap() -> None:
    """Make all four packages importable. Idempotent. Does not add any scripts/ dir -- see
    :func:`ensure_scripts_on_path` for that."""
    for import_name, repo_dir_name in PACKAGE_DIRS.items():
        _ensure_src_on_path(import_name, repo_dir_name)


bootstrap()
