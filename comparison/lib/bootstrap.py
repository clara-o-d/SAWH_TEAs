"""sys.path wiring for the four SAWH device packages.

All four packages are ``pip install -e``'d into the active environment
(``solar-lumped-sawh``, ``waste-heat-lumped-sawh``,
``waste-heat-cycle-lumped-sawh``, ``waste-heat-cycle-lumped-no-loop-sawh``),
so ``import solar_lumped`` / ``import waste_heat_lumped`` / etc. work with no
path surgery. This module still adds ``src/`` fallbacks (defensive, in case a
package is ever run outside its editable install) and, unconditionally, adds
each package's ``scripts/`` directory to ``sys.path`` so that adapters can
import the *scripts* (``run_solar_sim``, ``run_waste_heat_sim``) which are
not part of the installed package.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PACKAGE_DIRS: dict[str, str] = {
    "solar_lumped": "solar_lumped",
    "waste_heat_lumped": "waste-heat_lumped",
    "waste_heat_cycle_lumped": "waste-heat_cycle_lumped",
    "waste_heat_cycle_lumped_no_loop": "waste-heat_cycle_lumped_no_loop",
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
    """Add one package's ``scripts/`` dir to ``sys.path`` (module-name import).

    Deliberately *not* called for all four packages in :func:`bootstrap` —
    several packages ship scripts with identical filenames (e.g.
    ``run_waste_heat_cycle_sim.py``, ``parameter_sweep.py``,
    ``npv_heatmap.py`` are duplicated between ``waste-heat_cycle_lumped`` and
    ``waste-heat_cycle_lumped_no_loop``), so blanket-adding every package's
    ``scripts/`` dir would create a real risk of importing the wrong module.
    Callers should only request the specific package(s) whose script modules
    they actually need to import (currently: ``solar_lumped``'s
    ``run_solar_sim`` and ``waste-heat_lumped``'s ``run_waste_heat_sim`` — the
    multi-cycle packages are driven directly via ``DeviceConfig`` +
    ``ode_system.run_daily_operation``, never via their scripts).
    """
    scripts_dir = REPO_ROOT / repo_dir_name / "scripts"
    if scripts_dir.is_dir() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def bootstrap() -> None:
    """Make all four packages importable via ``import <package>``.

    Idempotent; safe to call multiple times (e.g. once per adapter module).
    Does NOT add any package's ``scripts/`` dir — see
    :func:`ensure_scripts_on_path` for that (opt-in, per package).
    """
    for import_name, repo_dir_name in PACKAGE_DIRS.items():
        _ensure_src_on_path(import_name, repo_dir_name)


bootstrap()
