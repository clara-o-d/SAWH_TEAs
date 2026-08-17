"""Guards the invariant that docs/parameters.xlsx is the single parameter source.

Two directions, both of which rot silently otherwise:

- every row in the workbook is read by some package (no dead rows accumulating);
- every ``_pv``/``_ev`` name resolves (already a KeyError at import, so it needs no
  test of its own -- importing the packages below is the check).
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from solar_lumped._parameters_xlsx import ECONOMICS, PHYSICS, SALTS

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PACKAGES = ("solar_lumped", "waste-heat", "sawh_bayesopt")

# BOM line items are read by prefix scan, not by row name, so a literal-string search
# will never find them. Their prefixes are checked separately below.
_BOM_PREFIXES = ("BOM: ", "Waste-heat BOM: ")


@pytest.fixture(scope="module")
def source_text() -> str:
    """Every git-tracked .py file in the three packages. Tracked-only on purpose:
    solar_lumped carries a .venv_gpu whose site-packages would both slow this to a crawl
    and match row names by coincidence."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "--", *(f"{p}/**/*.py" for p in _PACKAGES)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git ls-files unavailable: {exc}")
    if not listing:
        pytest.skip("sibling packages not checked out")
    return "\n".join(
        (_REPO_ROOT / rel).read_text(errors="ignore")
        for rel in listing
        if (_REPO_ROOT / rel).is_file()
    )


@pytest.mark.parametrize("sheet_name", ["Physics", "Economics"])
def test_every_workbook_row_is_read_somewhere(sheet_name: str, source_text: str) -> None:
    sheet = PHYSICS if sheet_name == "Physics" else ECONOMICS
    orphans = [
        name
        for name in sheet
        if not name.startswith(_BOM_PREFIXES) and f'"{name}"' not in source_text
    ]
    assert not orphans, (
        f"{sheet_name} rows no source file names: {orphans}. Either wire them up or "
        f"delete them -- an unread row is a parameter that silently disagrees with code."
    )


def test_both_bom_prefixes_are_populated() -> None:
    for prefix in _BOM_PREFIXES:
        assert any(name.startswith(prefix) for name in ECONOMICS), prefix


def test_salts_sheet_is_the_only_salt_price_source() -> None:
    """The Economics sheet used to carry its own 'Salt price, X (c_salt)' rows that had
    drifted from the Salts sheet (LiCl 1.50 vs 0.55). Nothing should reintroduce them."""
    assert not [name for name in ECONOMICS if name.startswith("Salt price,")]
    assert {"LiCl", "NaCl", "CaCl2", "MgCl2"} <= set(SALTS)
    assert not list(_REPO_ROOT.rglob("salt_catalog.csv"))
