#!/usr/bin/env python3
"""Köppen-Geiger class for every mapped two-stage cell -> analysis/two_stage/koppen.csv.

The figures group locations by these instead of the pipeline's rule-labelled regimes
(climate.label_regimes), which only steer the physics campaign's oversampling. The
classes come from the global 100 arc-second Köppen-Geiger raster bundled with kgcpy
(Rubel et al. 2017, doi:10.1127/metz/2016/0816), looked up at each cell's lat/lon.
Köppen needs monthly precipitation, which the weather cache does not have, so the
classes are looked up rather than computed. Needs kgcpy (pip install kgcpy) only to
regenerate the CSV:

    python analysis/two_stage/koppen_labels.py
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
# Pixel value -> class, kgcpy's kg_zoneNum.csv. 32 is ocean.
CLASSES = ("Af", "Am", "As", "Aw", "BSh", "BSk", "BWh", "BWk", "Cfa", "Cfb", "Cfc", "Csa", "Csb", "Csc", "Cwa",
           "Cwb", "Cwc", "Dfa", "Dfb", "Dfc", "Dfd", "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "EF", "ET")
OCEAN = 32
# Main groups, with B split into desert and steppe: the two arid types differ most for SAWH.
GROUPS = {"A": "tropical", "BW": "desert", "BS": "steppe", "C": "temperate", "D": "continental", "E": "polar/alpine",
          "o": "ocean"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, default=_REPO / "sawh_bayesopt" / "outputs" / "two_stage" / "main" /
                    "maps_daily_hindsight.csv", help="Any maps_<mode>.csv: every cell with its lat/lon.")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "koppen.csv")
    args = ap.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    with resources.files("kgcpy").joinpath("kmz_int_reshape.png").open("rb") as fh:
        raster = np.asarray(Image.open(fh))
    cells = pd.read_csv(args.maps, usecols=["cell", "lat", "lon"])
    ny, nx = raster.shape
    # Same pixel convention as kgcpy.lookupCZ.
    x = np.clip(np.round((cells.lon.to_numpy() + 180) * nx / 360 - 0.5).astype(int), 0, nx - 1)
    y = np.clip(np.round((90 - cells.lat.to_numpy()) * ny / 180 - 0.5).astype(int), 0, ny - 1)
    codes = raster[y, x].astype(int)
    # Coastal cells can land on an ocean pixel: take the commonest land class within a
    # widening window (~3 km per pixel), up to ~120 km for small islands.
    for i in np.flatnonzero(codes == OCEAN):
        for r in (1, 2, 4, 10, 40):
            win = raster[max(y[i] - r, 0):y[i] + r + 1, max(x[i] - r, 0):x[i] + r + 1].ravel()
            land = win[win != OCEAN]
            if land.size:
                codes[i] = np.bincount(land).argmax()
                break
    # What is left is open ocean in the weather cache itself (elevation 0, mid-Atlantic and
    # mid-Indian Ocean points): labelled "ocean" so the figures can drop them.
    cls = np.array(CLASSES + ("ocean",))[codes - 1]
    cells["koppen"] = cls
    cells["koppen_group"] = [GROUPS[c[:2] if c[0] == "B" else c[0]] for c in cls]
    cells.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(cells)} cells")
    print(cells.koppen_group.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
