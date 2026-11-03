"""Population of every cell in data/koppen.csv -> data/population.csv (lat, lon, population).

One-off, from GHS-POP R2023A, 2020 epoch, 30 arcsec, WGS84 (JRC; Schiavina et al. 2023):
  https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/
  GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip

    python3 scripts/build_population.py path/to/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif

Needs rasterio (not a package dependency: this runs once and its output is committed).
The raster is summed to 0.1 deg, then every 0.1 deg pixel goes to its nearest cell on the
sphere: the cells are a ~1 deg grid, so each holds the people of its Voronoi region. Pixels
more than MAX_DEG from every cell (small islands with no cell) are dropped and reported.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data"
BLOCK = 12  # 30 arcsec x 12 = 0.1 deg
MAX_DEG = 1.5


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


cells = pd.read_csv(DATA / "koppen.csv")
tree = cKDTree(unit(cells.lat.to_numpy(), cells.lon.to_numpy()))
max_chord = 2 * np.sin(np.radians(MAX_DEG) / 2)
pop, dropped, total = np.zeros(len(cells)), 0.0, 0.0
with rasterio.open(sys.argv[1]) as src:
    t = src.transform
    rows = 10 * BLOCK * 10  # 10 deg strips
    for r0 in range(0, src.height - src.height % BLOCK, rows):
        n = min(rows, src.height - r0) // BLOCK * BLOCK
        a = src.read(1, window=Window(0, r0, src.width // BLOCK * BLOCK, n)).astype(np.float64)
        a[a < 0] = 0.0  # nodata
        a = a.reshape(n // BLOCK, BLOCK, -1, BLOCK).sum(axis=(1, 3))
        i, j = np.nonzero(a)
        lat = t.f + t.e * (r0 + (i + 0.5) * BLOCK)
        lon = t.c + t.a * (j + 0.5) * BLOCK
        dist, idx = tree.query(unit(lat, lon))
        v = a[i, j]
        ok = dist <= max_chord
        np.add.at(pop, idx[ok], v[ok])
        dropped += v[~ok].sum()
        total += v.sum()
        print(f"rows {r0}-{r0 + n}: {total / 1e9:.3f} B so far", flush=True)
print(f"total {total / 1e9:.3f} B; assigned {pop.sum() / 1e9:.3f} B; dropped {dropped / 1e6:.1f} M "
      f"(> {MAX_DEG} deg from any cell)")
pd.DataFrame({"lat": cells.lat, "lon": cells.lon, "population": pop.round(1)}).to_csv(DATA / "population.csv", index=False)
print(f"wrote {DATA / 'population.csv'}")
