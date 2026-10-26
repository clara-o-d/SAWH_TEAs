# Water-source comparison

SAWH against the alternatives it has to beat, on cost per m³.

- `desal_vs_sawh_map.py`: delivered desalination cost (coastal RO plus conveyance, per
  `desalination_cost_methods.tex`, Kocher & Menon 2023) against SAWH LCOW on a world map.
  Writes to `outputs/`. `analysis/two_stage/make_figures.py` reuses its cost model for
  Fig. 10.
- `site_elevation_cache.csv`: cached site elevations for the conveyance term.

`benchmark/tariff/` holds scraped per-country tap-water tariffs, the next comparator.
