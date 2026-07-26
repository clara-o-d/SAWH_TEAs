# Solar SAWH — black-box TEA

Excel-based techno-economic analysis for passive solar atmospheric water harvesting. This is a **simplified black-box model**: you supply daily water yield (from the lumped simulation, field data, or literature) and the workbook computes capital and operating costs and levelized cost of water (LCOW).

## Workbook

Open [`solar_sawh_tea.xlsx`](solar_sawh_tea.xlsx) in Excel or LibreOffice Calc.

| Sheet | Purpose |
|-------|---------|
| **Inputs** | Editable assumptions (yellow cells) and derived quantities |
| **CAPEX** | Device bill of materials, installed capital, annualized CAPEX (CRF) |
| **OPEX** | Fixed and variable operating costs per m² footprint |
| **LCOW** | Summary, LCOW (USD/m³), and cost breakdown by segment |

## Costing

Formulas mirror `solar_lumped/src/solar_lumped/economics/lcow.py`:

```
LCOW = annual_cost / (utilization_factor × gross_annual_water_m³)

annual_cost = CRF × installed_CAPEX
            + hydrogel_replacement
            + maintenance × installed_CAPEX
            + fixed_energy
            + electricity (optional active heat)
            + extra cycling energy
```

Default economics come from `solar_lumped/src/solar_lumped/data/economics/lcow_economic_params.csv`. Default daily yield (~1.22 kg/m²/d) matches the Cambridge replay profile (2024-06-03, MIT campus weather).

## Regenerate

```bash
pip install openpyxl
python scripts/build_tea_workbook.py
```
