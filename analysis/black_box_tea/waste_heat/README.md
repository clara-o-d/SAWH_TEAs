# Waste-heat SAWH — black-box TEA

Excel-based techno-economic analysis for waste-heat-driven atmospheric water harvesting. This is a **simplified black-box model**: you supply daily water yield (from the lumped simulation or measurements) and the workbook computes capital and operating costs and LCOW.

## Workbook

Open [`waste_heat_sawh_tea.xlsx`](waste_heat_sawh_tea.xlsx) in Excel or LibreOffice Calc.

| Sheet | Purpose |
|-------|---------|
| **Inputs** | Editable assumptions (yellow cells) and derived quantities |
| **CAPEX** | Device bill of materials, installed capital, annualized CAPEX (CRF) |
| **Parasitic electricity** | Shaft power, motor efficiency, and annual grid cost for pumps and controls |
| **OPEX** | Fixed and variable operating costs per m² footprint |
| **LCOW** | Summary, LCOW (USD/m³), and cost breakdown by segment |

## Costing

Hardware **CAPEX** uses midpoint estimates from the two-bed vacuum patent BOM (transfer pump, vacuum pump, chambers, valves, condenser, coolant loop, water pump, controller/sensors, purge pump, and structural hardware). **OPEX** includes hydrogel replacement (LiCl hydrogel cost equation, same structure as `solar_black_box`) and **parasitic grid electricity** for pumps and controls (shaft power ÷ motor efficiency × operating hours). Supplemental electric desorption heat is optional and defaults to zero when waste heat supplies desorption.

Default daily yield and thermal efficiency come from the lumped datacenter-baseline profile (`run_waste_heat_sim.py --daily`).

## Regenerate

```bash
pip install openpyxl
python scripts/build_tea_workbook.py
```
