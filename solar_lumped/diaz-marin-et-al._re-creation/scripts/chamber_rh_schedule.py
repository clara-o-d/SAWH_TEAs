"""RH switch times from Díaz-Marín et al. (2024) source-data workbook (Fig. 5 kinetics).

The ``Kinetics (Env chamber)`` sheet records uptake vs time only (no RH column).
Switch times are inferred as the last near-plateau point before sustained desorption
on a lightly smoothed uptake curve. For PAM--LiCl 4 g/g these match the paper's
reported absorption timescales (τ_abs ≈ 2430, 2630, 3200 min) within ~1--2 %.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

_SCRIPT = Path(__file__).resolve()
_DIAZ_DIR = _SCRIPT.parent.parent
ESM_XLSX = _DIAZ_DIR / "reference" / "41467_2024_53291_MOESM3_ESM.xlsx"

# Workbook material label → figure-5 panel key
_MATERIAL_TO_PANEL: dict[str, str] = {
    "PAM-LiCl 4gg": "5c",
    "PAM-LiCl 2gg": "5d",
    "PVA-LiCl 4gg": "5e",
    "PAM-LiCl 4gg 1.5H": "5i",
}


@dataclass(frozen=True, slots=True)
class ChamberRhSchedule:
    """Fixed RH schedule for one 20 → high → 20 % cycle."""

    panel: str
    rh_high_pct: int
    t_high_to_20_min: float
    t_end_min: float
    material: str
    cycle_label: str


def _parse_kinetics_columns(df: pd.DataFrame) -> list[tuple[str, str, int, int]]:
    row0 = df.iloc[0]
    row1 = df.iloc[1]
    materials: list[str | None] = []
    current: str | None = None
    for col in range(df.shape[1]):
        value = row0.iloc[col]
        if pd.notna(value) and str(value).strip():
            current = str(value).strip()
        materials.append(current)

    series: list[tuple[str, str, int, int]] = []
    for col in range(1, df.shape[1], 2):
        material = materials[col]
        cycle = row1.iloc[col]
        if material is None or pd.isna(cycle):
            continue
        series.append((material, str(cycle).strip(), col, col + 1))
    return series


def detect_high_to_20_switch_min(
    time_min: np.ndarray,
    uptake_g_g: np.ndarray,
    *,
    smooth_pts: int = 30,
    plateau_frac: float = 0.90,
    window_pts: int = 100,
) -> float:
    """Return t [min] when chamber RH is stepped from high back to 20 %."""
    t = np.asarray(time_min, dtype=float)
    u = np.asarray(uptake_g_g, dtype=float)
    mask = np.isfinite(t) & np.isfinite(u)
    t, u = t[mask], u[mask]
    if len(t) < window_pts + 2:
        raise ValueError("insufficient kinetics points")

    order = np.argsort(t)
    t, u = t[order], u[order]
    t, uniq_idx = np.unique(t, return_index=True)
    u = u[uniq_idx]
    if len(t) < window_pts + 2:
        raise ValueError("insufficient kinetics points after deduplication")
    u_smooth = uniform_filter1d(u, size=smooth_pts)
    du = np.gradient(u_smooth, t)
    u_peak = float(np.max(u_smooth))

    for i in range(len(t) - window_pts * 2, window_pts, -1):
        if u_smooth[i] < plateau_frac * u_peak:
            continue
        pre_slope = float(np.mean(du[i - window_pts : i]))
        post_slope = float(np.mean(du[i : i + window_pts]))
        if pre_slope > -1e-6 and post_slope < -5e-6:
            return float(t[i])

    return float(t[int(np.argmax(u_smooth))])


def load_chamber_rh_schedules(
    xlsx_path: Path | None = None,
    *,
    panels: frozenset[str] | None = None,
) -> dict[tuple[str, int], ChamberRhSchedule]:
    """Load inferred RH switch times for all figure-5 panels from the ESM workbook."""
    path = ESM_XLSX if xlsx_path is None else Path(xlsx_path)
    df = pd.read_excel(path, sheet_name="Kinetics (Env chamber)", header=None)

    out: dict[tuple[str, int], ChamberRhSchedule] = {}
    for material, cycle_label, t_col, u_col in _parse_kinetics_columns(df):
        panel = _MATERIAL_TO_PANEL.get(material)
        if panel is None:
            continue
        if panels is not None and panel not in panels:
            continue

        rh_high_pct = int(cycle_label.replace(" %RH", "").split("-")[1])
        data = df.iloc[3:, [t_col, u_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            continue

        t = data.iloc[:, 0].to_numpy()
        u = data.iloc[:, 1].to_numpy()
        switch_t = detect_high_to_20_switch_min(t, u)
        out[(panel, rh_high_pct)] = ChamberRhSchedule(
            panel=panel,
            rh_high_pct=rh_high_pct,
            t_high_to_20_min=switch_t,
            t_end_min=float(t[-1]),
            material=material,
            cycle_label=cycle_label,
        )
    return out


def format_schedule_table(schedules: dict[tuple[str, int], ChamberRhSchedule]) -> str:
    lines = [
        "Panel | RH cycle | t (high→20 %) [min] | t_end [min]",
        "------|----------|---------------------|------------",
    ]
    for (panel, rh_pct), sched in sorted(schedules.items()):
        lines.append(
            f"{panel}  | 20–{rh_pct}–20 | {sched.t_high_to_20_min:8.1f} | {sched.t_end_min:8.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    schedules = load_chamber_rh_schedules()
    print(format_schedule_table(schedules))
