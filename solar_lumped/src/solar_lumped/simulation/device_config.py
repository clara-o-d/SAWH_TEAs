"""Device configuration dataclass."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from solar_lumped.physics.adsorbent import DEFAULT_MOF_NAME, MofProperties, get_mof
from solar_lumped.physics.device_balances import DeviceThermalParams
from solar_lumped.physics.mass_transfer import MassTransferParams
from solar_lumped.physics.salt_properties import (
    DRY_COMPOSITE_DENSITY_KG_M3,
    SaltProperties,
    get_salt,
    salt_molarity_from_composite,
)
from solar_lumped.physics.sorbent import SorbentKind
from solar_lumped.physics import table_s3

# Desorption integration mode (see ``run_daily_cycle`` in ode_system.py).
DesorptionSolverMode = Literal["quasi_steady", "segregated", "coupled_bdf"]

DESORPTION_SOLVER_CHOICES: tuple[DesorptionSolverMode, ...] = (
    "quasi_steady",
    "segregated",
    "coupled_bdf",
)


def register_desorption_solver_cli(
    parser: argparse.ArgumentParser,
    *,
    default: DesorptionSolverMode = "quasi_steady",
) -> None:
    """Add ``--desorption-solver`` to a figure/script CLI."""
    parser.add_argument(
        "--desorption-solver",
        choices=DESORPTION_SOLVER_CHOICES,
        default=default,
        metavar="MODE",
        help=(
            "Desorption integrator: quasi_steady (algebraic Eqs 1/3/4), "
            "segregated (COMSOL v6.2 sequential 100 s steps), "
            "coupled_bdf (fully coupled lumped ODE, SciPy BDF)."
        ),
    )


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    sorbent: SorbentKind = "hydrogel"
    mof_name: str = DEFAULT_MOF_NAME
    salt_name: str = "LiCl"
    salt_to_polymer_ratio: float = 4.0
    hydrogel_thickness_m: float = table_s3.H0_M
    vapor_gap_m: float = table_s3.L_G_M
    insulation_gap_m: float = table_s3.L_INS_M
    g_conv_m_s: float = table_s3.G_CHAMBER_M_S
    hydrogel_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3
    fin_area_ratio: float = table_s3.FIN_AREA_RATIO
    condenser_thickness_m: float = table_s3.L_C_M
    condenser_rho_kg_m3: float = table_s3.RHO_AL_KG_M3
    condenser_cp_j_kg_k: float = table_s3.CP_AL_J_KG_K
    h_fg_j_per_kg: float = table_s3.H_FG_J_PER_KG
    tilt_deg: float = table_s3.TILT_DEG
    thermal: DeviceThermalParams | None = None
    # Override catalog salt formula weight (g/mol) for sensitivity sweeps.
    salt_formula_weight_g_mol: float | None = None
    # Desorption integration: quasi_steady (default) solves Eqs 1/3/4 algebraically
    # each ODE step; segregated mimics COMSOL v6.2's sequential 100 s solver with
    # small surface capacitances; coupled_bdf advances all lumped states together
    # with SciPy's variable-order BDF.
    desorption_solver: DesorptionSolverMode = "quasi_steady"
    # Uniform surface/gel temperature at desorption start. None → desorption-start
    # ambient temperature (transient solvers) or algebraic steady state (quasi_steady).
    segregated_initial_temp_c: float | None = None
    # Per-component desorption-start temperatures (T_gel, T_abs, T_glass, T_cond) in
    # °C. Takes precedence over ``segregated_initial_temp_c`` when set — e.g. to
    # match the first digitized Wilson data point (all solvers, including quasi_steady).
    coupled_initial_temps_c: tuple[float, float, float, float] | None = None

    def desorption_surface_ic_c(self) -> tuple[float, float, float, float] | None:
        """Configured (T_gel, T_abs, T_glass, T_cond) at desorption start, if any."""
        if self.coupled_initial_temps_c is not None:
            return self.coupled_initial_temps_c
        if self.segregated_initial_temp_c is not None:
            t = self.segregated_initial_temp_c
            return (t, t, t, t)
        return None

    @property
    def segregated_desorption(self) -> bool:
        """True when using the COMSOL-style segregated desorption integrator."""
        return self.desorption_solver == "segregated"

    @property
    def coupled_ode_desorption(self) -> bool:
        """True when using the fully coupled BDF ODE desorption integrator."""
        return self.desorption_solver == "coupled_bdf"

    def salt(self) -> SaltProperties:
        return get_salt(self.salt_name)

    def mof(self) -> MofProperties:
        return get_mof(self.mof_name)

    def mass_params(self) -> MassTransferParams:
        if self.sorbent == "mof":
            props = self.mof()
            return MassTransferParams(
                g_conv_m_s=props.g_conv_m_s,
                h0_ref_m=self.hydrogel_thickness_m,
                vapor_gap_m=self.vapor_gap_m,
                tilt_deg=self.tilt_deg,
                c_s_mol_m3=0.0,
                ions_per_formula=1,
                rho_solution_kg_m3=1000.0,
                salt_name="MOF",
                formula_weight_g_mol=1.0,
                salt_to_polymer_ratio=1.0,
            )
        s = self.salt()
        fw = (
            self.salt_formula_weight_g_mol
            if self.salt_formula_weight_g_mol is not None
            else s.formula_weight_g_mol
        )
        return MassTransferParams(
            g_conv_m_s=self.g_conv_m_s,
            h0_ref_m=self.hydrogel_thickness_m,
            vapor_gap_m=self.vapor_gap_m,
            tilt_deg=self.tilt_deg,
            c_s_mol_m3=salt_molarity_from_composite(
                self.salt_to_polymer_ratio,
                self.hydrogel_density_kg_m3,
                fw,
            ),
            ions_per_formula=s.ions_per_formula,
            rho_solution_kg_m3=s.rho_solution_kg_m3,
            salt_name=s.name,
            formula_weight_g_mol=fw,
            salt_to_polymer_ratio=self.salt_to_polymer_ratio,
        )

    def thermal_params(self) -> DeviceThermalParams:
        if self.thermal is not None:
            return self.thermal
        if self.sorbent == "mof":
            h_des = self.mof().h_des_j_per_kg
        elif self.salt_name == "LiCl":
            # Wilson Table S3 COMSOL value (2320 kJ/kg), not the broader Díaz-Marín
            # literature range in salt_heat_of_desorption.csv (~2850 kJ/kg).
            h_des = table_s3.H_DES_J_PER_KG
        else:
            h_des = self.salt().h_des_j_per_kg
        return DeviceThermalParams(
            insulation_gap_m=self.insulation_gap_m,
            vapor_gap_m=self.vapor_gap_m,
            eps_abs=table_s3.EPS_ABS,
            tau_glass=table_s3.TAU_GLASS,
            eps_gel=table_s3.EPS_GEL,
            eps_al=table_s3.EPS_AL,
            eps_glass=table_s3.EPS_GLASS,
            tilt_deg=self.tilt_deg,
            h_des_j_per_kg=h_des,
        )

    def condenser_thermal_mass_j_m2_k(self) -> float:
        return (
            self.condenser_rho_kg_m3
            * self.condenser_cp_j_kg_k
            * self.condenser_thickness_m
        )

    @classmethod
    def comsol_table_s3(cls, **overrides: object) -> DeviceConfig:
        """Wilson Table S3 / Note S1 COMSOL SAWH device defaults."""
        return cls(**overrides)  # type: ignore[arg-type]

    @classmethod
    def baseline(cls, **overrides: object) -> DeviceConfig:
        """Wilson Fig. 2 baseline device (Table S3, tilt 30°, fin area ratio 7.1)."""
        base = {
            "tilt_deg": 30.0,
            "fin_area_ratio": 7.1,
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    @classmethod
    def atacama_field(cls, **overrides: object) -> DeviceConfig:
        """Wilson Atacama field-test geometry (Methods): tilt 25°, fin area ratio 5."""
        base = {
            "tilt_deg": 25.0,
            "fin_area_ratio": 5.0,
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]
