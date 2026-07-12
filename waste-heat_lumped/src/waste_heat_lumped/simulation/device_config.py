"""Device configuration for fluid-heated daily-cycle SAWH."""

from __future__ import annotations

from dataclasses import dataclass

from waste_heat_lumped.physics import device_defaults as dd
from waste_heat_lumped.physics import table_s3
from waste_heat_lumped.physics.device_balances import DeviceThermalParams
from waste_heat_lumped.physics.mass_transfer import MassTransferParams
from waste_heat_lumped.physics.salt_properties import (
    DRY_COMPOSITE_DENSITY_KG_M3,
    SaltProperties,
    get_salt,
    salt_molarity_from_composite,
)
from waste_heat_lumped.physics.sorbent import SorbentKind


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    sorbent: SorbentKind = "hydrogel"
    salt_name: str = dd.DEFAULT_SALT_NAME
    salt_to_polymer_ratio: float = dd.SALT_TO_POLYMER_RATIO
    hydrogel_thickness_m: float = dd.H0_M
    vapor_gap_m: float = dd.VAPOR_GAP_M
    g_conv_m_s: float = dd.G_CHAMBER_M_S
    hydrogel_density_kg_m3: float = DRY_COMPOSITE_DENSITY_KG_M3
    fin_area_ratio: float = dd.FIN_AREA_RATIO
    condenser_thickness_m: float = dd.CONDENSER_THICKNESS_M
    condenser_rho_kg_m3: float = dd.CONDENSER_RHO_KG_M3
    condenser_cp_j_kg_k: float = dd.CONDENSER_CP_J_KG_K
    h_fg_j_per_kg: float = dd.H_FG_J_PER_KG
    tilt_deg: float = dd.TILT_DEG
    # Fixed loop-fluid setpoints (active during desorption only)
    t_f_c: float = dd.T_F_C
    m_dot_f_kg_s_m2: float = dd.M_DOT_F_KG_S_M2
    ua_gel_w_k: float = dd.UA_GEL_W_K
    fluid_cp_j_kg_k: float = dd.FLUID_CP_J_KG_K
    gel_thermal_mass_j_m2_k: float = dd.GEL_THERMAL_MASS_J_M2_K
    salt_formula_weight_g_mol: float | None = None
    salt_weight_factor: float = 1.0

    def salt(self) -> SaltProperties:
        return get_salt(self.salt_name)

    def mass_params(self) -> MassTransferParams:
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
            salt_weight_factor=self.salt_weight_factor,
        )

    def thermal_params(self) -> DeviceThermalParams:
        if self.salt_name == "LiCl":
            h_des = table_s3.H_DES_J_PER_KG
        else:
            h_des = self.salt().h_des_j_per_kg
        return DeviceThermalParams(
            vapor_gap_m=self.vapor_gap_m,
            eps_gel=dd.GEL_EMISSIVITY,
            eps_al=dd.CONDENSER_EMISSIVITY,
            tilt_deg=self.tilt_deg,
            h_des_j_per_kg=h_des,
            gel_thermal_mass_j_m2_k=self.gel_thermal_mass_j_m2_k,
            t_f_c=self.t_f_c,
            m_dot_f_kg_s_m2=self.m_dot_f_kg_s_m2,
            ua_gel_w_k=self.ua_gel_w_k,
            fluid_cp_j_kg_k=self.fluid_cp_j_kg_k,
        )

    def condenser_thermal_mass_j_m2_k(self) -> float:
        return (
            self.condenser_rho_kg_m3
            * self.condenser_cp_j_kg_k
            * self.condenser_thickness_m
        )

    @classmethod
    def datacenter_baseline(cls, **overrides: object) -> DeviceConfig:
        """Data-center fluid-heated daily cycle defaults."""
        return cls(**overrides)  # type: ignore[arg-type]

    @classmethod
    def baseline(cls, **overrides: object) -> DeviceConfig:
        return cls.datacenter_baseline(**overrides)
