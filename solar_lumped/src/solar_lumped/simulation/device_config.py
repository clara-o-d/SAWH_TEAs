"""Device configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from solar_lumped.physics.device_balances import DeviceThermalParams
from solar_lumped.physics.mass_transfer import MassTransferParams
from solar_lumped.physics.salt_properties import SaltProperties, get_salt, salt_molarity_from_composite
from solar_lumped.physics import table_s3


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    salt_name: str = "LiCl"
    salt_to_polymer_ratio: float = 4.0
    hydrogel_thickness_m: float = table_s3.H0_M
    vapor_gap_m: float = table_s3.L_G_M
    insulation_gap_m: float = table_s3.L_INS_M
    g_conv_m_s: float = table_s3.G_CHAMBER_M_S
    hydrogel_density_kg_m3: float = table_s3.RHO_COMPOSITE_KG_M3
    fin_area_ratio: float = table_s3.FIN_AREA_RATIO
    condenser_thickness_m: float = table_s3.L_AL_M
    condenser_rho_kg_m3: float = table_s3.RHO_AL_KG_M3
    condenser_cp_j_kg_k: float = table_s3.CP_AL_J_KG_K
    h_fg_j_per_kg: float = table_s3.H_FG_J_PER_KG
    tilt_deg: float = table_s3.TILT_DEG
    thermal: DeviceThermalParams | None = None

    def salt(self) -> SaltProperties:
        return get_salt(self.salt_name)

    def mass_params(self) -> MassTransferParams:
        s = self.salt()
        return MassTransferParams(
            g_conv_m_s=self.g_conv_m_s,
            h0_ref_m=self.hydrogel_thickness_m,
            vapor_gap_m=self.vapor_gap_m,
            tilt_deg=self.tilt_deg,
            c_s_mol_m3=salt_molarity_from_composite(
                self.salt_to_polymer_ratio,
                self.hydrogel_density_kg_m3,
                s.formula_weight_g_mol,
            ),
            ions_per_formula=s.ions_per_formula,
            rho_solution_kg_m3=s.rho_solution_kg_m3,
            salt_name=s.name,
            formula_weight_g_mol=s.formula_weight_g_mol,
            salt_to_polymer_ratio=self.salt_to_polymer_ratio,
        )

    def thermal_params(self) -> DeviceThermalParams:
        if self.thermal is not None:
            return self.thermal
        s = self.salt()
        return DeviceThermalParams(
            insulation_gap_m=self.insulation_gap_m,
            vapor_gap_m=self.vapor_gap_m,
            u_gel_w_m2_k=table_s3.U_GEL_W_M2_K,
            eps_abs=table_s3.EPS_ABS,
            tau_glass=table_s3.TAU_GLASS,
            eps_gel=table_s3.EPS_GEL,
            eps_al=table_s3.EPS_AL,
            tilt_deg=self.tilt_deg,
            h_des_j_per_kg=s.h_des_j_per_kg,
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
