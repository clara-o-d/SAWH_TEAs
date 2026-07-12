"""Device configuration for waste-heat two-bed SAWH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from waste_heat_cycle_lumped.physics import device_defaults as dd
from waste_heat_cycle_lumped.physics.adsorbent import MofProperties, get_mof
from waste_heat_cycle_lumped.physics.contactor_balances import ContactorThermalParams

SorbentKind = Literal["hydrogel", "mof"]


@dataclass(frozen=True, slots=True)
class ControllerParams:
    m_f_base_kg_s_m2: float = dd.M_F_BASE_KG_S_M2
    m_f_min_kg_s_m2: float = dd.M_F_MIN_KG_S_M2
    m_f_max_kg_s_m2: float = dd.M_F_MAX_KG_S_M2
    c_vac_base_kg_s_pa_m2: float = dd.C_VAC_BASE_KG_S_PA_M2
    c_vac_min_kg_s_pa_m2: float = dd.C_VAC_MIN_KG_S_PA_M2
    c_vac_max_kg_s_pa_m2: float = dd.C_VAC_MAX_KG_S_PA_M2
    k_t_per_k: float = dd.K_T_PER_K
    k_m_per_kg_m2: float = dd.K_M_PER_KG_M2
    k_p_per_kg_s_m2: float = dd.K_P_PER_KG_S_M2


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    sorbent: SorbentKind = dd.DEFAULT_SORBENT  # type: ignore[assignment]
    salt_name: str = dd.DEFAULT_SALT_NAME
    salt_to_polymer_ratio: float = dd.SALT_TO_POLYMER_RATIO
    hydrogel_thickness_m: float = dd.H0_M
    hydrogel_density_kg_m3: float = dd.RHO_COMPOSITE_KG_M3
    g_conv_m_s: float = dd.G_CHAMBER_M_S
    vapor_gap_m: float = dd.VAPOR_GAP_M
    tilt_deg: float = dd.TILT_DEG
    mof_name: str = dd.DEFAULT_MOF_NAME
    tau_half_s: float = dd.TAU_HALF_S
    rh_desorber_switch: float = dd.RH_DESORBER_SWITCH
    p_cond_pa: float = dd.P_COND_PA
    controller: ControllerParams | None = None
    thermal: ContactorThermalParams | None = None

    def mof(self) -> MofProperties:
        return get_mof(self.mof_name)

    def thermal_params(self) -> ContactorThermalParams:
        if self.thermal is not None:
            return self.thermal
        return ContactorThermalParams(p_vacuum_pa=self.p_cond_pa)

    def controller_params(self) -> ControllerParams:
        return self.controller if self.controller is not None else ControllerParams()

    def condenser_thermal_mass_j_m2_k(self) -> float:
        return self.thermal_params().condenser_thermal_mass_j_m2_k

    @classmethod
    def datacenter_baseline(cls, **overrides: object) -> DeviceConfig:
        return cls(**overrides)  # type: ignore[arg-type]

    @classmethod
    def mof_baseline(cls, **overrides: object) -> DeviceConfig:
        base = {"sorbent": "mof"}
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]
