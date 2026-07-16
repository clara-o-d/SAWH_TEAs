"""Uniform ``ConfigAdapter`` interface over the four SAWH device packages.

Each adapter hides that package's real (already working, already-tested)
entry points behind a common shape so the comparison scripts never need
per-config branching:

* ``passive``      -> ``solar_lumped``                       (script-driven)
* ``single_loop``  -> ``waste_heat_lumped``                  (script-driven)
* ``multi_loop``   -> ``waste_heat_cycle_lumped``             (direct DeviceConfig + ode_system)
* ``multi_noloop`` -> ``waste_heat_cycle_lumped_no_loop``     (direct DeviceConfig + ode_system,
                                                                same call shape as multi_loop --
                                                                the HTF-loop removal is entirely
                                                                internal to that package's physics)

``passive`` and ``single_loop`` are driven through their packages'
``scripts/run_*.py`` modules (``run_solar_simulation`` /
``run_waste_heat_simulation``) because that is where the CLI-argument ->
``DeviceConfig`` construction logic (and the LCOW/NPV wiring) actually lives.
``multi_loop`` / ``multi_noloop`` skip their scripts entirely (per the task
brief, to avoid importing private script internals across packages with
colliding filenames) and instead call ``DeviceConfig(...)`` and
``simulation.ode_system.run_daily_operation(...)`` directly, exactly as
``scripts/run_waste_heat_cycle_sim.py --daily`` does internally.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, fields
from typing import Any, Protocol

from comparison.lib import bootstrap  # noqa: F401  (side effect: src/ on sys.path)
from comparison.lib.heat_input_axis import HeatInputMapping, map_heat_input_frac
from comparison.lib.scenario import BASELINE_SCENARIO, Scenario

_SCENARIO: Scenario = BASELINE_SCENARIO


def _replace_econ(econ: Any, **overrides: Any) -> Any:
    """``dataclasses.replace``-equivalent for the frozen, ``init=False`` ``LCOEconomicParams``.

    ``LCOEconomicParams.__init__`` takes arbitrary kwargs and falls back to
    the package's CSV-loaded defaults for anything missing -- so a plain
    ``dataclasses.replace`` would silently drop any *current* field value
    that isn't in ``overrides`` (they'd revert to the CSV default instead of
    staying at ``econ``'s current value). Build the full current-value dict
    explicitly instead.
    """
    if not overrides:
        return econ
    current = {f.name: getattr(econ, f.name) for f in fields(econ)}
    current.update(overrides)
    return type(econ)(**current)


@dataclass(frozen=True, slots=True)
class SimOutput:
    """Result of one adapter ``simulate()`` call."""

    config_id: str
    daily_yield_kg_per_m2: float
    thermal_efficiency: float
    cycles_per_day: float
    heat_input_frac: float
    heat_input_physical_value: float
    heat_input_unit: str
    heat_input_param_name: str
    econ: Any
    material_kwargs: dict[str, Any]
    raw: Any = None


class ConfigAdapter(Protocol):
    config_id: str
    display_name: str
    color: str

    def econ_defaults(self) -> Any: ...

    def bom_line_items(self) -> tuple[tuple[str, float], ...]: ...

    def simulate(self, *, econ: Any, heat_input_frac: float, **econ_overrides: Any) -> SimOutput: ...

    def npv(
        self,
        daily_yield_kg_per_m2: float,
        water_price_usd_per_m3: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> Any: ...

    def lcow(
        self,
        daily_yield_kg_per_m2: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> float: ...


def _scenario_econ(params_cls: Any) -> Any:
    """Baseline ``LCOEconomicParams`` with the shared scenario econ fields pinned.

    The packages' own CSV defaults already match ``scenario.py`` (device
    lifetime 20 yr, discount rate 0.08, electricity $0.10/kWh) as of writing;
    this makes that agreement explicit/robust instead of implicit, so a
    future change to one package's CSV defaults can't silently desync the
    comparison from its own stated scenario.
    """
    econ = params_cls()
    return _replace_econ(
        econ,
        device_lifetime_years=_SCENARIO.device_lifetime_years,
        discount_rate=_SCENARIO.discount_rate,
        electricity_price_usd_per_kwh=_SCENARIO.electricity_price_usd_per_kwh,
    )


class PassiveAdapter:
    """``solar_lumped`` -- solar-driven, fixed 12h/12h daily cycle."""

    config_id = "passive"
    display_name = "Passive (solar)"
    color = "#0072B2"

    def __init__(self) -> None:
        bootstrap.ensure_scripts_on_path("solar_lumped")
        self._mod = importlib.import_module("run_solar_sim")

    def econ_defaults(self) -> Any:
        from solar_lumped.economics.params import LCOEconomicParams

        return _scenario_econ(LCOEconomicParams)

    def bom_line_items(self) -> tuple[tuple[str, float], ...]:
        from solar_lumped.economics.params import DEVICE_BOM_USD_PER_M2

        return tuple(DEVICE_BOM_USD_PER_M2)

    def _build_args(self, mapping: HeatInputMapping) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        self._mod.register_solar_sim_arguments(parser)
        args = parser.parse_args([])
        args.weather_mode = "baseline"
        args.salt = _SCENARIO.salt_name
        args.salt_loading = _SCENARIO.salt_to_polymer_ratio
        args.hydrogel_thickness_mm = _SCENARIO.hydrogel_thickness_m * 1e3
        self._mod.resolve_solar_sim_arguments(args, parser)
        return args

    def simulate(self, *, econ: Any, heat_input_frac: float, **econ_overrides: Any) -> SimOutput:
        econ = _replace_econ(econ, **econ_overrides)
        mapping = map_heat_input_frac(self.config_id, heat_input_frac)
        args = self._build_args(mapping)
        result = self._mod.run_solar_simulation(
            args,
            econ=econ,
            baseline_profile_kwargs={
                "solar_w_m2": mapping.physical_value,
                "relative_humidity": _SCENARIO.rh_amb,
                "temperature_c": _SCENARIO.t_amb_c,
                "h_amb_w_m2_k": _SCENARIO.h_amb_w_m2_k,
            },
        )
        return SimOutput(
            config_id=self.config_id,
            daily_yield_kg_per_m2=float(result.daily_yield_kg_per_m2),
            thermal_efficiency=float(result.thermal_efficiency),
            cycles_per_day=1.0,
            heat_input_frac=float(heat_input_frac),
            heat_input_physical_value=mapping.physical_value,
            heat_input_unit=mapping.unit,
            heat_input_param_name=mapping.param_name,
            econ=econ,
            material_kwargs={
                "salt_name": _SCENARIO.salt_name,
                "salt_to_polymer_ratio": _SCENARIO.salt_to_polymer_ratio,
                "hydrogel_thickness_m": _SCENARIO.hydrogel_thickness_m,
                "sorbent": "hydrogel",
            },
            raw=result,
        )

    def npv(
        self,
        daily_yield_kg_per_m2: float,
        water_price_usd_per_m3: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> Any:
        from solar_lumped.economics.npv import npv_from_daily_yield

        return npv_from_daily_yield(
            daily_yield_kg_per_m2,
            water_price_usd_per_m3,
            econ=econ,
            cycles_per_day=cycles_per_day,
            **material_kwargs,
        )

    def lcow(
        self,
        daily_yield_kg_per_m2: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> float:
        from solar_lumped.economics.lcow import lcow_from_daily_yield

        return lcow_from_daily_yield(
            daily_yield_kg_per_m2,
            econ=econ,
            cycles_per_day=cycles_per_day,
            **material_kwargs,
        )


class SingleLoopAdapter:
    """``waste_heat_lumped`` -- single fixed 12h/12h cycle/day, HTF loop, no vacuum pump."""

    config_id = "single_loop"
    display_name = "Single-loop (waste heat)"
    color = "#E69F00"

    def __init__(self) -> None:
        bootstrap.ensure_scripts_on_path("waste-heat_lumped")
        self._mod = importlib.import_module("run_waste_heat_sim")

    def econ_defaults(self) -> Any:
        from waste_heat_lumped.economics.params import LCOEconomicParams

        return _scenario_econ(LCOEconomicParams)

    def bom_line_items(self) -> tuple[tuple[str, float], ...]:
        from waste_heat_lumped.economics.params import DEVICE_BOM_USD_PER_M2

        return tuple(DEVICE_BOM_USD_PER_M2)

    def _build_args(self, mapping: HeatInputMapping) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        self._mod.register_waste_heat_sim_arguments(parser)
        self._mod.register_cyclic_warmup_arguments(parser)
        args = parser.parse_args([])
        args.salt = _SCENARIO.salt_name
        args.salt_loading = _SCENARIO.salt_to_polymer_ratio
        args.hydrogel_thickness_mm = _SCENARIO.hydrogel_thickness_m * 1e3
        args.t_amb_c = _SCENARIO.t_amb_c
        args.rh = _SCENARIO.rh_amb
        args.h_amb = _SCENARIO.h_amb_w_m2_k
        args.t_f_c = mapping.physical_value
        return args

    def simulate(self, *, econ: Any, heat_input_frac: float, **econ_overrides: Any) -> SimOutput:
        econ = _replace_econ(econ, **econ_overrides)
        mapping = map_heat_input_frac(self.config_id, heat_input_frac)
        args = self._build_args(mapping)
        result = self._mod.run_waste_heat_simulation(args, econ=econ)
        return SimOutput(
            config_id=self.config_id,
            daily_yield_kg_per_m2=float(result.daily_yield_kg_per_m2),
            thermal_efficiency=float(result.thermal_efficiency),
            cycles_per_day=1.0,
            heat_input_frac=float(heat_input_frac),
            heat_input_physical_value=mapping.physical_value,
            heat_input_unit=mapping.unit,
            heat_input_param_name=mapping.param_name,
            econ=econ,
            material_kwargs={
                "salt_name": _SCENARIO.salt_name,
                "salt_to_polymer_ratio": _SCENARIO.salt_to_polymer_ratio,
                "hydrogel_thickness_m": _SCENARIO.hydrogel_thickness_m,
                "sorbent": "hydrogel",
            },
            raw=result,
        )

    def npv(
        self,
        daily_yield_kg_per_m2: float,
        water_price_usd_per_m3: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> Any:
        from waste_heat_lumped.economics.npv import npv_from_daily_yield

        return npv_from_daily_yield(
            daily_yield_kg_per_m2,
            water_price_usd_per_m3,
            econ=econ,
            cycles_per_day=cycles_per_day,
            **material_kwargs,
        )

    def lcow(
        self,
        daily_yield_kg_per_m2: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> float:
        from waste_heat_lumped.economics.lcow import lcow_from_daily_yield

        return lcow_from_daily_yield(
            daily_yield_kg_per_m2,
            econ=econ,
            cycles_per_day=cycles_per_day,
            **material_kwargs,
        )


class _CycleAdapterBase:
    """Shared implementation for ``multi_loop`` and ``multi_noloop``.

    Both packages expose an identical ``DeviceConfig`` / ``datacenter_baseline_profile``
    / ``run_daily_operation`` call shape -- the pumped-HTF-loop-vs-direct-coupling
    difference between the two is entirely internal to each package's
    ``simulation.ode_system`` physics, so one adapter implementation (parameterized
    by package import name) covers both.
    """

    config_id: str
    display_name: str
    color: str
    _package_name: str

    def __init__(self) -> None:
        self._device_config_mod = importlib.import_module(
            f"{self._package_name}.simulation.device_config"
        )
        self._profiles_mod = importlib.import_module(f"{self._package_name}.weather.profiles")
        self._ode_mod = importlib.import_module(f"{self._package_name}.simulation.ode_system")
        self._dd_mod = importlib.import_module(f"{self._package_name}.physics.device_defaults")
        self._params_mod = importlib.import_module(f"{self._package_name}.economics.params")
        self._npv_mod = importlib.import_module(f"{self._package_name}.economics.npv")
        self._lcow_mod = importlib.import_module(f"{self._package_name}.economics.lcow")

    def econ_defaults(self) -> Any:
        return _scenario_econ(self._params_mod.LCOEconomicParams)

    def bom_line_items(self) -> tuple[tuple[str, float], ...]:
        return tuple(self._params_mod.DEVICE_BOM_USD_PER_M2)

    def simulate(self, *, econ: Any, heat_input_frac: float, **econ_overrides: Any) -> SimOutput:
        econ = _replace_econ(econ, **econ_overrides)
        mapping = map_heat_input_frac(self.config_id, heat_input_frac)
        config = self._device_config_mod.DeviceConfig(
            salt_name=_SCENARIO.salt_name,
            salt_to_polymer_ratio=_SCENARIO.salt_to_polymer_ratio,
            hydrogel_thickness_m=_SCENARIO.hydrogel_thickness_m,
        )
        profile = self._profiles_mod.datacenter_baseline_profile(
            tau_half_s=config.tau_half_s,
            t_amb_c=_SCENARIO.t_amb_c,
            rh=_SCENARIO.rh_amb,
            h_amb=_SCENARIO.h_amb_w_m2_k,
            t_wh_in_c=mapping.physical_value,
            m_dot_wh_kg_s_m2=self._dd_mod.M_WH_KG_S_M2,
        )
        yield_kg, eta, results = self._ode_mod.run_daily_operation(
            profile, config, n_cycles=None
        )
        cycles_per_day = float(len(results))
        return SimOutput(
            config_id=self.config_id,
            daily_yield_kg_per_m2=float(yield_kg),
            thermal_efficiency=float(eta),
            cycles_per_day=cycles_per_day,
            heat_input_frac=float(heat_input_frac),
            heat_input_physical_value=mapping.physical_value,
            heat_input_unit=mapping.unit,
            heat_input_param_name=mapping.param_name,
            econ=econ,
            material_kwargs={
                "salt_name": _SCENARIO.salt_name,
                "salt_to_polymer_ratio": _SCENARIO.salt_to_polymer_ratio,
                "hydrogel_thickness_m": _SCENARIO.hydrogel_thickness_m,
            },
            raw=(config, profile, results),
        )

    def npv(
        self,
        daily_yield_kg_per_m2: float,
        water_price_usd_per_m3: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> Any:
        return self._npv_mod.npv_from_daily_yield(
            daily_yield_kg_per_m2,
            water_price_usd_per_m3,
            econ=econ,
            cycles_per_day=cycles_per_day,
            **material_kwargs,
        )

    def lcow(
        self,
        daily_yield_kg_per_m2: float,
        *,
        econ: Any,
        cycles_per_day: float,
        **material_kwargs: Any,
    ) -> float:
        return self._lcow_mod.lcow_from_daily_yield(
            daily_yield_kg_per_m2,
            econ=econ,
            cycles_per_day=cycles_per_day,
            **material_kwargs,
        )


class MultiLoopAdapter(_CycleAdapterBase):
    """``waste_heat_cycle_lumped`` -- multi-cycle/day, HTF loop AND vacuum pump."""

    config_id = "multi_loop"
    display_name = "Multi-loop (waste heat)"
    color = "#009E73"
    _package_name = "waste_heat_cycle_lumped"


class MultiNoLoopAdapter(_CycleAdapterBase):
    """``waste_heat_cycle_lumped_no_loop`` -- multi-cycle/day, direct waste-heat coupling."""

    config_id = "multi_noloop"
    display_name = "Multi, no loop (waste heat)"
    color = "#CC79A7"
    _package_name = "waste_heat_cycle_lumped_no_loop"


ALL_CONFIG_IDS: tuple[str, ...] = ("passive", "single_loop", "multi_loop", "multi_noloop")

_ADAPTER_CLASSES: dict[str, type] = {
    "passive": PassiveAdapter,
    "single_loop": SingleLoopAdapter,
    "multi_loop": MultiLoopAdapter,
    "multi_noloop": MultiNoLoopAdapter,
}

_INSTANCES: dict[str, ConfigAdapter] = {}


def get_adapter(config_id: str) -> ConfigAdapter:
    """Lazily construct and cache the adapter for ``config_id``."""
    if config_id not in _ADAPTER_CLASSES:
        raise ValueError(
            f"Unknown config_id {config_id!r}; expected one of {ALL_CONFIG_IDS}"
        )
    if config_id not in _INSTANCES:
        _INSTANCES[config_id] = _ADAPTER_CLASSES[config_id]()
    return _INSTANCES[config_id]


def get_adapters(config_ids: tuple[str, ...] | list[str] | None = None) -> dict[str, ConfigAdapter]:
    """Return ``{config_id: adapter}`` for ``config_ids`` (default: all four), in order."""
    ids = tuple(config_ids) if config_ids else ALL_CONFIG_IDS
    return {cid: get_adapter(cid) for cid in ids}
