"""The Slurm array task table (site_sweep.array_tasks) decides what the global sweep
actually covers. Nothing downstream notices if it silently skips sites -- the CSV just
comes out short -- so the coverage and pairing properties are pinned here."""

from __future__ import annotations

import pytest

from solar_lumped.site_sweep import (
    GROUP_RUNS,
    SCENARIOS,
    array_tasks,
    scenario_groups,
)
from solar_lumped.weather import grid_land_points


def test_every_site_scenario_pair_is_covered_exactly_once() -> None:
    """Each group is chunked over its OWN grid, so coverage has to be checked per group
    rather than against one site count."""
    counts = {step: len(grid_land_points(step)) for step in {r.step_deg for r in GROUP_RUNS.values()}}
    seen: set[tuple[float, int, str]] = set()
    for _gid, step, start, end, names in array_tasks(counts.__getitem__):
        for name in names:
            for site in range(start, end):
                key = (step, site, name)
                assert key not in seen, f"task table covers {key} twice"
                seen.add(key)
    expected = sum(
        counts[GROUP_RUNS[key].step_deg] * len(names)
        for key, names in scenario_groups().items()
    )
    assert len(seen) == expected


def test_coarse_grid_is_a_subset_of_the_fine_one() -> None:
    """Every group runs the 3-degree grid today, so this guards the option rather than the
    current config: if a group is ever re-coarsened for cost (the instant groups were, while
    they were still a stiff penalty), its rows must still pair with the others' at the same
    sites. A 6-degree grid that were not a subset would silently break every
    scenario-vs-scenario comparison."""
    fine = {tuple(p) for p in grid_land_points(3.0)}
    coarse = {tuple(p) for p in grid_land_points(6.0)}
    assert coarse, "6-degree grid is empty"
    assert coarse <= fine


def test_every_scenario_group_has_a_run_config() -> None:
    """A scenario added to SCENARIOS with a new (instant, ambient) combination would
    otherwise KeyError inside the sbatch prologue, mid-array, on the cluster."""
    assert set(scenario_groups()) == set(GROUP_RUNS)
    assert sum(len(v) for v in scenario_groups().values()) == len(SCENARIOS)


def test_task_indices_are_grouped_contiguously() -> None:
    """The array script's per-group submission ranges (--array=8-37,46-60) rely on it."""
    gids = [gid for gid, *_ in array_tasks(lambda step: len(grid_land_points(step)))]
    assert gids == sorted(gids)


def test_chunks_are_within_the_configured_width() -> None:
    """Parametrized over the steps GROUP_RUNS actually configures, not a fixed list: when
    every group moved back to 3 degrees, a hardcoded 6.0 case asserted on an empty set."""
    tasks = array_tasks(lambda step: len(grid_land_points(step)))
    for step in {r.step_deg for r in GROUP_RUNS.values()}:
        widths = [end - start for _gid, s, start, end, _n in tasks if s == step]
        limit = max(r.sites_per_chunk for r in GROUP_RUNS.values() if r.step_deg == step)
        assert widths, f"no tasks at {step} deg"
        assert max(widths) <= limit
