"""Patent hardware bill of materials for waste-heat two-bed SAWH (USD per m² footprint)."""

from __future__ import annotations

# Midpoints of patent BOM cost ranges (USD per 1 m² footprint module).
DEVICE_BOM_USD_PER_M2: tuple[tuple[str, float], ...] = (
    ("Transfer pump (18)", 550.0),
    ("Vacuum pump (28)", 3500.0),
    ("Chambers (22A, 22B) with door assemblies (38)", 1050.0),
    ("Three-way valve (32) + check valve (30)", 275.0),
    ("Condenser (24)", 850.0),
    ("Coolant source (26)", 325.0),
    ("Water pump (34)", 165.0),
    ("Controller (16) + sensors (36)", 800.0),
    ("Purge pump (234)", 400.0),
    ("Structural housing, manifolds, plumbing, fasteners", 1350.0),
)

C_DEVICE_USD: float = sum(cost for _, cost in DEVICE_BOM_USD_PER_M2)
