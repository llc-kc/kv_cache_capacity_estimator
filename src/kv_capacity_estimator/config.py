# SPDX-License-Identifier: Apache-2.0
"""Public JSON-friendly configuration helpers for the simulator API."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .defaults import (
    DEFAULT_CAPACITIES,
    DEFAULT_CAPACITY_LABELS,
    DEFAULT_PAGE_SIZE,
)
from .simulator import SimulationConfig

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*$", re.IGNORECASE)
_SIZE_MULTIPLIERS = {
    None: 1,
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}
_CONFIG_FIELDS = frozenset(
    {
        "page_size",
        "kv_bytes_per_token",
        "capacities",
        "include_partial_page",
        "warm_up_ratio",
        "target_hit_rate_ratio",
    }
)


def parse_size(value: str) -> int:
    """Parse a decimal or binary byte size such as ``10GB`` or ``100GiB``."""
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(f"invalid capacity {value!r}; examples: 10GB, 100GiB")
    number = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else None
    return math.floor(number * _SIZE_MULTIPLIERS[unit])


def simulation_config_from_dict(config: Mapping[str, Any]) -> SimulationConfig:
    """Build ``SimulationConfig`` from JSON-compatible values.

    Only ``kv_bytes_per_token`` is required. Defaults for every omitted field
    live in this package. Capacities may be a comma-separated string, a list of
    size strings, or ``[label, bytes-or-size]`` pairs.
    """
    if not isinstance(config, Mapping):
        raise TypeError("simulator config must be a JSON object")
    unknown_fields = set(config).difference(_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError(
            "unknown simulator config fields: "
            + ", ".join(sorted(str(field) for field in unknown_fields))
        )
    if "kv_bytes_per_token" not in config:
        raise ValueError("simulator config requires 'kv_bytes_per_token'")

    values = dict(config)
    values.setdefault("page_size", DEFAULT_PAGE_SIZE)
    raw_capacities = values.get("capacities", DEFAULT_CAPACITY_LABELS)
    values["capacities"] = _normalize_capacities(raw_capacities)
    return SimulationConfig(**values)


def _normalize_capacities(value: Any) -> tuple[tuple[str, int], ...]:
    if isinstance(value, str):
        entries: Sequence[Any] = tuple(
            item.strip() for item in value.split(",") if item.strip()
        )
    elif isinstance(value, Sequence):
        entries = value
    else:
        raise TypeError("'capacities' must be a string or JSON array")

    capacities = []
    for entry in entries:
        if isinstance(entry, str):
            label = entry
            capacity_bytes = parse_size(entry)
        elif (
            isinstance(entry, Sequence)
            and not isinstance(entry, (str, bytes))
            and len(entry) == 2
            and isinstance(entry[0], str)
        ):
            label = entry[0]
            capacity_bytes = _normalize_size(entry[1], "capacities")
        else:
            raise TypeError(
                "each capacity must be a size string or [label, bytes-or-size]"
            )
        capacities.append((label, capacity_bytes))
    if not capacities:
        raise ValueError("'capacities' must contain at least one capacity")
    return tuple(capacities)


def _normalize_size(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"'{field}' must be integer bytes or a size string")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return parse_size(value)
    raise TypeError(f"'{field}' must be integer bytes or a size string")
