# SPDX-License-Identifier: Apache-2.0
"""Reusable KV-cache capacity simulation building blocks."""

from importlib.metadata import PackageNotFoundError, version

from .config import (
    DEFAULT_CAPACITIES,
    DEFAULT_CAPACITY_LABELS,
    DEFAULT_PAGE_SIZE,
    parse_size,
    simulation_config_from_dict,
)
from .analysis import (
    MattsonStack,
    OnlineLRUAnalyzer,
    RequestAnalysis,
    analyze_capacities,
    request_capacity_stats,
)
from .models import (
    CapacityResult,
    PageRequest,
    RequestCapacityStats,
    TokenIdsRequest,
)
from .pages import chained_page_hashes, requests_from_token_ids
from .simulator import (
    ReplaySimulator,
    SimulationConfig,
    SimulationResult,
    SimulationTimings,
    simulate,
)

try:
    __version__ = version("kv-capacity-estimator")
except PackageNotFoundError:
    # The source tree can still be imported directly without installing a wheel.
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "CapacityResult",
    "DEFAULT_CAPACITIES",
    "DEFAULT_CAPACITY_LABELS",
    "DEFAULT_PAGE_SIZE",
    "MattsonStack",
    "OnlineLRUAnalyzer",
    "PageRequest",
    "ReplaySimulator",
    "RequestAnalysis",
    "RequestCapacityStats",
    "SimulationConfig",
    "SimulationResult",
    "SimulationTimings",
    "TokenIdsRequest",
    "analyze_capacities",
    "request_capacity_stats",
    "chained_page_hashes",
    "requests_from_token_ids",
    "parse_size",
    "simulate",
    "simulation_config_from_dict",
]
