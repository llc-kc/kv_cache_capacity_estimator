# SPDX-License-Identifier: Apache-2.0
"""Console and JSON rendering for simulation results."""

from __future__ import annotations

import json
from dataclasses import asdict

from .simulator import SimulationResult


def human_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    raise AssertionError("unreachable")


def format_table(simulation: SimulationResult) -> str:
    header = (
        f"{'Capacity':>12} {'Pages':>12} {'Page hit':>11} "
        f"{'Reuse hit':>11} {'Prefix hit':>12}"
    )
    lines = [
        f"requests={simulation.requests}, "
        f"warm_up_requests={simulation.warm_up_requests}, "
        f"total_requests={simulation.total_requests}, "
        f"page_accesses={simulation.page_accesses}, "
        f"page_bytes={human_bytes(simulation.page_bytes)}",
        header,
        "-" * len(header),
    ]
    for result in simulation.results:
        pages = (
            "infinite"
            if result.capacity_pages is None
            else f"{result.capacity_pages:,d}"
        )
        lines.append(
            f"{result.requested_capacity:>12} {pages:>12} "
            f"{result.page_hit_rate:>10.2%} "
            f"{result.reusable_page_hit_rate:>10.2%} "
            f"{result.prefix_page_hit_rate:>11.2%}"
        )
    stats = simulation.request_capacity_stats
    values = (
        ("mean", stats.mean_capacity_bytes),
        ("p50", stats.p50_capacity_bytes),
        ("p90", stats.p90_capacity_bytes),
        ("p95", stats.p95_capacity_bytes),
        ("p99", stats.p99_capacity_bytes),
        ("p999", stats.p999_capacity_bytes),
    )
    lines.extend(
        [
            "",
            "Per-request target capacity "
            f"({stats.target_hit_rate_ratio:.2%} of theoretical prefix hit rate, "
            f"samples={stats.samples}):",
            "  " + "  ".join(
                f"{name}={'n/a' if value is None else human_bytes(value)}"
                for name, value in values
            ),
            "",
            "Timing (wall clock):",
            "  "
            f"input/tokenize={simulation.timings.input_processing_seconds:.3f}s  "
            f"page_build={simulation.timings.page_build_seconds:.3f}s  "
            f"reuse_distance={simulation.timings.reuse_distance_seconds:.3f}s",
            "  "
            f"fixed_capacity={simulation.timings.fixed_capacity_seconds:.3f}s  "
            "capacity_requirement="
            f"{simulation.timings.capacity_requirement_seconds:.3f}s  "
            f"total={simulation.timings.total_seconds:.3f}s",
        ]
    )
    return "\n".join(lines)


def format_json(simulation: SimulationResult) -> str:
    return json.dumps(asdict(simulation), indent=2)
