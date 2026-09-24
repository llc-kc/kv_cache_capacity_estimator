# SPDX-License-Identifier: Apache-2.0
"""Storage-independent models shared by all replay sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenIdsRequest:
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class PageRequest:
    page_keys: tuple[bytes, ...]


@dataclass(frozen=True)
class CapacityResult:
    requested_capacity: str
    requested_capacity_bytes: int | None
    effective_capacity_bytes: int | None
    capacity_pages: int | None
    page_hits: int
    page_misses: int
    page_hit_rate: float
    reusable_page_hits: int
    reusable_page_hit_rate: float
    prefix_hit_pages: int
    prefix_page_hit_rate: float


@dataclass(frozen=True)
class RequestCapacityStats:
    target_hit_rate_ratio: float
    samples: int
    mean_capacity_bytes: float | None
    p50_capacity_bytes: int | None
    p90_capacity_bytes: int | None
    p95_capacity_bytes: int | None
    p99_capacity_bytes: int | None
    p999_capacity_bytes: int | None
