# SPDX-License-Identifier: Apache-2.0
"""Mattson stack-distance analysis over a page-access trace."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .models import CapacityResult, PageRequest, RequestCapacityStats


logger = logging.getLogger(__name__)


class FenwickTree:
    """Store one marker at every page key's latest access position."""

    def __init__(self, size: int = 0) -> None:
        if size < 0:
            raise ValueError("size must not be negative")
        self._tree = [0] * (size + 1)

    def append(self) -> None:
        """Add an index while retaining the Fenwick sums already stored."""
        index = len(self._tree)
        low_bit = index & -index
        previous_range_sum = self.prefix_sum(index - 1) - self.prefix_sum(
            index - low_bit
        )
        self._tree.append(previous_range_sum)

    @property
    def max_index(self) -> int:
        return len(self._tree) - 1

    def add(self, index: int, delta: int) -> None:
        while index < len(self._tree):
            self._tree[index] += delta
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        result = 0
        while index > 0:
            result += self._tree[index]
            index -= index & -index
        return result


class MattsonStack:
    """Compute exact LRU stack distance in O(log N) per page access."""

    def __init__(self, max_accesses: int | None = None) -> None:
        if max_accesses is not None and max_accesses < 0:
            raise ValueError("max_accesses must not be negative")
        self._latest_positions: dict[bytes, int] = {}
        self._positions = FenwickTree(max_accesses or 0)
        self._time = 0

    def access(self, key: bytes) -> int | None:
        """Return distinct-page reuse distance, or ``None`` for a cold access."""
        self._time += 1
        if self._time > self._positions.max_index:
            self._positions.append()
        old_position = self._latest_positions.get(key)
        if old_position is None:
            distance = None
        else:
            distance = self._positions.prefix_sum(
                self._time - 1
            ) - self._positions.prefix_sum(old_position)
            self._positions.add(old_position, -1)
        self._positions.add(self._time, 1)
        self._latest_positions[key] = self._time
        return distance


@dataclass(frozen=True)
class RequestAnalysis:
    """LRU results computed when one request arrives."""

    page_accesses: int
    reusable_accesses: int
    infinite_page_hits: int
    infinite_prefix_hits: int
    capacity_page_hits: tuple[int, ...]
    capacity_prefix_hits: tuple[int, ...]
    required_capacity_bytes: int


class OnlineLRUAnalyzer:
    """Incrementally analyze requests while preserving one LRU history."""

    def __init__(
        self,
        *,
        page_bytes: int,
        capacities: Sequence[tuple[str, int]],
        target_hit_rate_ratio: float,
    ) -> None:
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if any(capacity < 0 for _, capacity in capacities):
            raise ValueError("capacity must not be negative")
        if not 0 <= target_hit_rate_ratio <= 1:
            raise ValueError("target_hit_rate_ratio must be between zero and one")
        self._page_bytes = page_bytes
        self._capacities = tuple(capacities)
        self._capacity_pages = tuple(
            capacity // page_bytes for _, capacity in capacities
        )
        self._target_hit_rate_ratio = target_hit_rate_ratio
        self._stack = MattsonStack()
        self._requests: list[RequestAnalysis] = []
        self._infinite_cache_pages = 0
        self.reuse_distance_seconds = 0.0
        self.fixed_capacity_seconds = 0.0
        self.capacity_requirement_seconds = 0.0

    @property
    def request_count(self) -> int:
        return len(self._requests)

    def process(self, page_keys: Sequence[bytes]) -> RequestAnalysis:
        """Compute all configured metrics for one newly arrived request."""
        started_at = time.perf_counter()
        distances = tuple(self._stack.access(key) for key in page_keys)
        self.reuse_distance_seconds += time.perf_counter() - started_at

        started_at = time.perf_counter()
        reusable_accesses = sum(distance is not None for distance in distances)
        self._infinite_cache_pages += len(distances) - reusable_accesses

        page_hits = [0] * len(self._capacity_pages)
        prefix_hits = [0] * len(self._capacity_pages)
        prefix_is_open = [True] * len(self._capacity_pages)
        infinite_prefix_hits = 0
        infinite_prefix_is_open = True
        for distance in distances:
            infinite_hit = distance is not None
            if infinite_prefix_is_open and infinite_hit:
                infinite_prefix_hits += 1
            else:
                infinite_prefix_is_open = False

            for index, pages in enumerate(self._capacity_pages):
                hit = distance is not None and distance < pages
                if hit:
                    page_hits[index] += 1
                if prefix_is_open[index] and hit:
                    prefix_hits[index] += 1
                else:
                    prefix_is_open[index] = False

        request_number = len(self._requests) + 1
        for index, (label, capacity_bytes) in enumerate(self._capacities):
            hit_rate = page_hits[index] / len(distances) if distances else 0.0
            logger.debug(
                f"request {request_number} configured capacity hit-rate estimate: "
                f"capacity={label}, "
                f"page_hits={page_hits[index]}, "
                f"page_accesses={len(distances)}, "
                f"hit_rate={hit_rate:.2f}",
            )
        self.fixed_capacity_seconds += time.perf_counter() - started_at

        started_at = time.perf_counter()
        requirement = _request_capacity_requirement(
            distances,
            request_number=request_number,
            page_bytes=self._page_bytes,
            target_hit_rate_ratio=self._target_hit_rate_ratio,
            infinite_cache_pages=self._infinite_cache_pages,
        )
        self.capacity_requirement_seconds += time.perf_counter() - started_at
        analysis = RequestAnalysis(
            page_accesses=len(distances),
            reusable_accesses=reusable_accesses,
            infinite_page_hits=reusable_accesses,
            infinite_prefix_hits=infinite_prefix_hits,
            capacity_page_hits=tuple(page_hits),
            capacity_prefix_hits=tuple(prefix_hits),
            required_capacity_bytes=requirement,
        )
        self._requests.append(analysis)
        return analysis

    def summarize(
        self, warm_up_requests: int = 0
    ) -> tuple[list[CapacityResult], RequestCapacityStats, int]:
        """Aggregate already-computed request results after a warm-up prefix."""
        if not 0 <= warm_up_requests <= len(self._requests):
            raise ValueError("warm_up_requests must be between zero and request count")

        measured = self._requests[warm_up_requests:]
        total_page_accesses = sum(request.page_accesses for request in measured)
        reusable_accesses = sum(request.reusable_accesses for request in measured)
        infinite_page_hits = sum(request.infinite_page_hits for request in measured)
        infinite_prefix_hits = sum(
            request.infinite_prefix_hits for request in measured
        )
        results = []
        for index, (label, capacity_bytes) in enumerate(self._capacities):
            hits = sum(request.capacity_page_hits[index] for request in measured)
            prefix_hits = sum(
                request.capacity_prefix_hits[index] for request in measured
            )
            results.append(
                CapacityResult(
                    requested_capacity=label,
                    requested_capacity_bytes=capacity_bytes,
                    effective_capacity_bytes=(
                        self._capacity_pages[index] * self._page_bytes
                    ),
                    capacity_pages=self._capacity_pages[index],
                    page_hits=hits,
                    page_misses=total_page_accesses - hits,
                    page_hit_rate=(
                        hits / total_page_accesses if total_page_accesses else 0
                    ),
                    reusable_page_hits=hits,
                    reusable_page_hit_rate=(
                        hits / reusable_accesses if reusable_accesses else 0
                    ),
                    prefix_hit_pages=prefix_hits,
                    prefix_page_hit_rate=(
                        prefix_hits / total_page_accesses
                        if total_page_accesses
                        else 0
                    ),
                )
            )

        results.append(
            CapacityResult(
                requested_capacity="infinite",
                requested_capacity_bytes=None,
                effective_capacity_bytes=None,
                capacity_pages=None,
                page_hits=infinite_page_hits,
                page_misses=total_page_accesses - infinite_page_hits,
                page_hit_rate=(
                    infinite_page_hits / total_page_accesses
                    if total_page_accesses
                    else 0
                ),
                reusable_page_hits=infinite_page_hits,
                reusable_page_hit_rate=(
                    infinite_page_hits / reusable_accesses
                    if reusable_accesses
                    else 0
                ),
                prefix_hit_pages=infinite_prefix_hits,
                prefix_page_hit_rate=(
                    infinite_prefix_hits / total_page_accesses
                    if total_page_accesses
                    else 0
                ),
            )
        )
        requirements = [request.required_capacity_bytes for request in measured]
        stats = _request_capacity_stats_from_requirements(
            requirements,
            target_hit_rate_ratio=self._target_hit_rate_ratio,
        )
        return results, stats, total_page_accesses


def analyze_capacities(
    requests: Sequence[PageRequest],
    *,
    page_bytes: int,
    capacities: Sequence[tuple[str, int]],
    warm_up_requests: int = 0,
    reuse_distances: Sequence[Sequence[int | None]] | None = None,
) -> list[CapacityResult]:
    """Evaluate each finite LRU capacity and an always-present infinite cache."""
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if any(capacity < 0 for _, capacity in capacities):
        raise ValueError("capacity must not be negative")
    if not 0 <= warm_up_requests <= len(requests):
        raise ValueError("warm_up_requests must be between zero and request count")

    if reuse_distances is None:
        reuse_distances = compute_reuse_distances(requests)
    if len(reuse_distances) != len(requests):
        raise ValueError("reuse_distances must match requests")

    measured_requests = requests[warm_up_requests:]
    total_page_accesses = sum(len(request.page_keys) for request in measured_requests)
    capacity_pages = [capacity // page_bytes for _, capacity in capacities]
    page_hits = [0] * len(capacities)
    prefix_hits = [0] * len(capacities)
    reusable_accesses = 0
    infinite_page_hits = 0
    infinite_prefix_hits = 0

    for request_index, distances in enumerate(reuse_distances):
        if request_index < warm_up_requests:
            continue
        prefix_is_open = [True] * len(capacities)
        infinite_prefix_is_open = True
        for distance in distances:
            infinite_hit = distance is not None
            if infinite_hit:
                infinite_page_hits += 1
            if infinite_prefix_is_open and infinite_hit:
                infinite_prefix_hits += 1
            else:
                infinite_prefix_is_open = False

            if distance is not None:
                reusable_accesses += 1
            for index, pages in enumerate(capacity_pages):
                hit = distance is not None and distance < pages
                if hit:
                    page_hits[index] += 1
                if prefix_is_open[index] and hit:
                    prefix_hits[index] += 1
                else:
                    prefix_is_open[index] = False

    results = []
    for index, (label, capacity_bytes) in enumerate(capacities):
        hits = page_hits[index]
        results.append(
            CapacityResult(
                requested_capacity=label,
                requested_capacity_bytes=capacity_bytes,
                effective_capacity_bytes=capacity_pages[index] * page_bytes,
                capacity_pages=capacity_pages[index],
                page_hits=hits,
                page_misses=total_page_accesses - hits,
                page_hit_rate=hits / total_page_accesses if total_page_accesses else 0,
                reusable_page_hits=hits,
                reusable_page_hit_rate=(
                    hits / reusable_accesses if reusable_accesses else 0
                ),
                prefix_hit_pages=prefix_hits[index],
                prefix_page_hit_rate=(
                    prefix_hits[index] / total_page_accesses
                    if total_page_accesses
                    else 0
                ),
            )
        )

    results.append(
        CapacityResult(
            requested_capacity="infinite",
            requested_capacity_bytes=None,
            effective_capacity_bytes=None,
            capacity_pages=None,
            page_hits=infinite_page_hits,
            page_misses=total_page_accesses - infinite_page_hits,
            page_hit_rate=(
                infinite_page_hits / total_page_accesses
                if total_page_accesses
                else 0
            ),
            reusable_page_hits=infinite_page_hits,
            reusable_page_hit_rate=(
                infinite_page_hits / reusable_accesses if reusable_accesses else 0
            ),
            prefix_hit_pages=infinite_prefix_hits,
            prefix_page_hit_rate=(
                infinite_prefix_hits / total_page_accesses
                if total_page_accesses
                else 0
            ),
        )
    )
    return results


def compute_reuse_distances(
    requests: Sequence[PageRequest],
) -> list[tuple[int | None, ...]]:
    """Return per-request LRU reuse distances while preserving trace state."""
    total_page_accesses = sum(len(request.page_keys) for request in requests)
    stack = MattsonStack(total_page_accesses)
    return [
        tuple(stack.access(key) for key in request.page_keys)
        for request in requests
    ]


def request_capacity_stats(
    reuse_distances: Sequence[Sequence[int | None]],
    *,
    page_bytes: int,
    target_hit_rate_ratio: float,
    request_number_offset: int = 0,
    initial_infinite_cache_pages: int = 0,
) -> RequestCapacityStats:
    """Compute the exact prefix-hit capacity required by every request."""
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if not 0 <= target_hit_rate_ratio <= 1:
        raise ValueError("target_hit_rate_ratio must be between zero and one")
    if initial_infinite_cache_pages < 0:
        raise ValueError("initial_infinite_cache_pages must not be negative")

    requirements = []
    infinite_cache_pages = initial_infinite_cache_pages
    for index, distances in enumerate(reuse_distances):
        # A cold access creates one new page. An infinite cache never evicts it,
        # so the number of cold accesses seen so far is its exact footprint.
        infinite_cache_pages += sum(distance is None for distance in distances)
        requirement = _request_capacity_requirement(
            distances,
            request_number=request_number_offset + index + 1,
            page_bytes=page_bytes,
            target_hit_rate_ratio=target_hit_rate_ratio,
            infinite_cache_pages=infinite_cache_pages,
        )
        requirements.append(requirement)
    return _request_capacity_stats_from_requirements(
        requirements,
        target_hit_rate_ratio=target_hit_rate_ratio,
    )


def _request_capacity_stats_from_requirements(
    requirements: Sequence[int],
    *,
    target_hit_rate_ratio: float,
) -> RequestCapacityStats:
    ordered = sorted(requirements)
    return RequestCapacityStats(
        target_hit_rate_ratio=target_hit_rate_ratio,
        samples=len(ordered),
        mean_capacity_bytes=(sum(ordered) / len(ordered) if ordered else None),
        p50_capacity_bytes=_nearest_rank(ordered, 0.50),
        p90_capacity_bytes=_nearest_rank(ordered, 0.90),
        p95_capacity_bytes=_nearest_rank(ordered, 0.95),
        p99_capacity_bytes=_nearest_rank(ordered, 0.99),
        p999_capacity_bytes=_nearest_rank(ordered, 0.999),
    )


def _request_capacity_requirement(
    distances: Sequence[int | None],
    *,
    request_number: int,
    page_bytes: int,
    target_hit_rate_ratio: float,
    infinite_cache_pages: int,
) -> int:
    page_count = len(distances)
    theoretical_prefix_distances = []
    for distance in distances:
        if distance is None:
            break
        theoretical_prefix_distances.append(distance)

    theoretical_prefix_hits = len(theoretical_prefix_distances)
    theoretical_hit_rate = (
        theoretical_prefix_hits / page_count if page_count else 0.0
    )
    target_ratio = Fraction(str(target_hit_rate_ratio))
    target_prefix_hits = (
        theoretical_prefix_hits * target_ratio.numerator
        + target_ratio.denominator
        - 1
    ) // target_ratio.denominator
    target_hit_rate = target_prefix_hits / page_count if page_count else 0.0

    if target_prefix_hits:
        maximum_lru_depth = max(
            theoretical_prefix_distances[:target_prefix_hits]
        )
        required_pages = maximum_lru_depth + 1
    else:
        maximum_lru_depth = None
        required_pages = 0
    requirement = required_pages * page_bytes

    logger.debug(
        f"request {request_number} capacity estimate: pages={page_count}, "
        f"theoretical_prefix_hits={theoretical_prefix_hits}, "
        f"theoretical_hit_rate={theoretical_hit_rate:.2f}, "
        f"target_prefix_hits={target_prefix_hits}, "
        f"target_hit_rate={target_hit_rate:.2f}, "
        f"maximum_lru_depth={maximum_lru_depth}, "
        f"required_capacity={requirement / 2**30:.2f} Gib, "
        f"accumulated_cache_capacity={infinite_cache_pages * page_bytes / 2**30:.2f} Gib",
    )
    return requirement


def _nearest_rank(ordered: Sequence[int], percentile: float) -> int | None:
    if not ordered:
        return None
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]
