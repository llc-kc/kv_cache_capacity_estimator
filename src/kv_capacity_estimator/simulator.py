# SPDX-License-Identifier: Apache-2.0
"""Source-agnostic KV capacity estimation orchestration."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .analysis import OnlineLRUAnalyzer, RequestAnalysis
from .models import CapacityResult, RequestCapacityStats, TokenIdsRequest
from .pages import chained_page_hashes


@dataclass(frozen=True)
class SimulationConfig:
    page_size: int
    kv_bytes_per_token: float
    capacities: Sequence[tuple[str, int]]
    include_partial_page: bool = False
    warm_up_ratio: float = 0.0
    target_hit_rate_ratio: float = 0.99

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")
        if not self.capacities:
            raise ValueError("at least one capacity is required")
        if not 0 <= self.warm_up_ratio <= 1:
            raise ValueError("warm_up_ratio must be between zero and one")
        if not 0 <= self.target_hit_rate_ratio <= 1:
            raise ValueError("target_hit_rate_ratio must be between zero and one")


@dataclass(frozen=True)
class SimulationResult:
    page_size_tokens: int
    kv_bytes_per_token: float
    page_bytes: int
    total_requests: int
    warm_up_requests: int
    requests: int
    page_accesses: int
    results: tuple[CapacityResult, ...]
    request_capacity_stats: RequestCapacityStats
    timings: SimulationTimings


@dataclass(frozen=True)
class SimulationTimings:
    """Wall-clock time spent in each simulation stage."""

    input_processing_seconds: float
    page_build_seconds: float
    reuse_distance_seconds: float
    fixed_capacity_seconds: float
    capacity_requirement_seconds: float
    total_seconds: float


class ReplaySimulator:
    """Stateful simulator that accepts token-ID requests one at a time."""

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._started_at = time.perf_counter()
        self._page_bytes = math.ceil(
            config.kv_bytes_per_token * config.page_size
        )
        self._analyzer = OnlineLRUAnalyzer(
            page_bytes=self._page_bytes,
            capacities=config.capacities,
            target_hit_rate_ratio=config.target_hit_rate_ratio,
        )
        self._input_processing_seconds = 0.0
        self._page_build_seconds = 0.0
        self._finished = False

    @property
    def request_count(self) -> int:
        return self._analyzer.request_count

    def process(self, request: TokenIdsRequest) -> RequestAnalysis:
        """Process one request immediately and retain only its compact metrics."""
        if self._finished:
            raise RuntimeError("cannot process requests after finish")
        started_at = time.perf_counter()
        page_keys = chained_page_hashes(
            request.input_ids,
            page_size=self._config.page_size,
            include_partial_page=self._config.include_partial_page,
        )
        self._page_build_seconds += time.perf_counter() - started_at
        return self.process_page_keys(page_keys)

    def process_page_keys(
        self, page_keys: Sequence[bytes]
    ) -> RequestAnalysis:
        """Update the LRU history from prebuilt chained page keys.

        This is equivalent to :meth:`process` after the caller has converted
        a request's token IDs into chained page hashes, for example with
        :func:`chained_page_hashes`. Use this entry point when page hashing
        can be parallelized outside the serialized simulation step, so that
        only the shared LRU-history update is performed here.
        """
        if self._finished:
            raise RuntimeError("cannot process requests after finish")
        return self._analyzer.process(tuple(page_keys))

    def finish(self) -> SimulationResult:
        """Finalize aggregate metrics after all requests have arrived."""
        if self._finished:
            raise RuntimeError("simulation has already finished")
        self._finished = True
        total_requests = self._analyzer.request_count
        warm_up_requests = math.floor(
            total_requests * self._config.warm_up_ratio
        )
        capacity_results, capacity_stats, page_accesses = (
            self._analyzer.summarize(warm_up_requests)
        )
        return SimulationResult(
            page_size_tokens=self._config.page_size,
            kv_bytes_per_token=self._config.kv_bytes_per_token,
            page_bytes=self._page_bytes,
            total_requests=total_requests,
            warm_up_requests=warm_up_requests,
            requests=total_requests - warm_up_requests,
            page_accesses=page_accesses,
            results=tuple(capacity_results),
            request_capacity_stats=capacity_stats,
            timings=SimulationTimings(
                input_processing_seconds=self._input_processing_seconds,
                page_build_seconds=self._page_build_seconds,
                reuse_distance_seconds=self._analyzer.reuse_distance_seconds,
                fixed_capacity_seconds=self._analyzer.fixed_capacity_seconds,
                capacity_requirement_seconds=(
                    self._analyzer.capacity_requirement_seconds
                ),
                total_seconds=time.perf_counter() - self._started_at,
            ),
        )

    def _record_input_processing(self, elapsed_seconds: float) -> None:
        self._input_processing_seconds += elapsed_seconds


def simulate(
    token_requests: Iterable[TokenIdsRequest], config: SimulationConfig
) -> SimulationResult:
    replay = ReplaySimulator(config)
    requests = iter(token_requests)
    while True:
        started_at = time.perf_counter()
        try:
            request = next(requests)
        except StopIteration:
            replay._record_input_processing(time.perf_counter() - started_at)
            break
        replay._record_input_processing(time.perf_counter() - started_at)
        replay.process(request)
    return replay.finish()
