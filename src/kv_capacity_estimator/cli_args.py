# SPDX-License-Identifier: Apache-2.0
"""Command-line arguments and output shared by both replay tools."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from .config import parse_size
from .defaults import (
    DEFAULT_CAPACITIES,
    DEFAULT_PAGE_SIZE,
)
from .models import TokenIdsRequest
from .pages import chained_page_hashes
from .reporting import format_json, format_table
from .simulator import SimulationConfig, SimulationResult


_RequestT = TypeVar("_RequestT")
logger = logging.getLogger(__name__)
_SENSITIVE_ARGUMENTS = frozenset({"api_key"})


def add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("trace", type=Path, help="path to the JSONL trace")
    parser.add_argument("--kv-bytes-per-token", type=float, required=True)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--capacities",
        default=DEFAULT_CAPACITIES,
        help=(
            "comma-separated capacities (default: "
            f"{DEFAULT_CAPACITIES})"
        ),
    )
    parser.add_argument("--include-partial-page", action="store_true")
    parser.add_argument(
        "--target-hit-rate-percent",
        "--expected-hit-rate-percent",
        type=_percentage,
        default=1.0,
        metavar="PERCENT",
        help=(
            "per-request target as a percentage of theoretical hit rate "
            "(default: 99%%)"
        ),
    )
    parser.add_argument(
        "--warm-up-percent",
        "--warm_up",
        type=_percentage,
        default=0.5,
        metavar="PERCENT",
        help="leading requests used only to initialize cache state (default: 50%%)",
    )
    parser.add_argument(
        "--log-token-ids",
        action="store_true",
        help="log each request's token IDs at INFO level",
    )
    parser.add_argument("--out-format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--log-req-interval",
        "--progress-interval",
        type=_non_negative_int,
        default=100,
        metavar="REQUESTS",
        help=(
            "log replay progress every N requests, including the theoretical "
            "prefix token hit rate over those N requests; use 0 to disable "
            "(default: 100)"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="logging level; DEBUG prints request and capacity details",
    )


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _percentage(value: str) -> float:
    normalized = value.strip()
    if normalized.endswith("%"):
        normalized = normalized[:-1].strip()
    try:
        parsed = float(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a percentage from 0 to 100"
        ) from error
    if not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be a percentage from 0 to 100")
    return parsed / 100


class TheoreticalPrefixHitTracker:
    """Track the token-level theoretical prefix hit rate over a request window.

    The theoretical prefix of a request is its leading run of pages that have
    already appeared in an earlier request. Each fully covered page contributes
    ``page_size`` tokens; a covered partial page contributes its real token
    count. The hit rate of a window is therefore the covered-token increment
    divided by the total-token increment of the requests in that window.
    """

    def __init__(
        self, *, page_size: int, include_partial_page: bool = False
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self._page_size = page_size
        self._include_partial_page = include_partial_page
        self._seen_pages: set[bytes] = set()
        self._window: deque[tuple[int, int]] = deque()
        self._window_hit_tokens = 0
        self._window_total_tokens = 0

    def record(self, request: TokenIdsRequest) -> tuple[int, int]:
        """Record one request and return its ``(hit_tokens, total_tokens)``."""
        input_ids = request.input_ids
        total_tokens = len(input_ids)
        page_keys = chained_page_hashes(
            input_ids,
            page_size=self._page_size,
            include_partial_page=self._include_partial_page,
        )
        hit_tokens = 0
        for index, page_key in enumerate(page_keys):
            if page_key in self._seen_pages:
                start = index * self._page_size
                hit_tokens += min(self._page_size, total_tokens - start)
            else:
                # Prefix sharing only covers the leading contiguous run.
                break
        self._seen_pages.update(page_keys)
        self._window.append((hit_tokens, total_tokens))
        self._window_hit_tokens += hit_tokens
        self._window_total_tokens += total_tokens
        return hit_tokens, total_tokens

    def advance_window(self, max_requests: int) -> None:
        """Drop the oldest requests until at most ``max_requests`` remain."""
        while len(self._window) > max_requests:
            hit_tokens, total_tokens = self._window.popleft()
            self._window_hit_tokens -= hit_tokens
            self._window_total_tokens -= total_tokens

    @property
    def window_requests(self) -> int:
        return len(self._window)

    @property
    def window_hit_rate(self) -> float:
        if not self._window_total_tokens:
            return 0.0
        return self._window_hit_tokens / self._window_total_tokens


def log_replay_progress(
    requests: Iterable[_RequestT],
    interval: int,
    *,
    page_size: int | None = None,
    include_partial_page: bool = False,
) -> Iterator[_RequestT]:
    """Yield requests while periodically logging replay throughput.

    When ``page_size`` is provided, the progress log also reports the
    theoretical prefix token hit rate across the most recent ``interval``
    requests (covered-token increment divided by total-token increment).
    Requests without ``input_ids`` are counted but excluded from that metric.
    """
    started_at = time.perf_counter()
    processed = 0
    tracker: TheoreticalPrefixHitTracker | None = (
        TheoreticalPrefixHitTracker(
            page_size=page_size, include_partial_page=include_partial_page
        )
        if interval and page_size is not None
        else None
    )
    for processed, request in enumerate(requests, 1):
        yield request
        if not interval:
            continue
        if tracker is not None:
            input_ids = getattr(request, "input_ids", None)
            if input_ids is not None:
                tracker.record(TokenIdsRequest(tuple(input_ids)))
                tracker.advance_window(interval)
        if processed % interval == 0:
            _log_progress(processed, started_at, tracker)
    if interval and processed % interval:
        _log_progress(processed, started_at, tracker)


def log_arguments(args: argparse.Namespace) -> None:
    """Log parsed replay arguments without exposing credential values."""
    values = {
        name: (
            "***"
            if name in _SENSITIVE_ARGUMENTS and value is not None
            else value
        )
        for name, value in vars(args).items()
    }
    logger.info(
        "replay arguments: %s",
        json.dumps(values, default=str, ensure_ascii=False, sort_keys=True),
    )


def log_request_details(
    requests: Iterable[_RequestT], log_token_ids: bool = False
) -> Iterator[_RequestT]:
    """Log request lengths at DEBUG and optionally token IDs at INFO."""
    for request_number, request in enumerate(requests, 1):
        input_ids = getattr(request, "input_ids")
        logger.debug("request %d length=%d tokens", request_number, len(input_ids))
        if log_token_ids:
            logger.info("request %d token_ids=%s", request_number, list(input_ids))
        yield request


def _log_progress(
    processed: int,
    started_at: float,
    tracker: TheoreticalPrefixHitTracker | None = None,
) -> None:
    elapsed = time.perf_counter() - started_at
    rate = processed / elapsed if elapsed else 0.0
    if tracker is None or tracker.window_requests == 0:
        logger.info(
            "replay progress: requests=%d, elapsed=%.2fs, rate=%.2f requests/s",
            processed,
            elapsed,
            rate,
        )
        return
    logger.info(
        "replay progress: requests=%d, elapsed=%.2fs, rate=%.2f requests/s, "
        "window_requests=%d, theoretical_prefix_hit_rate=%.2f%%",
        processed,
        elapsed,
        rate,
        tracker.window_requests,
        tracker.window_hit_rate * 100,
    )


def simulation_config(args: argparse.Namespace) -> SimulationConfig:
    labels = [item.strip() for item in args.capacities.split(",") if item.strip()]
    return SimulationConfig(
        page_size=args.page_size,
        kv_bytes_per_token=args.kv_bytes_per_token,
        capacities=[(label, parse_size(label)) for label in labels],
        include_partial_page=args.include_partial_page,
        warm_up_ratio=args.warm_up_percent,
        target_hit_rate_ratio=args.target_hit_rate_percent,
    )


def print_simulation(result: SimulationResult, out_format: str) -> None:
    print(format_json(result) if out_format == "json" else format_table(result))
