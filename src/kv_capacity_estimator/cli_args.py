# SPDX-License-Identifier: Apache-2.0
"""Command-line arguments and output shared by both replay tools."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from .config import parse_size
from .defaults import (
    DEFAULT_CAPACITIES,
    DEFAULT_PAGE_SIZE,
)
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
        default=0.99,
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
        "--progress-interval",
        type=_non_negative_int,
        default=100,
        metavar="REQUESTS",
        help="log progress every N requests; use 0 to disable (default: 100)",
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


def log_replay_progress(
    requests: Iterable[_RequestT], interval: int
) -> Iterator[_RequestT]:
    """Yield requests while periodically logging replay throughput."""
    started_at = time.perf_counter()
    processed = 0
    for processed, request in enumerate(requests, 1):
        yield request
        if interval and processed % interval == 0:
            _log_progress(processed, started_at)
    if interval and processed % interval:
        _log_progress(processed, started_at)


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


def _log_progress(processed: int, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    rate = processed / elapsed if elapsed else 0.0
    logger.info(
        "replay progress: requests=%d, elapsed=%.2fs, rate=%.2f requests/s",
        processed,
        elapsed,
        rate,
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
