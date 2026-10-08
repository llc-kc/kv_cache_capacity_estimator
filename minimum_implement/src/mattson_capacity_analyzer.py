# SPDX-License-Identifier: Apache-2.0
"""Build KV-cache capacity/hit-rate curves with Mattson stack distances.

The default input parser reads JSON Lines. A row may be either an input-id
array or an object::

    [1, 2, 3, 4]
    {"input_ids": [1, 2, 3, 4]}

Only full pages are included by default. Page keys are prefix chained, so the
same token block under two different preceding contexts has different keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from trace_parser import TraceParser, parse_jsonl, resolve_parser


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


@dataclass(frozen=True)
class Request:
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


class FenwickTree:
    """Fenwick tree storing one marker at every key's latest access position."""

    def __init__(self, size: int) -> None:
        self._tree = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        tree = self._tree
        while index < len(tree):
            tree[index] += delta
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        result = 0
        tree = self._tree
        while index > 0:
            result += tree[index]
            index -= index & -index
        return result


class MattsonStack:
    """Compute exact LRU stack distance in O(log N) per page access."""

    def __init__(self, max_accesses: int) -> None:
        self._latest_positions: dict[bytes, int] = {}
        self._positions = FenwickTree(max_accesses)
        self._time = 0

    def access(self, key: bytes) -> int | None:
        """Return distinct-page reuse distance, or None for a cold access."""
        self._time += 1
        old_position = self._latest_positions.get(key)
        if old_position is None:
            distance = None
        else:
            # Active keys whose latest access is newer than this key are exactly
            # the distinct keys above it in the conceptual infinite LRU stack.
            distance = self._positions.prefix_sum(
                self._time - 1
            ) - self._positions.prefix_sum(old_position)
            self._positions.add(old_position, -1)
        self._positions.add(self._time, 1)
        self._latest_positions[key] = self._time
        return distance


def parse_size(value: str) -> int:
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(f"invalid capacity {value!r}; examples: 10GB, 100GiB")
    number = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else None
    return math.floor(number * _SIZE_MULTIPLIERS[unit])


def chained_page_hashes(
    input_ids: Sequence[int],
    *,
    page_size: int,
    include_partial_page: bool = False,
) -> tuple[bytes, ...]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    chain = hashlib.sha256(b"kv-prefix-v1\0").digest()
    page_keys: list[bytes] = []
    for start in range(0, len(input_ids), page_size):
        page = input_ids[start : start + page_size]
        if len(page) < page_size and not include_partial_page:
            break
        payload = struct.pack("<I", len(page)) + b"".join(
            struct.pack("<Q", int(token_id)) for token_id in page
        )
        chain = hashlib.sha256(chain + payload).digest()
        page_keys.append(chain)
    return tuple(page_keys)


def load_requests(
    path: Path,
    *,
    page_size: int,
    include_partial_page: bool = False,
    trace_parser: TraceParser = parse_jsonl,
) -> list[Request]:
    return [
        Request(
            chained_page_hashes(
                parsed.input_ids,
                page_size=page_size,
                include_partial_page=include_partial_page,
            )
        )
        for parsed in trace_parser(path)
    ]


def analyze_capacities(
    requests: Sequence[Request],
    *,
    page_bytes: int,
    capacities: Sequence[tuple[str, int]],
) -> list[CapacityResult]:
    """Evaluate every requested LRU capacity plus an infinite-capacity cache."""
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    total_page_accesses = sum(len(request.page_keys) for request in requests)
    stack = MattsonStack(total_page_accesses)
    capacity_pages = [capacity_bytes // page_bytes for _, capacity_bytes in capacities]
    page_hits = [0] * len(capacities)
    reusable_page_hits = [0] * len(capacities)
    prefix_hits = [0] * len(capacities)
    reusable_accesses = 0
    seen_keys: set[bytes] = set()
    infinite_page_hits = 0
    infinite_prefix_hits = 0

    for request in requests:
        if not request.page_keys:
            continue
        prefix_is_open = [True] * len(capacities)
        infinite_prefix_is_open = True
        for key in request.page_keys:
            # An infinite cache never evicts, so membership alone determines
            # whether this access hits; no stack-distance comparison is needed.
            infinite_hit = key in seen_keys
            seen_keys.add(key)
            if infinite_hit:
                infinite_page_hits += 1
            if infinite_prefix_is_open and infinite_hit:
                infinite_prefix_hits += 1
            else:
                infinite_prefix_is_open = False

            distance = stack.access(key)
            if distance is not None:
                reusable_accesses += 1
            for index, pages in enumerate(capacity_pages):
                hit = distance is not None and distance < pages
                if hit:
                    page_hits[index] += 1
                    reusable_page_hits[index] += 1
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
                page_hit_rate=(
                    hits / total_page_accesses if total_page_accesses else 0
                ),
                reusable_page_hits=reusable_page_hits[index],
                reusable_page_hit_rate=(
                    reusable_page_hits[index] / reusable_accesses
                    if reusable_accesses
                    else 0
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


def _human_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f}{unit}" if unit != "B" else f"{value}B"
        value /= 1024
    raise AssertionError("unreachable")


def _print_table(results: Sequence[CapacityResult]) -> None:
    header = (
        f"{'Capacity':>12} {'Pages':>12} {'Page hit':>11} "
        f"{'Reuse hit':>11} {'Prefix hit':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        pages = (
            "infinite"
            if result.capacity_pages is None
            else f"{result.capacity_pages:,d}"
        )
        print(
            f"{result.requested_capacity:>12} "
            f"{pages:>12} "
            f"{result.page_hit_rate:>10.2%} "
            f"{result.reusable_page_hit_rate:>10.2%} "
            f"{result.prefix_page_hit_rate:>11.2%}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="request input-ID trace")
    parser.add_argument(
        "--trace-parser",
        default="jsonl",
        metavar="NAME|MODULE:FUNCTION",
        help=(
            "input parser: built-in 'jsonl' (default), or an importable "
            "module:function accepting a Path and yielding ParsedRequest objects"
        ),
    )
    parser.add_argument(
        "--kv-bytes-per-token",
        type=float,
        required=True,
        help="physical L3 bytes occupied by one token's KV cache",
    )
    parser.add_argument(
        "--page-size", 
        type=int, 
        default=64, 
        help="tokens per page, used to reduce the nubmer of kv instances and thus accelerate the analysis"
    )
    parser.add_argument(
        "--capacities",
        required=True,
        help="comma-separated capacities, for example 10GB,100GB,1TiB",
    )
    parser.add_argument("--include-partial-page", action="store_true")
    parser.add_argument("--out-format", choices=("table", "json"), default="table")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.kv_bytes_per_token <= 0:
        raise ValueError("kv_bytes_per_token must be positive")
    page_bytes = math.ceil(args.kv_bytes_per_token * args.page_size)
    capacity_labels = [item.strip() for item in args.capacities.split(",")]
    capacities = [(label, parse_size(label)) for label in capacity_labels if label]
    if not capacities:
        raise ValueError("at least one capacity is required")
    requests = load_requests(
        args.trace,
        page_size=args.page_size,
        include_partial_page=args.include_partial_page,
        trace_parser=resolve_parser(args.trace_parser),
    )
    results = analyze_capacities(
        requests, page_bytes=page_bytes, capacities=capacities
    )
    if args.out_format == "json":
        output = {
            "page_size_tokens": args.page_size,
            "kv_bytes_per_token": args.kv_bytes_per_token,
            "page_bytes": page_bytes,
            "requests": len(requests),
            "page_accesses": sum(len(request.page_keys) for request in requests),
            "results": [asdict(result) for result in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(
            f"requests={len(requests)}, "
            f"page_accesses={sum(len(r.page_keys) for r in requests)}, "
            f"page_bytes={_human_bytes(page_bytes)}"
        )
        _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
