# SPDX-License-Identifier: Apache-2.0
"""Convert OpenAI-compatible request JSONL to token-ID JSONL."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from kv_capacity_estimator.cli_args import log_replay_progress
from kv_capacity_estimator.models import TokenIdsRequest
from kv_capacity_estimator.tokenize import tokenize_openai_trace
from kv_capacity_estimator.tokenize.cli import (
    add_tokenizer_arguments,
    build_tokenizer,
    close_tokenizer,
)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="OpenAI request JSONL input")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "token-ID JSONL output; omit to write next to the input trace as "
            "<trace-stem>.token_ids<trace-suffix>"
        ),
    )
    parser.add_argument(
        "--log-req-interval",
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
    )
    add_tokenizer_arguments(parser)
    return parser


def write_token_ids(
    requests: Iterable[TokenIdsRequest], output_file: TextIO
) -> int:
    """Write one compact token-ID array per request and return the row count."""
    count = 0
    for request in requests:
        json.dump(list(request.input_ids), output_file, separators=(",", ":"))
        output_file.write("\n")
        count += 1
    return count


def default_output_path(trace: Path) -> Path:
    """Return the default output path next to ``trace``.

    The ``.token_ids`` marker is inserted before the original suffix, e.g.
    ``requests.jsonl`` -> ``requests.token_ids.jsonl``. A trace without a
    suffix simply gets ``.token_ids`` appended.
    """
    if trace.suffix:
        return trace.with_name(f"{trace.stem}.token_ids{trace.suffix}")
    return trace.with_name(f"{trace.name}.token_ids")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = default_output_path(args.trace)
    if args.trace.resolve() == args.output.resolve():
        parser.error("input and output paths must be different")
    logging.basicConfig(level=getattr(logging, args.log_level))
    tokenizer = build_tokenizer(args, parser)
    try:
        requests = tokenize_openai_trace(
            args.trace,
            tokenizer,
            workers=args.tokenize_workers,
            on_error="empty",
        )
        requests = log_replay_progress(requests, args.log_req_interval)
        with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
            count = write_token_ids(requests, output_file)
    finally:
        close_tokenizer(tokenizer)
    logging.info("wrote %d token-ID requests to %s", count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
