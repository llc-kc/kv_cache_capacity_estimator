# SPDX-License-Identifier: Apache-2.0
"""Replay an offline token-ID trace without a tokenizer dependency."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable

from kv_cache_simulator.cli_args import (
    add_simulation_arguments,
    log_arguments,
    log_request_details,
    log_replay_progress,
    print_simulation,
    simulation_config,
)
from kv_cache_simulator.simulator import simulate
from kv_cache_simulator.token_id_replay import resolve_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulation_arguments(parser)
    parser.add_argument(
        "--trace-parser",
        default="jsonl",
        metavar="NAME|MODULE:FUNCTION",
        help="built-in 'jsonl' or an importable module:function",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    log_arguments(args)
    token_requests = log_replay_progress(
        resolve_parser(args.trace_parser)(args.trace),
        args.progress_interval,
    )
    token_requests = log_request_details(token_requests, args.log_token_ids)
    start_time = time.time()
    logging.info("begin replay")
    result = simulate(token_requests, simulation_config(args))
    print_simulation(result, args.out_format)
    end_time = time.time()
    logging.info("finish replay, time used: %.2f seconds", end_time - start_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
