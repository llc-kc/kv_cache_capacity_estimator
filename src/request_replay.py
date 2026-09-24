# SPDX-License-Identifier: Apache-2.0
"""Tokenize and replay OpenAI-compatible requests."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable

from kv_capacity_estimator.cli_args import (
    add_simulation_arguments,
    log_arguments,
    log_request_details,
    log_replay_progress,
    print_simulation,
    simulation_config,
)
from kv_capacity_estimator.simulator import simulate
from kv_capacity_estimator.tokenize import tokenize_openai_trace
from kv_capacity_estimator.tokenize.cli import (
    add_tokenizer_arguments,
    build_tokenizer,
    close_tokenizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulation_arguments(parser)
    add_tokenizer_arguments(parser)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    command_started_at = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    log_arguments(args)
    tokenizer_started_at = time.perf_counter()
    tokenizer = build_tokenizer(args, parser)
    tokenizer_init_seconds = time.perf_counter() - tokenizer_started_at

    try:
        start_time = time.perf_counter()
        logging.info("begin replay")
        token_requests = log_replay_progress(
            tokenize_openai_trace(
                args.trace,
                tokenizer,
                workers=args.tokenize_workers,
            ),
            args.progress_interval,
        )
        token_requests = log_request_details(token_requests, args.log_token_ids)
        result = simulate(token_requests, simulation_config(args))
        end_time = time.perf_counter()
        logging.info("finish replay, time used: %.2f seconds", end_time - start_time)
        timings = result.timings
        logging.info(
            "timing: tokenizer_init=%.3fs, tokenize=%.3fs, page_build=%.3fs, "
            "reuse_distance=%.3fs, fixed_capacity=%.3fs, "
            "capacity_requirement=%.3fs, simulation_total=%.3fs, command_total=%.3fs",
            tokenizer_init_seconds,
            timings.input_processing_seconds,
            timings.page_build_seconds,
            timings.reuse_distance_seconds,
            timings.fixed_capacity_seconds,
            timings.capacity_requirement_seconds,
            timings.total_seconds,
            time.perf_counter() - command_started_at,
        )
    finally:
        close_tokenizer(tokenizer)
    print_simulation(result, args.out_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
