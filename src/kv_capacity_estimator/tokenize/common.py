# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for tokenizing OpenAI-compatible request traces."""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from ..models import TokenIdsRequest


logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    def tokenize(self, request: Mapping[str, Any]) -> TokenIdsRequest: ...


def request_body(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the request body from a direct request or an OpenAI Batch row."""
    body = request.get("body", request)
    if not isinstance(body, Mapping):
        raise ValueError("OpenAI batch request body must be an object")
    return body


def validate_prompt_kind(body: Mapping[str, Any]) -> None:
    has_prompt = body.get("prompt") is not None
    has_messages = body.get("messages") is not None
    if has_prompt == has_messages:
        raise ValueError("request must contain exactly one of prompt or messages")


def parse_openai_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read direct OpenAI requests or Batch API rows from JSON Lines."""
    for _, value in _iter_openai_jsonl(path):
        yield value


def _iter_openai_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("request must be a JSON object")
                yield line_number, value
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"line {line_number}: {error}") from error


def tokenize_openai_trace(
    path: Path, tokenizer: Tokenizer, *, workers: int = 1
) -> Iterator[TokenIdsRequest]:
    """Tokenize a trace concurrently while yielding results in input order."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    succeeded = 0
    failed = 0
    if workers == 1:
        for line_number, request in _iter_openai_jsonl(path):
            try:
                token_request = _tokenize_request(line_number, request, tokenizer)
            except (RuntimeError, ValueError) as error:
                failed += 1
                logger.error("skipping request after tokenization error: %s", error)
            else:
                succeeded += 1
                yield token_request
        logger.info(
            "tokenization summary: succeeded=%d, failed=%d", succeeded, failed
        )
        return

    # Keep submission bounded so completed out-of-order requests cannot retain
    # an entire trace's token IDs while an earlier request is still running.
    max_in_flight = workers * 2
    pending: deque[Future[tuple[int, TokenIdsRequest]]] = deque()
    requests = iter(_iter_openai_jsonl(path))
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="request-tokenizer",
    )
    try:
        for _ in range(max_in_flight):
            try:
                line_number, request = next(requests)
            except StopIteration:
                break
            pending.append(
                executor.submit(
                    _tokenize_request_with_line,
                    line_number,
                    request,
                    tokenizer,
                )
            )

        while pending:
            try:
                _line_number, token_request = pending.popleft().result()
            except (RuntimeError, ValueError) as error:
                failed += 1
                logger.error("skipping request after tokenization error: %s", error)
            else:
                succeeded += 1
                yield token_request
            try:
                line_number, request = next(requests)
            except StopIteration:
                continue
            pending.append(
                executor.submit(
                    _tokenize_request_with_line,
                    line_number,
                    request,
                    tokenizer,
                )
            )
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    logger.info("tokenization summary: succeeded=%d, failed=%d", succeeded, failed)


def _tokenize_request_with_line(
    line_number: int,
    request: Mapping[str, Any],
    tokenizer: Tokenizer,
) -> tuple[int, TokenIdsRequest]:
    return line_number, _tokenize_request(line_number, request, tokenizer)


def _tokenize_request(
    line_number: int,
    request: Mapping[str, Any],
    tokenizer: Tokenizer,
) -> TokenIdsRequest:
    try:
        return tokenizer.tokenize(request)
    except (RuntimeError, ValueError) as error:
        raise type(error)(f"line {line_number}: {error}") from error
