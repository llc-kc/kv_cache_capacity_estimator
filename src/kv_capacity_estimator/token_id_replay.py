# SPDX-License-Identifier: Apache-2.0
"""Offline token-ID trace input with no tokenizer dependency."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

from .models import TokenIdsRequest


TokenIdsParser = Callable[[Path], Iterable[TokenIdsRequest]]


def as_token_ids_request(value: object) -> TokenIdsRequest:
    if isinstance(value, list):
        input_ids = value
    elif isinstance(value, dict):
        input_ids = value.get(
            "input_ids", value.get("token_ids", value.get("tokens"))
        )
        if input_ids is None:
            raise ValueError("request object is missing input_ids")
    else:
        raise ValueError("request must be a token-ID array or an object")

    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        raise ValueError("input_ids must be an array of integers")
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or not 0 <= token_id < 2**64
        for token_id in input_ids
    ):
        raise ValueError("input_ids must contain unsigned 64-bit integers")

    return TokenIdsRequest(tuple(input_ids))


def parse_jsonl(path: Path) -> Iterator[TokenIdsRequest]:
    """Parse one request or a batch of requests from each JSONL row."""
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if (
                    isinstance(row, list)
                    and row
                    and not all(
                        isinstance(token_id, int)
                        and not isinstance(token_id, bool)
                        for token_id in row
                    )
                ):
                    for value in row:
                        yield as_token_ids_request(value)
                else:
                    yield as_token_ids_request(row)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"line {line_number}: {error}") from error


BUILTIN_PARSERS: dict[str, TokenIdsParser] = {"jsonl": parse_jsonl}


def resolve_parser(spec: str) -> TokenIdsParser:
    builtin = BUILTIN_PARSERS.get(spec)
    if builtin is not None:
        return builtin
    if ":" not in spec:
        names = ", ".join(sorted(BUILTIN_PARSERS))
        raise ValueError(
            f"unknown trace parser {spec!r}; use one of [{names}] or module:function"
        )
    module_name, attribute_name = spec.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("custom trace parser must use module:function syntax")
    try:
        parser = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"cannot load trace parser {spec!r}: {error}") from error
    if not callable(parser):
        raise ValueError(f"trace parser {spec!r} is not callable")
    return parser
