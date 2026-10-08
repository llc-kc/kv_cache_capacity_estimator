# SPDX-License-Identifier: Apache-2.0
"""Pluggable parsers for KV-cache request traces.

A parser accepts a trace path and yields :class:`ParsedRequest` objects.  The
analyzer deliberately knows nothing about the trace's on-disk representation.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class ParsedRequest:
    """Storage-independent request representation consumed by the analyzer."""

    input_ids: tuple[int, ...]


TraceParser = Callable[[Path], Iterable[ParsedRequest]]


def _as_parsed_request(value: object) -> ParsedRequest:
    if isinstance(value, list):
        input_ids = value
    elif isinstance(value, dict):
        input_ids = value.get(
            "input_ids", value.get("token_ids", value.get("tokens"))
        )
        if input_ids is None:
            raise ValueError("request object is missing input_ids")
    else:
        raise ValueError("request must be an input-ID array or an object")

    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        raise ValueError("input_ids must be an array of integers")
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in input_ids
    ):
        raise ValueError("input_ids must contain only integers")
    return ParsedRequest(tuple(input_ids))


def parse_jsonl(path: Path) -> Iterator[ParsedRequest]:
    """Parse JSON Lines containing arrays, objects, or batches of either.

    In particular, a line such as ``[1, 2, 3]`` is one request, matching the
    format used by ``test_data/req_input_ids.log``.
    """

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, list) and row and not isinstance(row[0], int):
                    values = row
                else:
                    values = [row]
                for value in values:
                    yield _as_parsed_request(value)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"line {line_number}: {error}") from error


BUILTIN_PARSERS: dict[str, TraceParser] = {"jsonl": parse_jsonl}


def resolve_parser(spec: str) -> TraceParser:
    """Resolve a built-in name or a ``module:function`` parser reference."""

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
