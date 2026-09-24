# SPDX-License-Identifier: Apache-2.0
"""Convert token-ID requests into prefix-chained KV page accesses."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence

from .models import PageRequest, TokenIdsRequest


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
            struct.pack("<Q", token_id) for token_id in page
        )
        chain = hashlib.sha256(chain + payload).digest()
        page_keys.append(chain)
    return tuple(page_keys)


def requests_from_token_ids(
    requests: Iterable[TokenIdsRequest],
    *,
    page_size: int,
    include_partial_page: bool = False,
) -> list[PageRequest]:
    return [
        PageRequest(
            chained_page_hashes(
                request.input_ids,
                page_size=page_size,
                include_partial_page=include_partial_page,
            )
        )
        for request in requests
    ]
