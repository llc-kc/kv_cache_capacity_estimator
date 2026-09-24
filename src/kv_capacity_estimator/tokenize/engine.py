# SPDX-License-Identifier: Apache-2.0
"""Tokenize OpenAI-compatible requests through an engine endpoint."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from ..models import TokenIdsRequest
from ..token_id_replay import as_token_ids_request
from .common import request_body, validate_prompt_kind


_TOKENIZE_FIELDS = {
    "model",
    "prompt",
    "messages",
    "tools",
    "continue_final_message",
    "chat_template",
    "chat_template_kwargs",
    "add_generation_prompt",
    "add_special_tokens",
    "mm_processor_kwargs",
}


class EngineTokenizer:
    """HTTP client for the SGLang/vLLM ``/tokenize`` endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        model: str | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.url = _tokenize_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.model = model

    def tokenize(self, request: Mapping[str, Any]) -> TokenIdsRequest:
        payload = build_tokenize_payload(request, model=self.model)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"tokenize endpoint returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"cannot call tokenize endpoint {self.url}: {error}") from error
        except OSError as error:
            raise RuntimeError(f"cannot call tokenize endpoint {self.url}: {error}") from error

        if not isinstance(result, dict) or "tokens" not in result:
            raise ValueError("tokenize response is missing tokens")
        tokens = result["tokens"]
        if tokens and isinstance(tokens, list) and isinstance(tokens[0], list):
            if len(tokens) != 1:
                raise ValueError("tokenize response contains multiple token sequences")
            tokens = tokens[0]
        return as_token_ids_request(tokens)


def _tokenize_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if not value:
        raise ValueError("engine URL must not be empty")
    if value.endswith("/tokenize"):
        return value
    return f"{value}/tokenize"


def build_tokenize_payload(
    request: Mapping[str, Any], *, model: str | None = None
) -> dict[str, Any]:
    """Translate a completion/chat request into a tokenize-endpoint request."""
    body = request_body(request)
    payload = {key: body[key] for key in _TOKENIZE_FIELDS if key in body}
    if model is not None:
        payload["model"] = model
    validate_prompt_kind(payload)
    return payload
