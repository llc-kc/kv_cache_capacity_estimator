# SPDX-License-Identifier: Apache-2.0
"""In-process tokenization using vLLM's rendering pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..models import TokenIdsRequest
from ..token_id_replay import as_token_ids_request
from .common import request_body, validate_prompt_kind


@dataclass(frozen=True)
class _VllmApi:
    ModelConfig: Any
    VllmConfig: Any
    ChatCompletionRequest: Any
    CompletionRequest: Any
    ErrorResponse: Any
    OnlineRenderer: Any
    extract_prompt_components: Any
    load_chat_template: Any
    renderer_from_config: Any


def _load_vllm_api() -> _VllmApi:
    try:
        from vllm.config import ModelConfig, VllmConfig
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
        from vllm.renderers import renderer_from_config
        from vllm.renderers.inputs.preprocess import extract_prompt_components
        from vllm.renderers.online_renderer import OnlineRenderer
    except ImportError as error:
        raise RuntimeError(
            "the vllm tokenize backend requires an installed vLLM package"
        ) from error

    return _VllmApi(
        ModelConfig=ModelConfig,
        VllmConfig=VllmConfig,
        ChatCompletionRequest=ChatCompletionRequest,
        CompletionRequest=CompletionRequest,
        ErrorResponse=ErrorResponse,
        OnlineRenderer=OnlineRenderer,
        extract_prompt_components=extract_prompt_components,
        load_chat_template=load_chat_template,
        renderer_from_config=renderer_from_config,
    )


class VllmTokenizer:
    """Run vLLM's OpenAI request renderer and tokenizer without an engine."""

    def __init__(
        self,
        model: str,
        *,
        tool_call_parser: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")

        api = _load_vllm_api()
        model_config = api.ModelConfig(
            model=model,
            tokenizer=model,
            trust_remote_code=trust_remote_code,
        )
        # Mirror vLLM's GPU-less render server: tokenization does not require
        # quantization kernels or model weights.
        model_config.quantization = None
        vllm_config = api.VllmConfig(model_config=model_config)
        renderer = api.renderer_from_config(vllm_config)
        online_renderer = api.OnlineRenderer(
            model_config=model_config,
            renderer=renderer,
            request_logger=None,
            chat_template=api.load_chat_template(None),
            chat_template_content_format="auto",
            enable_auto_tools=tool_call_parser is not None,
            tool_parser=tool_call_parser,
        )
        online_renderer.warmup()

        self._api = api
        self._model_config = model_config
        self._base_renderer = renderer
        self._renderer = online_renderer

    def close(self) -> None:
        self._base_renderer.shutdown()

    def tokenize(self, request: Mapping[str, Any]) -> TokenIdsRequest:
        body = dict(request_body(request))
        validate_prompt_kind(body)

        if body.get("messages") is not None:
            # Count only input tokens instead of reserving the trace request's
            # original generation budget.
            body["max_tokens"] = 0
            body["max_completion_tokens"] = 0
            parsed_request = self._api.ChatCompletionRequest.model_validate(body)
            result = _run(self._renderer.render_chat(parsed_request, skip_mm_cache=True))
            if isinstance(result, self._api.ErrorResponse):
                raise ValueError(result.error.message)
            _, engine_inputs = result
        else:
            body["max_tokens"] = 0
            parsed_request = self._api.CompletionRequest.model_validate(body)
            result = _run(
                self._renderer.render_completion(parsed_request, skip_mm_cache=True)
            )
            if isinstance(result, self._api.ErrorResponse):
                raise ValueError(result.error.message)
            engine_inputs = result

        if len(engine_inputs) != 1:
            raise ValueError("request contains multiple prompt sequences")
        components = self._api.extract_prompt_components(
            self._model_config, engine_inputs[0]
        )
        if components.token_ids is None:
            raise ValueError("vLLM renderer did not produce token IDs")
        return as_token_ids_request(components.token_ids)


def _run(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    close = getattr(awaitable, "close", None)
    if close is not None:
        close()
    raise RuntimeError("VllmTokenizer.tokenize cannot run inside an active event loop")
