# SPDX-License-Identifier: Apache-2.0
"""In-process tokenization using SGLang's OpenAI request pipeline."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Union

from ..models import TokenIdsRequest
from ..token_id_replay import as_token_ids_request
from .common import request_body, validate_prompt_kind


logger = logging.getLogger(__name__)

# Tool-call parsers whose models SGLang renders with a native encoder instead
# of an HF Jinja chat template (see sglang's ``resolve_chat_encoding_spec``).
_RECIPE_PARSER_SPECS = {
    "deepseekv41": "dsv41",
    "deepseekv4": "dsv4",
}


@dataclass(frozen=True)
class _SglangApi:
    TokenizeRequest: Any
    FunctionCallParser: Any
    PartialJsonAllow: Any
    detect_jinja_template_content_format: Any
    envs: Any
    get_config: Any
    get_tokenizer: Any
    normalize_assistant_tool_call_arguments: Any
    normalize_tool_content: Any
    partial_json_loads: Any
    process_content_for_template_format: Any
    resolve_chat_encoding_spec: Any


def _load_sglang_api() -> _SglangApi:
    try:
        from partial_json_parser.core.options import Allow as PartialJsonAllow
        from sglang.srt.entrypoints.openai.chat_encoding import (
            resolve_chat_encoding_spec,
        )
        from sglang.srt.entrypoints.openai.protocol import TokenizeRequest
        from sglang.srt.entrypoints.openai.serving_chat import (
            normalize_assistant_tool_call_arguments,
            normalize_tool_content,
        )
        from sglang.srt.environ import envs
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.function_call.utils import (
            _partial_json_loads as partial_json_loads,
        )
        from sglang.srt.parser.jinja_template_utils import (
            detect_jinja_template_content_format,
            process_content_for_template_format,
        )
        from sglang.srt.utils.hf_transformers import config as hf_config_module
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
    except ImportError as error:
        raise RuntimeError(
            "cannot import the SGLang tokenize backend; install SGLang and its "
            f"runtime dependencies ({error})"
        ) from error

    return _SglangApi(
        TokenizeRequest=TokenizeRequest,
        FunctionCallParser=FunctionCallParser,
        PartialJsonAllow=PartialJsonAllow,
        detect_jinja_template_content_format=detect_jinja_template_content_format,
        envs=envs,
        get_config=hf_config_module.get_config,
        get_tokenizer=get_tokenizer,
        normalize_assistant_tool_call_arguments=normalize_assistant_tool_call_arguments,
        normalize_tool_content=normalize_tool_content,
        partial_json_loads=partial_json_loads,
        process_content_for_template_format=process_content_for_template_format,
        resolve_chat_encoding_spec=resolve_chat_encoding_spec,
    )


def _recover_tool_call_arguments(
    message: dict[str, Any], api: _SglangApi, tool_names: set[str]
) -> int:
    """Recover malformed history calls using SGLang's parser invariants.

    SGLang's GLM-4.7 detector emits one ``ToolCallItem`` per native
    ``<tool_call>`` block and every item's arguments are a JSON object. Some
    captured traces collapse adjacent calls into one OpenAI tool call by
    concatenating their names and storing their argument objects in an array.
    Split that representation back into the calls the detector would emit.
    """
    if message.get("role") != "assistant":
        return 0
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return 0

    recovered = 0
    normalized_tool_calls = []
    for tool_call in tool_calls:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict):
            normalized_tool_calls.append(tool_call)
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            normalized_tool_calls.append(tool_call)
            continue
        try:
            parsed, consumed = api.partial_json_loads(
                arguments, api.PartialJsonAllow.ALL
            )
        except Exception:
            normalized_tool_calls.append(tool_call)
            continue
        if consumed != len(arguments):
            normalized_tool_calls.append(tool_call)
            continue

        if isinstance(parsed, dict):
            function["arguments"] = parsed
            normalized_tool_calls.append(tool_call)
            recovered += 1
            continue

        if not parsed or not isinstance(parsed, list) or not all(
            isinstance(item, dict) for item in parsed
        ):
            normalized_tool_calls.append(tool_call)
            continue

        function_name = function.get("name")
        split_names = _split_concatenated_tool_name(
            function_name, len(parsed), tool_names
        )
        if split_names is None:
            normalized_tool_calls.append(tool_call)
            continue

        for index, (name, item_arguments) in enumerate(zip(split_names, parsed)):
            recovered_call = dict(tool_call)
            recovered_function = dict(function)
            recovered_function["name"] = name
            recovered_function["arguments"] = item_arguments
            recovered_call["function"] = recovered_function
            if index:
                call_id = recovered_call.get("id")
                if isinstance(call_id, str):
                    recovered_call["id"] = f"{call_id}_{index + 1}"
            normalized_tool_calls.append(recovered_call)
        recovered += len(parsed)

    message["tool_calls"] = normalized_tool_calls
    return recovered


def _split_concatenated_tool_name(
    name: Any, call_count: int, tool_names: set[str]
) -> list[str] | None:
    """Return the unique split of a collapsed tool name, if one exists."""
    if not isinstance(name, str) or call_count < 1:
        return None

    candidates = {candidate for candidate in tool_names if candidate}
    if call_count == 1:
        return [name]

    # The common corruption is ``foofoo`` paired with ``[{...}, {...}]``.
    if len(name) % call_count == 0:
        part_length = len(name) // call_count
        part = name[:part_length]
        if part * call_count == name and (not candidates or part in candidates):
            return [part] * call_count

    if not candidates:
        return None

    solutions: list[list[str]] = []

    def search(offset: int, parts: list[str]) -> None:
        if len(solutions) > 1:
            return
        if len(parts) == call_count:
            if offset == len(name):
                solutions.append(parts.copy())
            return
        for candidate in candidates:
            if name.startswith(candidate, offset):
                search(offset + len(candidate), [*parts, candidate])

    search(0, [])
    return solutions[0] if len(solutions) == 1 else None


class SglangTokenizer:
    """Run SGLang's tokenizer and chat-template helpers without an engine."""

    def __init__(
        self,
        model: str,
        *,
        tool_call_parser: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")

        api = _load_sglang_api()
        tokenizer = api.get_tokenizer(
            model,
            tokenizer_mode="auto",
            trust_remote_code=trust_remote_code,
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        if isinstance(chat_template, dict):
            chat_template = next(iter(chat_template.values()), None)
            tokenizer.chat_template = chat_template
        if (
            tool_call_parser is not None
            and tool_call_parser not in api.FunctionCallParser.ToolCallParserEnum
        ):
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        # Some models (e.g. DeepSeek-V4 / V4.1) ship no HF Jinja chat template;
        # SGLang renders their prompts through a native encoder instead.
        chat_encoding_spec = _RECIPE_PARSER_SPECS.get(tool_call_parser)
        if chat_encoding_spec is None:
            hf_config = None
            try:
                hf_config = api.get_config(
                    model, trust_remote_code=trust_remote_code
                )
            except Exception as error:  # config is only needed for spec detection
                logger.warning(
                    "could not load model config for chat-encoding detection: %s",
                    error,
                )
            if hf_config is not None:
                try:
                    chat_encoding_spec = api.resolve_chat_encoding_spec(
                        hf_config=hf_config,
                        tokenizer=tokenizer,
                        tool_call_parser=tool_call_parser,
                    )
                except Exception as error:
                    logger.warning(
                        "chat-encoding spec detection failed: %s", error
                    )

        self._api = api
        self._tokenizer = tokenizer
        self._tool_call_parser = tool_call_parser
        self._chat_encoding_spec = chat_encoding_spec
        self._recipe = None
        self._default_effort = None
        if chat_encoding_spec in ("dsv4", "dsv41"):
            self._recipe = self._load_deepseek_recipe(chat_encoding_spec, tokenizer)
            effort_env = (
                api.envs.SGLANG_DSV41_REASONING_EFFORT
                if chat_encoding_spec == "dsv41"
                else api.envs.SGLANG_DSV4_REASONING_EFFORT
            )
            raw_effort = (effort_env.get() or "").strip()
            self._default_effort = (
                _numeric_effort(raw_effort) if raw_effort else None
            )
        self._content_format = (
            api.detect_jinja_template_content_format(chat_template)
            if chat_template
            else None
        )
        try:
            self._tokenizer_auto_adds_specials = len(tokenizer.encode("")) > 0
        except Exception:
            self._tokenizer_auto_adds_specials = True

    @staticmethod
    def _load_deepseek_recipe(spec: str, tokenizer: Any) -> Any:
        """DeepSeek-V4 / V4.1 prompt encoder (deepseek-recipe), as the served
        chat path builds it."""
        try:
            from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe
        except ImportError as error:
            raise RuntimeError(
                "tokenizing DeepSeek-V4 / V4.1 requests requires a SGLang build "
                f"with the deepseek prompt encoder ({error})"
            ) from error
        return DeepSeekRecipe.from_hf_tokenizer(spec, tokenizer)

    def tokenize(self, request: Mapping[str, Any]) -> TokenIdsRequest:
        body = dict(request_body(request))
        validate_prompt_kind(body)
        parsed_request = self._api.TokenizeRequest.model_validate(body)

        if parsed_request.messages is not None:
            if self._recipe is not None:
                token_ids = self._tokenize_recipe_chat(parsed_request)
            else:
                token_ids = self._tokenize_chat(parsed_request)
        else:
            prompts = (
                parsed_request.prompt
                if isinstance(parsed_request.prompt, list)
                else [parsed_request.prompt]
            )
            if len(prompts) != 1:
                raise ValueError("request contains multiple prompt sequences")
            token_ids = self._tokenizer.encode(
                prompts[0], add_special_tokens=parsed_request.add_special_tokens
            )
        return as_token_ids_request(token_ids)

    def _tokenize_recipe_chat(self, request: Any) -> list[int]:
        """Render a chat request with the DeepSeek recipe encoder, mirroring
        SGLang's ``DeepSeekRecipeChatEncodingAdapter.encode_messages``."""
        chat_request = request.to_chat_completion_request()
        # The served adapter ignores the caller-normalized messages and rebuilds
        # them from the request dump; history tool-call arguments stay on the
        # wire as strings and recipe serializes object arguments itself.
        wire = chat_request.model_dump(
            exclude_unset=True, by_alias=True, mode="json"
        )
        wire_messages = _null_content_as_empty(wire["messages"])
        continue_final = chat_request.continue_final_message
        task = getattr(chat_request, "task", None)
        task_prefix = None
        trailing = _trailing_assistant_turn(wire_messages)
        if trailing is not None:
            if not continue_final:
                wire_messages[-1] = {"role": "user", "content": trailing["content"]}
            elif task is not None:
                # A task renders its own tail, so the prefix follows it.
                task_prefix = wire_messages.pop()["content"]
                continue_final = False
            else:
                wire_messages[-1] = trailing

        thinking = (chat_request.chat_template_kwargs or {}).get(
            "thinking", self._api.envs.SGLANG_DEFAULT_THINKING.get()
        )
        body: dict[str, Any] = {
            "messages": wire_messages,
            "thinking": bool(thinking),
            "continue_final_message": continue_final,
            "response_format": wire.get("response_format"),
        }
        tools = self._select_tools(chat_request)
        if tools is not None:
            body["tools"] = [
                {k: v for k, v in tool.items() if v is not None}
                | {
                    "function": {
                        k: v
                        for k, v in tool["function"].items()
                        if v is not None
                    }
                }
                for tool in tools
            ]
            body["tool_choice"] = wire.get("tool_choice")
        body = {key: value for key, value in body.items() if value is not None}

        effort = _numeric_effort(chat_request.reasoning_effort)
        if effort is None or effort == "none":
            effort = self._default_effort
        budget = None
        if effort is not None and not isinstance(effort, str):
            if self._chat_encoding_spec != "dsv41":
                raise ValueError(
                    "DeepSeek-V4 reasoning_effort must be a named tier, got "
                    f"{effort!r}"
                )
            from sglang.srt.parser.deepseek_recipe_renderer import parse_effort_budget

            budget = parse_effort_budget(effort)
        elif effort is not None:
            body["reasoning_effort"] = effort

        prompt_ids = self._recipe.encode(
            body, task=task, effort_budget=budget
        )
        if task_prefix:
            prompt_ids += self._recipe.encode_text(task_prefix)
        return prompt_ids

    def _tokenize_chat(self, request: Any) -> list[int]:
        if self._content_format is None:
            raise ValueError("the model tokenizer does not define a chat template")
        chat_request = request.to_chat_completion_request()
        messages = [message.model_dump() for message in chat_request.messages]
        tool_names = {
            tool.function.name
            for tool in (chat_request.tools or [])
            if getattr(getattr(tool, "function", None), "name", None)
        }
        recovered_tool_calls = 0
        for message in messages:
            strict = self._tool_call_parser != "kimi_k3"
            try:
                self._api.normalize_assistant_tool_call_arguments(
                    message, strict=strict
                )
            except ValueError:
                recovered = _recover_tool_call_arguments(
                    message, self._api, tool_names
                )
                if not recovered:
                    raise
                recovered_tool_calls += recovered
                self._api.normalize_assistant_tool_call_arguments(
                    message, strict=strict
                )
        if recovered_tool_calls:
            logger.warning(
                "recovered %d malformed assistant tool call(s) using "
                "SGLang-compatible parsing",
                recovered_tool_calls,
            )

        tools = self._select_tools(chat_request)
        normalized_messages = []
        for message in messages:
            if message.get("content") is None:
                message["content"] = ""
            message = self._api.process_content_for_template_format(
                message,
                self._content_format,
                [],
                [],
                [],
                [],
            )
            message["content"] = self._api.normalize_tool_content(
                message["role"], message.get("content")
            )
            normalized_messages.append(message)

        normalized_messages, assistant_prefix = self._handle_final_assistant(
            normalized_messages, chat_request.continue_final_message
        )
        template_kwargs: dict[str, Any] = {}
        if chat_request.reasoning_effort is not None:
            template_kwargs["reasoning_effort"] = chat_request.reasoning_effort
        if chat_request.chat_template_kwargs:
            template_kwargs.update(chat_request.chat_template_kwargs)

        try:
            rendered_prompt = self._tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
                return_dict=False,
                **template_kwargs,
            )
        except (TypeError, ValueError):
            flat_tools = (
                [tool["function"] if "function" in tool else tool for tool in tools]
                if tools
                else None
            )
            rendered_prompt = self._tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=flat_tools,
                return_dict=False,
                **template_kwargs,
            )

        encode_kwargs = (
            {"add_special_tokens": False}
            if self._tokenizer_auto_adds_specials
            else {}
        )
        prompt_ids = self._tokenizer.encode(rendered_prompt, **encode_kwargs)
        if assistant_prefix is not None:
            prefix_ids = self._tokenizer.encode(assistant_prefix)
            bos_token_id = getattr(self._tokenizer, "bos_token_id", None)
            if prefix_ids and prefix_ids[0] == bos_token_id:
                prefix_ids = prefix_ids[1:]
            prompt_ids += prefix_ids
        return prompt_ids

    @staticmethod
    def _select_tools(request: Any) -> list[dict[str, Any]] | None:
        if not request.tools or request.tool_choice == "none":
            return None
        if isinstance(request.tool_choice, str):
            selected = request.tools
        else:
            name = request.tool_choice.function.name
            selected = [tool for tool in request.tools if tool.function.name == name]
        return [tool.model_dump() for tool in selected] or None

    @staticmethod
    def _handle_final_assistant(
        messages: list[dict[str, Any]], continue_final_message: bool
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not messages or messages[-1].get("role") != "assistant":
            return messages, None
        content = messages[-1].get("content")
        if not isinstance(content, str):
            return messages, None
        if continue_final_message:
            return messages[:-1], content
        messages[-1] = {"role": "user", "content": content}
        return messages, None


def _null_content_as_empty(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """As SGLang renders them: a null message content becomes an empty string."""
    return [
        message if message.get("content") is not None else {**message, "content": ""}
        for message in messages
    ]


def _trailing_assistant_turn(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """The trailing text turn SGLang continues or, without
    continue_final_message, re-sends as a user turn, without the tool calls
    SGLang discards (mirrors the DeepSeek recipe serving adapter)."""
    last = messages[-1] if messages else None
    if last and last["role"] == "assistant" and isinstance(last.get("content"), str):
        return {key: value for key, value in last.items() if key != "tool_calls"}
    return None


def _numeric_effort(effort: Union[str, int, float, None]) -> Union[str, int, float, None]:
    """A numeric string effort ("30", "0.3") as the number it spells."""
    if isinstance(effort, str) and re.fullmatch(r"\d+(\.\d+)?", effort.strip()):
        return float(effort) if "." in effort else int(effort)
    return effort
