# SPDX-License-Identifier: Apache-2.0
"""Heuristic checks that ``--tool-call-parser`` matches the served model family.

The tool-call parser selects the tool-call detector *and*, for some families,
the prompt encoder (for example SGLang renders DeepSeek-V4 / V4.1 requests with
a native recipe encoder keyed off ``deepseekv4`` / ``deepseekv41``). Passing the
parser of another family therefore tokenizes prompts with the wrong format and
silently skews every capacity number in the replay.

SGLang/vLLM cannot reject such combinations themselves (parser names are free
strings and generic parsers such as ``hermes`` are intentionally portable), so
this module only emits a warning when *both* sides are confidently identifiable
and disagree.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FamilyRule:
    parsers: frozenset[str]
    # Substrings matched (case-insensitively) against architecture names and
    # model_type values.
    config_hints: frozenset[str]
    # Whole letter-only tokens matched against the model path basename.
    path_tokens: frozenset[str]


# Parser names grouped by the model family they encode. Generic / portable
# parsers (hermes, pythonic, response_template, gpt-oss-style reuse, ...) are
# intentionally absent: naming another family's parser is the failure mode here.
_FAMILY_RULES: dict[str, _FamilyRule] = {
    "glm": _FamilyRule(
        parsers=frozenset({"glm", "glm45", "glm47", "flint"}),
        config_hints=frozenset({"glm", "chatglm"}),
        path_tokens=frozenset({"glm", "chatglm"}),
    ),
    "deepseek": _FamilyRule(
        parsers=frozenset(
            {"deepseekv3", "deepseekv31", "deepseekv32", "deepseekv4", "deepseekv41"}
        ),
        config_hints=frozenset({"deepseek"}),
        path_tokens=frozenset({"deepseek", "dsv"}),
    ),
    "qwen": _FamilyRule(
        parsers=frozenset({"qwen", "qwen25", "qwen3_coder", "qwen3coder"}),
        config_hints=frozenset({"qwen"}),
        path_tokens=frozenset({"qwen"}),
    ),
    "kimi": _FamilyRule(
        parsers=frozenset({"kimi_k2", "kimi_k3", "kimik2", "kimik3"}),
        config_hints=frozenset({"kimi", "moonshot"}),
        path_tokens=frozenset({"kimi", "moonshot"}),
    ),
    "llama": _FamilyRule(
        parsers=frozenset({"llama3"}),
        config_hints=frozenset({"llama"}),
        path_tokens=frozenset({"llama"}),
    ),
    "mistral": _FamilyRule(
        parsers=frozenset({"mistral"}),
        config_hints=frozenset({"mistral", "mixtral"}),
        path_tokens=frozenset({"mistral", "mixtral"}),
    ),
    "minimax": _FamilyRule(
        parsers=frozenset({"minimax-m2", "minimax-m3", "minimaxm2", "minimaxm3"}),
        config_hints=frozenset({"minimax"}),
        path_tokens=frozenset({"minimax"}),
    ),
    "gemma": _FamilyRule(
        parsers=frozenset({"gemma4"}),
        config_hints=frozenset({"gemma"}),
        path_tokens=frozenset({"gemma"}),
    ),
    "cohere": _FamilyRule(
        parsers=frozenset({"cohere_command4", "coherecommand4"}),
        config_hints=frozenset({"cohere", "command"}),
        path_tokens=frozenset({"cohere", "command"}),
    ),
    "gigachat": _FamilyRule(
        parsers=frozenset({"gigachat3", "gigachat35"}),
        config_hints=frozenset({"gigachat"}),
        path_tokens=frozenset({"gigachat"}),
    ),
    "hunyuan": _FamilyRule(
        parsers=frozenset({"hunyuan"}),
        config_hints=frozenset({"hunyuan"}),
        path_tokens=frozenset({"hunyuan"}),
    ),
    "internlm": _FamilyRule(
        parsers=frozenset({"interns1"}),
        config_hints=frozenset({"internlm", "interns"}),
        path_tokens=frozenset({"internlm", "interns"}),
    ),
    "step": _FamilyRule(
        parsers=frozenset({"step3"}),
        config_hints=frozenset({"step"}),
        path_tokens=frozenset({"step"}),
    ),
    "spark": _FamilyRule(
        parsers=frozenset({"spark25"}),
        config_hints=frozenset({"spark"}),
        path_tokens=frozenset({"spark"}),
    ),
}

_PARSER_TO_FAMILY = {
    parser: family
    for family, rule in _FAMILY_RULES.items()
    for parser in rule.parsers
}

_TOKEN_RUN = re.compile(r"[a-z]+")


def _normalize_parser_name(tool_call_parser: str) -> str:
    """Canonicalize across SGLang/vLLM spelling variants.

    ``deepseek_v41`` / ``deepseek-v41`` / ``deepseekv41`` all name the same
    parser across engine versions, so underscores and hyphens are dropped.
    """
    return tool_call_parser.strip().lower().replace("_", "").replace("-", "")


def _normalize_config_hint(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "")


def _iter_config_strings(hf_config: Any) -> Iterable[str]:
    """Architecture and model_type strings exposed by an HF config object."""
    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, str):
        yield architectures
    elif isinstance(architectures, (list, tuple)):
        for architecture in architectures:
            if isinstance(architecture, str):
                yield architecture
    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str):
        yield model_type
    # Multimodal configs (e.g. DeepSeek-V4.1) nest the text model config.
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and text_config is not hf_config:
        nested_type = getattr(text_config, "model_type", None)
        if isinstance(nested_type, str):
            yield nested_type


def _family_from_config(hf_config: Any) -> Optional[tuple[str, str]]:
    strings = [_normalize_config_hint(value) for value in _iter_config_strings(hf_config)]
    for family, rule in _FAMILY_RULES.items():
        for hint in rule.config_hints:
            normalized_hint = _normalize_config_hint(hint)
            for value in strings:
                if normalized_hint in value:
                    return family, f"config({value})"
    return None


def _family_from_path(model: str) -> Optional[tuple[str, str]]:
    basename = model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    tokens = set(_TOKEN_RUN.findall(basename.lower()))
    for family, rule in _FAMILY_RULES.items():
        if tokens & set(rule.path_tokens):
            return family, f"path({model})"
    return None


def detect_tool_parser_mismatch(
    model: str | None,
    tool_call_parser: str | None,
    *,
    hf_config: Any = None,
) -> Optional[str]:
    """Return a warning message when the parser looks wrong for the model.

    Returns ``None`` when the combination is consistent or cannot be judged
    (parser unset, parser generic/unknown, model family unknown).
    """
    if not tool_call_parser:
        return None
    parser_family = _PARSER_TO_FAMILY.get(_normalize_parser_name(tool_call_parser))
    if parser_family is None:
        # Generic or unknown parser: do not second-guess portable choices.
        return None

    identity: Optional[tuple[str, str]] = None
    if hf_config is not None:
        identity = _family_from_config(hf_config)
    if identity is None and model:
        identity = _family_from_path(model)
    if identity is None:
        return None

    model_family, evidence = identity
    if model_family == parser_family:
        return None

    expected = ", ".join(sorted(_FAMILY_RULES[model_family].parsers))
    return (
        f"tool-call parser {tool_call_parser!r} does not match the {model_family} "
        f"model {model!r} (family inferred from {evidence}); this parser belongs "
        f"to the {parser_family} family. The parser for this model is usually one "
        f"of: {expected}. A mismatched parser renders tool calls (and possibly the "
        "whole prompt) with another model's format, which skews token counts; pass "
        "the parser the serving engine actually uses for this model, or omit "
        "--tool-call-parser when the trace has no tool calls."
    )


def log_tool_parser_mismatch(
    model: str | None,
    tool_call_parser: str | None,
    *,
    hf_config: Any = None,
) -> Optional[str]:
    """Log the mismatch warning, if any, and return the message."""
    message = detect_tool_parser_mismatch(model, tool_call_parser, hf_config=hf_config)
    if message is not None:
        logger.warning(message)
    return message
