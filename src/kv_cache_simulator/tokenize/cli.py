# SPDX-License-Identifier: Apache-2.0
"""Shared command-line setup for OpenAI request tokenizers."""

from __future__ import annotations

import argparse
import os

from .common import Tokenizer
from .engine import EngineTokenizer
from .sglang import SglangTokenizer
from .vllm import VllmTokenizer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_tokenizer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the tokenizer options shared by request-oriented CLIs."""
    parser.add_argument(
        "--tokenize-backend",
        choices=("engine", "sglang", "vllm"),
        default="engine",
        help="use a deployed endpoint, or import SGLang/vLLM in this process",
    )
    parser.add_argument(
        "--engine-url",
        help="engine base URL, /v1 URL, or full /tokenize URL",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--model",
        help="engine model override, or local model/config/tokenizer path",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--tool-call-parser",
        help="SGLang/vLLM tool call parser, for example glm47",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--tokenize-workers",
        type=_positive_int,
        default=4,
        metavar="N",
        help="number of concurrent tokenization workers (default: 4)",
    )


def build_tokenizer(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Tokenizer:
    """Build the tokenizer selected by parsed command-line arguments."""
    if args.tokenize_backend == "engine":
        if not args.engine_url:
            parser.error("--engine-url is required for --tokenize-backend engine")
        return EngineTokenizer(
            args.engine_url,
            api_key=args.api_key,
            timeout=args.timeout,
            model=args.model,
        )
    if args.tokenize_backend == "vllm":
        if not args.model:
            parser.error("--model is required for --tokenize-backend vllm")
        return VllmTokenizer(
            args.model,
            tool_call_parser=args.tool_call_parser,
            trust_remote_code=args.trust_remote_code,
        )
    if not args.model:
        parser.error("--model is required for --tokenize-backend sglang")
    return SglangTokenizer(
        args.model,
        tool_call_parser=args.tool_call_parser,
        trust_remote_code=args.trust_remote_code,
    )


def close_tokenizer(tokenizer: Tokenizer) -> None:
    """Release resources held by tokenizers that require explicit shutdown."""
    if isinstance(tokenizer, VllmTokenizer):
        tokenizer.close()
