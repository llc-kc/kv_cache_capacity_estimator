# SPDX-License-Identifier: Apache-2.0

import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kv_cache_simulator.tokenize import (
    EngineTokenizer,
    SglangTokenizer,
    VllmTokenizer,
    build_tokenize_payload,
    tokenize_openai_trace,
)
from kv_cache_simulator.tokenize.vllm import _VllmApi
from kv_cache_simulator.tokenize.sglang import _SglangApi
from kv_cache_simulator.models import TokenIdsRequest
from kv_cache_simulator.simulator import SimulationConfig, simulate
from kv_cache_simulator.token_id_replay import parse_jsonl
from kv_cache_simulator.cli_args import log_request_details
from request_replay import build_parser as build_request_parser
from openai_to_token_ids import build_parser as build_convert_parser
from openai_to_token_ids import main as convert_main


class FakeTokenizer:
    def __init__(self):
        self.requests = []

    def tokenize(self, request):
        self.requests.append(request)
        return TokenIdsRequest((1, 2, 3, 4))


class FakeVllmModelConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.quantization = "fp8"


class FakeVllmConfig:
    def __init__(self, *, model_config):
        self.model_config = model_config

    def shutdown(self):
        self.was_shutdown = True


class FakeVllmRequest:
    last_body = None

    @classmethod
    def model_validate(cls, body):
        cls.last_body = body
        return SimpleNamespace(**body)


class FakeVllmErrorResponse:
    pass


class FakeOnlineRenderer:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.warmed_up = False

    def warmup(self):
        self.warmed_up = True

    async def render_chat(self, request, *, skip_mm_cache):
        self.chat_request = request
        self.skip_mm_cache = skip_mm_cache
        return [], [{"prompt_token_ids": [11, 12, 13]}]

    async def render_completion(self, request, *, skip_mm_cache):
        self.completion_request = request
        self.skip_mm_cache = skip_mm_cache
        return [{"prompt_token_ids": [21, 22]}]


def fake_vllm_api():
    return _VllmApi(
        ModelConfig=FakeVllmModelConfig,
        VllmConfig=FakeVllmConfig,
        ChatCompletionRequest=FakeVllmRequest,
        CompletionRequest=FakeVllmRequest,
        ErrorResponse=FakeVllmErrorResponse,
        OnlineRenderer=FakeOnlineRenderer,
        extract_prompt_components=lambda _config, value: SimpleNamespace(
            token_ids=value["prompt_token_ids"]
        ),
        load_chat_template=lambda value: value,
        renderer_from_config=lambda config: config,
    )


class FakeSglangMessage:
    def __init__(self, value):
        self.value = value

    def model_dump(self):
        return json.loads(json.dumps(self.value))


class FakeSglangTool:
    def __init__(self, value):
        self.value = value
        self.function = SimpleNamespace(name=value["function"]["name"])

    def model_dump(self):
        return json.loads(json.dumps(self.value))


class FakeSglangRequest:
    @classmethod
    def model_validate(cls, body):
        value = cls()
        value.messages = (
            [FakeSglangMessage(message) for message in body["messages"]]
            if body.get("messages") is not None
            else None
        )
        value.prompt = body.get("prompt")
        value.add_special_tokens = body.get("add_special_tokens", True)
        value.tools = [FakeSglangTool(tool) for tool in body.get("tools", [])]
        value.tool_choice = body.get("tool_choice") or (
            "auto" if value.tools else "none"
        )
        value.reasoning_effort = body.get("reasoning_effort")
        value.continue_final_message = body.get("continue_final_message", False)
        value.chat_template_kwargs = body.get("chat_template_kwargs")
        return value

    def to_chat_completion_request(self):
        return self


class FakeSglangFunctionCallParser:
    ToolCallParserEnum = {"glm47": object}
    constructor_calls = 0

    def __init__(self, tools, parser, tokenizer=None):
        type(self).constructor_calls += 1


class FakeSglangTokenizerImpl:
    chat_template = "{{ messages }} image"
    bos_token_id = 1

    def __init__(self):
        self.template_call = None
        self.encode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return "rendered"

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        if text == "":
            return [1]
        if text == "rendered":
            return [31, 32]
        if text == "partial":
            return [1, 40]
        return [11]


def fake_sglang_api():
    FakeSglangFunctionCallParser.constructor_calls = 0
    tokenizer = FakeSglangTokenizerImpl()

    def normalize_arguments(message, *, strict):
        for tool_call in message.get("tool_calls", []):
            arguments = tool_call["function"].get("arguments")
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    if strict:
                        raise ValueError(
                            "Assistant tool call function.arguments must be a "
                            "JSON object."
                        )
                    continue
                tool_call["function"]["arguments"] = parsed

    def process_content(message, content_format, *_buffers):
        if content_format == "openai":
            return message
        return message

    partial_json_allow = SimpleNamespace(ALL=object())
    return _SglangApi(
        TokenizeRequest=FakeSglangRequest,
        FunctionCallParser=FakeSglangFunctionCallParser,
        PartialJsonAllow=partial_json_allow,
        detect_jinja_template_content_format=lambda _template: "openai",
        get_tokenizer=lambda *_args, **_kwargs: tokenizer,
        normalize_assistant_tool_call_arguments=normalize_arguments,
        normalize_tool_content=lambda _role, content: content,
        partial_json_loads=lambda value, _allow: (json.loads(value), len(value)),
        process_content_for_template_format=process_content,
    )


class ReplaySourcesTest(unittest.TestCase):
    def test_token_id_replay_runs_without_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "tokens.jsonl"
            trace.write_text("[1, 2, 3, 4]\n[1, 2, 5, 6]\n", encoding="utf-8")
            result = simulate(
                parse_jsonl(trace),
                SimulationConfig(
                    page_size=2,
                    kv_bytes_per_token=4,
                    capacities=(("16B", 16),),
                ),
            )
        self.assertEqual(result.requests, 2)
        self.assertEqual(result.page_accesses, 4)
        self.assertEqual(result.results[0].page_hits, 1)
        self.assertGreaterEqual(result.timings.input_processing_seconds, 0)
        self.assertGreaterEqual(result.timings.page_build_seconds, 0)
        self.assertGreaterEqual(result.timings.reuse_distance_seconds, 0)
        self.assertGreaterEqual(result.timings.fixed_capacity_seconds, 0)
        self.assertGreaterEqual(result.timings.capacity_requirement_seconds, 0)
        self.assertGreaterEqual(
            result.timings.total_seconds,
            result.timings.input_processing_seconds,
        )

    def test_invalid_token_id_has_line_context(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "tokens.jsonl"
            trace.write_text("[1]\n[-1]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                list(parse_jsonl(trace))

    def test_multimodal_messages_are_passed_through(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]
        payload = build_tokenize_payload(
            {
                "model": "vision-model",
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 100,
            }
        )
        self.assertIs(payload["messages"], messages)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)

    def test_openai_batch_row_body_is_supported(self):
        payload = build_tokenize_payload(
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            },
            model="override",
        )
        self.assertEqual(payload["model"], "override")
        self.assertEqual(payload["messages"][0]["content"], "hi")

    def test_openai_trace_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            trace.write_text(
                '{"messages": [{"role": "user", "content": "first"}]}\n'
                '{"prompt": "second"}\n',
                encoding="utf-8",
            )
            tokenizer = FakeTokenizer()
            tokens = list(tokenize_openai_trace(trace, tokenizer))
        self.assertEqual(tokens, [TokenIdsRequest((1, 2, 3, 4))] * 2)
        self.assertEqual(tokenizer.requests[0]["messages"][0]["content"], "first")
        self.assertEqual(tokenizer.requests[1]["prompt"], "second")

    def test_openai_trace_parallel_tokenization_preserves_order(self):
        class DelayedTokenizer:
            def __init__(self):
                self.completion_order = []

            def tokenize(self, request):
                sequence = request["sequence"]
                time.sleep((4 - sequence) * 0.02)
                self.completion_order.append(sequence)
                return TokenIdsRequest((sequence,))

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            trace.write_text(
                "".join(
                    json.dumps({"prompt": str(sequence), "sequence": sequence})
                    + "\n"
                    for sequence in range(1, 5)
                ),
                encoding="utf-8",
            )
            tokenizer = DelayedTokenizer()
            tokens = list(tokenize_openai_trace(trace, tokenizer, workers=4))

        self.assertEqual(
            [request.input_ids for request in tokens],
            [(1,), (2,), (3,), (4,)],
        )
        self.assertNotEqual(tokenizer.completion_order, [1, 2, 3, 4])

    def test_openai_trace_parallel_error_is_logged_and_skipped(self):
        class FailingTokenizer:
            def tokenize(self, request):
                if request["sequence"] == 2:
                    raise ValueError("cannot tokenize request")
                return TokenIdsRequest((request["sequence"],))

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            trace.write_text(
                "".join(
                    json.dumps({"prompt": str(sequence), "sequence": sequence})
                    + "\n"
                    for sequence in range(1, 6)
                ),
                encoding="utf-8",
            )
            with self.assertLogs(
                "kv_cache_simulator.tokenize.common", level="INFO"
            ) as captured:
                tokens = list(
                    tokenize_openai_trace(trace, FailingTokenizer(), workers=2)
                )

        self.assertEqual(
            [request.input_ids for request in tokens],
            [(1,), (3,), (4,), (5,)],
        )
        self.assertIn("line 2: cannot tokenize request", captured.output[0])
        self.assertIn("succeeded=4, failed=1", captured.output[-1])

    def test_openai_trace_serial_error_is_logged_and_skipped(self):
        class FailingTokenizer:
            def tokenize(self, request):
                if request["sequence"] == 2:
                    raise RuntimeError("tokenizer unavailable")
                return TokenIdsRequest((request["sequence"],))

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            trace.write_text(
                '{"prompt": "first", "sequence": 1}\n'
                '{"prompt": "second", "sequence": 2}\n'
                '{"prompt": "third", "sequence": 3}\n',
                encoding="utf-8",
            )
            with self.assertLogs(
                "kv_cache_simulator.tokenize.common", level="INFO"
            ) as captured:
                tokens = list(
                    tokenize_openai_trace(trace, FailingTokenizer(), workers=1)
                )

        self.assertEqual(
            [request.input_ids for request in tokens],
            [(1,), (3,)],
        )
        self.assertIn("line 2: tokenizer unavailable", captured.output[0])
        self.assertIn("succeeded=2, failed=1", captured.output[-1])

    def test_openai_trace_rejects_non_positive_worker_count(self):
        with self.assertRaisesRegex(ValueError, "workers must be positive"):
            list(tokenize_openai_trace(Path("unused.jsonl"), FakeTokenizer(), workers=0))

    def test_request_replay_defaults_to_four_tokenize_workers(self):
        parser = build_request_parser()
        args = parser.parse_args(
            [
                "requests.jsonl",
                "--kv-bytes-per-token",
                "1",
                "--capacities",
                "1GiB",
            ]
        )
        self.assertEqual(args.tokenize_workers, 4)
        self.assertEqual(args.capacities, "1GiB")
        self.assertEqual(args.target_hit_rate_percent, 0.99)
        self.assertEqual(args.warm_up_percent, 0.5)
        self.assertFalse(args.log_token_ids)

        args = parser.parse_args(
            [
                "requests.jsonl",
                "--kv-bytes-per-token",
                "1",
            ]
        )
        self.assertEqual(
            args.capacities,
            "100GiB,200GiB,400GiB,800GiB,1TiB,2TiB,4TiB,6TiB,8TiB,"
            "12TiB,16TiB,24TiB,32TiB,64TiB",
        )

        args = parser.parse_args(
            [
                "requests.jsonl",
                "--kv-bytes-per-token",
                "1",
                "--capacities",
                "1GiB",
                "--tokenize-workers",
                "8",
            ]
        )
        self.assertEqual(args.tokenize_workers, 8)

        args = parser.parse_args(
            [
                "requests.jsonl",
                "--kv-bytes-per-token",
                "1",
                "--capacities",
                "1GiB",
                "--target-hit-rate-percent",
                "97.5%",
                "--warm-up-percent",
                "25",
                "--log-token-ids",
            ]
        )
        self.assertEqual(args.target_hit_rate_percent, 0.975)
        self.assertEqual(args.warm_up_percent, 0.25)
        self.assertTrue(args.log_token_ids)

    def test_openai_to_token_ids_uses_existing_trace_tokenizer(self):
        parser = build_convert_parser()
        args = parser.parse_args(["requests.jsonl", "--engine-url", "http://engine"])
        self.assertEqual(args.tokenize_workers, 4)
        self.assertIsNone(args.output)

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            output = Path(directory) / "tokens.jsonl"
            trace.write_text(
                '{"prompt": "first"}\n{"prompt": "second"}\n',
                encoding="utf-8",
            )
            with patch("openai_to_token_ids.build_tokenizer", return_value=FakeTokenizer()):
                result = convert_main(
                    [
                        str(trace),
                        "--engine-url",
                        "http://engine",
                        "--output",
                        str(output),
                        "--progress-interval",
                        "0",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "[1,2,3,4]\n[1,2,3,4]\n")

    def test_request_details_logs_length_at_debug_and_ids_at_info(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            trace.write_text('{"prompt": "hello"}\n', encoding="utf-8")
            with self.assertLogs(
                "kv_cache_simulator.cli_args", level="DEBUG"
            ) as captured:
                list(
                    log_request_details(
                        tokenize_openai_trace(trace, FakeTokenizer()),
                        log_token_ids=True,
                    )
                )

        messages = [record.getMessage() for record in captured.records]
        self.assertIn("request 1 length=4 tokens", messages)
        self.assertIn("request 1 token_ids=[1, 2, 3, 4]", messages)

    def test_request_details_does_not_materialize_ids_without_flag(self):
        class NonIterableTokenIds(tuple):
            def __iter__(self):
                raise AssertionError("token IDs were materialized for disabled logging")

        class NonIterableTokenizer:
            def tokenize(self, _request):
                return TokenIdsRequest(NonIterableTokenIds((1, 2, 3, 4)))

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            trace.write_text('{"prompt": "hello"}\n', encoding="utf-8")
            tokens = list(
                log_request_details(
                    tokenize_openai_trace(trace, NonIterableTokenizer()),
                    log_token_ids=False,
                )
            )

        self.assertEqual(len(tokens), 1)

    @patch("urllib.request.urlopen")
    def test_engine_tokenizer_calls_engine_endpoint_and_reads_tokens(self, urlopen):
        response = io.BytesIO(json.dumps({"tokens": [7, 8], "count": 2}).encode())
        response.__enter__ = lambda value: value
        response.__exit__ = lambda *_args: None
        urlopen.return_value = response

        tokenizer = EngineTokenizer(
            "http://localhost:30000", api_key="secret", model="served-model"
        )
        result = tokenizer.tokenize({"prompt": "hello", "model": "trace-model"})

        self.assertEqual(result, TokenIdsRequest((7, 8)))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:30000/tokenize")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(json.loads(request.data)["model"], "served-model")

    @patch(
        "kv_cache_simulator.tokenize.vllm._load_vllm_api",
        side_effect=fake_vllm_api,
    )
    def test_vllm_tokenizer_renders_chat_in_process(self, _load_vllm_api):
        tokenizer = VllmTokenizer(
            "models/GLM-5.2-FP8",
            tool_call_parser="glm47",
        )
        result = tokenizer.tokenize(
            {
                "body": {
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {}},
                        }
                    ],
                    "tool_choice": "auto",
                    "max_tokens": 4096,
                }
            }
        )

        self.assertEqual(result, TokenIdsRequest((11, 12, 13)))
        self.assertEqual(FakeVllmRequest.last_body["max_tokens"], 0)
        self.assertEqual(FakeVllmRequest.last_body["max_completion_tokens"], 0)
        self.assertEqual(FakeVllmRequest.last_body["tool_choice"], "auto")
        self.assertEqual(FakeOnlineRenderer.last_kwargs["tool_parser"], "glm47")
        self.assertTrue(FakeOnlineRenderer.last_kwargs["enable_auto_tools"])
        model_config = FakeOnlineRenderer.last_kwargs["model_config"]
        self.assertEqual(model_config.kwargs["model"], "models/GLM-5.2-FP8")
        self.assertIsNone(model_config.quantization)
        tokenizer.close()
        self.assertTrue(tokenizer._base_renderer.was_shutdown)

    @patch(
        "kv_cache_simulator.tokenize.vllm._load_vllm_api",
        side_effect=fake_vllm_api,
    )
    def test_vllm_tokenizer_renders_completion_in_process(self, _load_vllm_api):
        tokenizer = VllmTokenizer("model-dir")

        result = tokenizer.tokenize({"prompt": "hello", "max_tokens": 100})

        self.assertEqual(result, TokenIdsRequest((21, 22)))
        self.assertEqual(FakeVllmRequest.last_body["max_tokens"], 0)

    @patch(
        "kv_cache_simulator.tokenize.sglang._load_sglang_api",
        side_effect=fake_sglang_api,
    )
    def test_sglang_tokenizer_renders_tools_in_process(self, _load_sglang_api):
        tokenizer = SglangTokenizer(
            "models/GLM-5.2-FP8",
            tool_call_parser="glm47",
        )
        result = tokenizer.tokenize(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query": "kv cache"}',
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}},
                    }
                ],
                "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )

        self.assertEqual(result, TokenIdsRequest((31, 32)))
        implementation = tokenizer._tokenizer
        messages, kwargs = implementation.template_call
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            {"query": "kv cache"},
        )
        self.assertEqual(kwargs["tools"][0]["function"]["name"], "lookup")
        self.assertTrue(kwargs["enable_thinking"])
        self.assertFalse(implementation.encode_calls[-1][1]["add_special_tokens"])
        self.assertEqual(FakeSglangFunctionCallParser.constructor_calls, 0)

    @patch(
        "kv_cache_simulator.tokenize.sglang._load_sglang_api",
        side_effect=fake_sglang_api,
    )
    def test_sglang_tokenizer_renders_completion_in_process(self, _load_sglang_api):
        tokenizer = SglangTokenizer("model-dir")

        result = tokenizer.tokenize(
            {"prompt": "hello", "add_special_tokens": False}
        )

        self.assertEqual(result, TokenIdsRequest((11,)))
        self.assertEqual(
            tokenizer._tokenizer.encode_calls[-1],
            ("hello", {"add_special_tokens": False}),
        )

    @patch(
        "kv_cache_simulator.tokenize.sglang._load_sglang_api",
        side_effect=fake_sglang_api,
    )
    def test_sglang_tokenizer_recovers_collapsed_parallel_tool_calls(
        self, _load_sglang_api
    ):
        tokenizer = SglangTokenizer(
            "models/GLM-5.2-FP8",
            tool_call_parser="glm47",
        )

        result = tokenizer.tokenize(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "running both analyses",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "execute_pythonexecute_python",
                                    "arguments": '[{"code": "first"}, {"code": "second"}]',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "tool not found",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "execute_python", "parameters": {}},
                    }
                ],
            }
        )

        self.assertEqual(result, TokenIdsRequest((31, 32)))
        messages, _kwargs = tokenizer._tokenizer.template_call
        recovered_calls = messages[0]["tool_calls"]
        self.assertEqual(
            [call["function"]["name"] for call in recovered_calls],
            ["execute_python", "execute_python"],
        )
        self.assertEqual(
            [call["function"]["arguments"] for call in recovered_calls],
            [{"code": "first"}, {"code": "second"}],
        )
        self.assertEqual(recovered_calls[0]["id"], "call_1")
        self.assertEqual(recovered_calls[1]["id"], "call_1_2")

    @patch(
        "kv_cache_simulator.tokenize.sglang._load_sglang_api",
        side_effect=fake_sglang_api,
    )
    def test_sglang_tokenizer_rejects_unrecoverable_array_arguments(
        self, _load_sglang_api
    ):
        tokenizer = SglangTokenizer("model-dir", tool_call_parser="glm47")

        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            tokenizer.tokenize(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '[{"q": 1}, {"q": 2}]',
                                    },
                                }
                            ],
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {}},
                        }
                    ],
                }
            )

    @patch(
        "kv_cache_simulator.tokenize.sglang._load_sglang_api",
        side_effect=fake_sglang_api,
    )
    def test_sglang_tokenizer_rejects_unknown_tool_parser(self, _load_sglang_api):
        with self.assertRaisesRegex(ValueError, "Unsupported tool_call_parser"):
            SglangTokenizer("model-dir", tool_call_parser="unknown")


if __name__ == "__main__":
    unittest.main()
