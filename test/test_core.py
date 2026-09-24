# SPDX-License-Identifier: Apache-2.0

import argparse
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from kv_cache_simulator import simulation_config_from_dict
from kv_cache_simulator.analysis import (
    MattsonStack,
    analyze_capacities,
    request_capacity_stats,
)
from kv_cache_simulator.cli_args import (
    log_arguments,
    log_replay_progress,
    parse_size,
)
from kv_cache_simulator.models import PageRequest as Request
from kv_cache_simulator.models import TokenIdsRequest as ParsedRequest
from kv_cache_simulator.pages import chained_page_hashes, requests_from_token_ids
from kv_cache_simulator.simulator import ReplaySimulator, SimulationConfig
from kv_cache_simulator.token_id_replay import parse_jsonl, resolve_parser


class CoreSimulationTest(unittest.TestCase):
    def test_json_config_uses_package_defaults(self):
        config = simulation_config_from_dict({"kv_bytes_per_token": 61505})

        self.assertEqual(config.page_size, 64)
        self.assertEqual(config.capacities[0], ("100GiB", 100 * 2**30))
        self.assertEqual(config.capacities[-1], ("64TiB", 64 * 2**40))
        self.assertFalse(config.include_partial_page)
        self.assertEqual(config.warm_up_ratio, 0)
        self.assertEqual(config.target_hit_rate_ratio, 0.99)

    def test_json_config_accepts_human_readable_overrides(self):
        config = simulation_config_from_dict(
            {
                "kv_bytes_per_token": 10,
                "page_size": 32,
                "capacities": ["1GiB", ["custom", "1.5GB"]],
                "target_hit_rate_ratio": 1.0,
            }
        )

        self.assertEqual(config.page_size, 32)
        self.assertEqual(
            config.capacities,
            (("1GiB", 2**30), ("custom", 1_500_000_000)),
        )
        self.assertEqual(config.target_hit_rate_ratio, 1.0)

    def test_json_config_rejects_missing_and_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "kv_bytes_per_token"):
            simulation_config_from_dict({})
        with self.assertRaisesRegex(ValueError, "unknown simulator config fields"):
            simulation_config_from_dict(
                {"kv_bytes_per_token": 10, "output_path": "result.json"}
            )

    def test_known_reuse_distances(self):
        stack = MattsonStack(6)
        distances = [stack.access(key) for key in b"ABCABA"]
        self.assertEqual(distances, [None, None, None, 2, 2, 1])

    def test_reuse_distance_stack_grows_without_a_trace_size(self):
        stack = MattsonStack()
        distances = [stack.access(key) for key in b"ABCABA"]
        self.assertEqual(distances, [None, None, None, 2, 2, 1])

        undersized_stack = MattsonStack(1)
        distances = [undersized_stack.access(key) for key in b"ABCABA"]
        self.assertEqual(distances, [None, None, None, 2, 2, 1])

    def test_stateful_replay_analyzes_each_request_as_it_arrives(self):
        replay = ReplaySimulator(
            SimulationConfig(
                page_size=1,
                kv_bytes_per_token=10,
                capacities=(("10B", 10), ("20B", 20)),
                target_hit_rate_ratio=1.0,
            )
        )

        first = replay.process(ParsedRequest((1, 2)))
        self.assertEqual(replay.request_count, 1)
        self.assertEqual(first.capacity_page_hits, (0, 0))
        self.assertEqual(first.required_capacity_bytes, 0)

        second = replay.process(ParsedRequest((1, 3)))
        self.assertEqual(replay.request_count, 2)
        self.assertEqual(second.capacity_page_hits, (0, 1))
        self.assertEqual(second.capacity_prefix_hits, (0, 1))
        self.assertEqual(second.required_capacity_bytes, 20)

        result = replay.finish()
        self.assertEqual([item.page_hits for item in result.results], [0, 1, 1])
        self.assertEqual(result.request_capacity_stats.samples, 2)
        with self.assertRaisesRegex(RuntimeError, "after finish"):
            replay.process(ParsedRequest((1,)))

    def test_warm_up_requirements_are_excluded_from_stats(self):
        replay = ReplaySimulator(
            SimulationConfig(
                page_size=1,
                kv_bytes_per_token=10,
                capacities=(("10B", 10),),
                warm_up_ratio=2 / 3,
                target_hit_rate_ratio=1.0,
            )
        )
        replay.process(ParsedRequest((1, 2, 3)))
        warm_up = replay.process(ParsedRequest((1,)))
        self.assertEqual(warm_up.required_capacity_bytes, 30)
        replay.process(ParsedRequest((4,)))

        result = replay.finish()
        self.assertEqual(result.warm_up_requests, 2)
        self.assertEqual(result.request_capacity_stats.samples, 1)
        self.assertEqual(result.request_capacity_stats.p50_capacity_bytes, 0)

    def test_prefix_hash_sharing(self):
        first = chained_page_hashes(range(8), page_size=4)
        second = chained_page_hashes(
            [0, 1, 2, 3, 8, 9, 10, 11], page_size=4
        )
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_capacity_curve(self):
        requests = [
            Request(tuple(bytes([key]) for key in b"ABC")),
            Request(tuple(bytes([key]) for key in b"ABA")),
        ]
        results = analyze_capacities(
            requests,
            page_bytes=10,
            capacities=[("10B", 10), ("20B", 20), ("30B", 30)],
        )
        self.assertEqual(
            [result.page_hits for result in results], [0, 1, 3, 3]
        )
        self.assertEqual(
            [result.prefix_hit_pages for result in results], [0, 0, 3, 3]
        )
        self.assertNotIn("full_prefix_hit_requests", asdict(results[0]))
        self.assertNotIn("full_prefix_hit_request_rate", asdict(results[0]))

    def test_infinite_capacity_is_always_reported(self):
        requests = [Request(tuple(bytes([key]) for key in b"ABCABA"))]
        results = analyze_capacities(
            requests,
            page_bytes=10,
            capacities=[("0B", 0)],
        )

        finite, infinite = results
        self.assertEqual(finite.page_hits, 0)
        self.assertEqual(infinite.requested_capacity, "infinite")
        self.assertIsNone(infinite.requested_capacity_bytes)
        self.assertIsNone(infinite.effective_capacity_bytes)
        self.assertIsNone(infinite.capacity_pages)
        self.assertEqual(infinite.page_hits, 3)
        self.assertEqual(infinite.page_misses, 3)
        self.assertEqual(infinite.page_hit_rate, 0.5)
        self.assertEqual(infinite.reusable_page_hit_rate, 1.0)

        other_infinite = analyze_capacities(
            requests,
            page_bytes=10,
            capacities=[("1TiB", 2**40)],
        )[-1]
        self.assertEqual(infinite, other_infinite)

    def test_capacity_units(self):
        self.assertEqual(parse_size("10GB"), 10_000_000_000)
        self.assertEqual(parse_size("10GiB"), 10 * 2**30)

    def test_warm_up_populates_cache_but_is_excluded_from_metrics(self):
        requests = [
            Request(tuple(bytes([key]) for key in b"AB")),
            Request(tuple(bytes([key]) for key in b"AB")),
        ]
        results = analyze_capacities(
            requests,
            page_bytes=10,
            capacities=[("20B", 20)],
            warm_up_requests=1,
        )

        self.assertEqual(results[0].page_hits, 2)
        self.assertEqual(results[0].page_misses, 0)
        self.assertEqual(results[-1].page_hits, 2)

    def test_request_prefix_capacity_and_percentiles(self):
        stats = request_capacity_stats(
            [
                (None, None),
                (1, 1),
                (0, 3),
            ],
            page_bytes=10,
            target_hit_rate_ratio=1.0,
        )

        self.assertEqual(stats.samples, 3)
        self.assertEqual(stats.mean_capacity_bytes, (0 + 20 + 40) / 3)
        self.assertEqual(stats.p50_capacity_bytes, 20)
        self.assertEqual(stats.p90_capacity_bytes, 40)
        self.assertEqual(stats.p95_capacity_bytes, 40)
        self.assertEqual(stats.p99_capacity_bytes, 40)
        self.assertEqual(stats.p999_capacity_bytes, 40)

    def test_request_capacity_requires_the_leading_prefix_not_any_pages(self):
        stats = request_capacity_stats(
            [tuple(range(99, -1, -1))],
            page_bytes=10,
            target_hit_rate_ratio=0.9,
        )

        self.assertEqual(stats.samples, 1)
        # The leading 90 pages have depths 99..10, so their maximum depth
        # requires 100 cache pages. Selecting any 90 pages would need only 90.
        self.assertEqual(stats.mean_capacity_bytes, 1_000)
        self.assertEqual(stats.p50_capacity_bytes, 1_000)

    def test_request_capacity_ratio_avoids_float_rounding_extra_page(self):
        stats = request_capacity_stats(
            [(0,) * 14 + (99,) + (0,) * 85],
            page_bytes=10,
            target_hit_rate_ratio=0.14,
        )

        self.assertEqual(stats.p50_capacity_bytes, 10)

    def test_request_log_includes_cumulative_infinite_cache_capacity(self):
        with self.assertLogs(
            "kv_cache_simulator.analysis", level="DEBUG"
        ) as captured:
            request_capacity_stats(
                [
                    (None, None),
                    (0, None),
                ],
                page_bytes=2**30,
                target_hit_rate_ratio=1.0,
                request_number_offset=3,
                initial_infinite_cache_pages=2,
            )

        messages = [record.getMessage() for record in captured.records]
        request_logs = [
            message
            for message in messages
            if "prefix capacity requirement" in message
        ]
        self.assertIn("request 4 prefix capacity requirement", request_logs[0])
        self.assertIn("accumulated_cache_pages=4", request_logs[0])
        self.assertIn("accumulated_cache_capacity_gib=4.000000", request_logs[0])
        self.assertIn("request 5 prefix capacity requirement", request_logs[1])
        self.assertIn("accumulated_cache_pages=5", request_logs[1])
        self.assertIn("accumulated_cache_capacity_gib=5.000000", request_logs[1])

    def test_request_log_includes_hit_rate_for_configured_capacities(self):
        replay = ReplaySimulator(
            SimulationConfig(
                page_size=1,
                kv_bytes_per_token=10,
                capacities=(("10B", 10), ("20B", 20)),
            )
        )
        replay.process(ParsedRequest((1, 2)))

        with self.assertLogs(
            "kv_cache_simulator.analysis", level="DEBUG"
        ) as captured:
            replay.process(ParsedRequest((1, 3)))

        messages = [record.getMessage() for record in captured.records]
        configured_capacity_logs = [
            message
            for message in messages
            if "configured capacity hit-rate estimate" in message
        ]
        self.assertEqual(len(configured_capacity_logs), 2)
        self.assertIn("request 2", configured_capacity_logs[0])
        self.assertIn("capacity=10B", configured_capacity_logs[0])
        self.assertIn("page_hits=0", configured_capacity_logs[0])
        self.assertIn("page_accesses=2", configured_capacity_logs[0])
        self.assertIn("hit_rate=0.000000", configured_capacity_logs[0])
        self.assertIn("capacity=20B", configured_capacity_logs[1])
        self.assertIn("page_hits=1", configured_capacity_logs[1])
        self.assertIn("hit_rate=0.500000", configured_capacity_logs[1])

    def test_replay_progress_logs_at_interval_and_completion(self):
        with self.assertLogs("kv_cache_simulator.cli_args", level="INFO") as captured:
            requests = list(log_replay_progress(range(5), interval=2))

        self.assertEqual(requests, list(range(5)))
        messages = [record.getMessage() for record in captured.records]
        self.assertIn("requests=2", messages[0])
        self.assertIn("requests=4", messages[1])
        self.assertIn("requests=5", messages[2])

    def test_replay_arguments_are_logged_with_credentials_redacted(self):
        with self.assertLogs(
            "kv_cache_simulator.cli_args", level="INFO"
        ) as captured:
            log_arguments(
                argparse.Namespace(
                    trace=Path("requests.jsonl"),
                    page_size=64,
                    api_key="secret-value",
                )
            )

        message = captured.records[0].getMessage()
        self.assertIn('"api_key": "***"', message)
        self.assertIn('"page_size": 64', message)
        self.assertIn('"trace": "requests.jsonl"', message)
        self.assertNotIn("secret-value", message)

    def test_jsonl_parser_accepts_one_input_id_array_per_line(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.log"
            trace.write_text("[1, 2, 3, 4]\n[]\n", encoding="utf-8")
            self.assertEqual(
                list(parse_jsonl(trace)),
                [ParsedRequest((1, 2, 3, 4)), ParsedRequest(())],
            )

    def test_jsonl_parser_accepts_objects_and_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.jsonl"
            trace.write_text(
                '{"token_ids": [1, 2]}\n'
                '[[3, 4], {"tokens": [5, 6]}]\n',
                encoding="utf-8",
            )
            self.assertEqual(
                list(parse_jsonl(trace)),
                [
                    ParsedRequest((1, 2)),
                    ParsedRequest((3, 4)),
                    ParsedRequest((5, 6)),
                ],
            )

    def test_page_requests_accept_replaceable_parser_output(self):
        def custom_parser(_path):
            yield ParsedRequest((1, 2, 3, 4))

        requests = requests_from_token_ids(
            custom_parser(Path("unused")),
            page_size=2,
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0].page_keys,
            chained_page_hashes((1, 2, 3, 4), page_size=2),
        )

    def test_resolve_builtin_parser(self):
        self.assertIs(resolve_parser("jsonl"), parse_jsonl)

    def test_resolve_importable_parser(self):
        self.assertIs(
            resolve_parser("kv_cache_simulator.token_id_replay:parse_jsonl"),
            parse_jsonl,
        )


if __name__ == "__main__":
    unittest.main()
