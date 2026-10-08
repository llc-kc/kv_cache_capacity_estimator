# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from mattson_capacity_analyzer import (
    MattsonStack,
    Request,
    analyze_capacities,
    chained_page_hashes,
    load_requests,
    parse_size,
)
from trace_parser import ParsedRequest, parse_jsonl, resolve_parser


class MattsonCapacityAnalyzerTest(unittest.TestCase):
    def test_known_reuse_distances(self):
        stack = MattsonStack(6)
        distances = [stack.access(key) for key in b"ABCABA"]
        self.assertEqual(distances, [None, None, None, 2, 2, 1])

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

    def test_load_requests_uses_replaceable_parser(self):
        def custom_parser(_path):
            yield ParsedRequest((1, 2, 3, 4))

        requests = load_requests(
            Path("unused"),
            page_size=2,
            trace_parser=custom_parser,
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0].page_keys,
            chained_page_hashes((1, 2, 3, 4), page_size=2),
        )

    def test_resolve_builtin_parser(self):
        self.assertIs(resolve_parser("jsonl"), parse_jsonl)

    def test_resolve_importable_parser(self):
        self.assertIs(resolve_parser("trace_parser:parse_jsonl"), parse_jsonl)


if __name__ == "__main__":
    unittest.main()
