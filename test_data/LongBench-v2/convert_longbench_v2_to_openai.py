#!/usr/bin/env python3
"""Convert LongBench v2 records into OpenAI-compatible request JSONL.

The input may be either a JSON array (as used by data_short.json) or a
JSONL file containing one LongBench v2 record per line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "context",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
)


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    """Load a JSON array, a single JSON object, or a JSONL file."""
    with path.open("r", encoding="utf-8-sig") as infile:
        first_char = ""
        while True:
            char = infile.read(1)
            if not char:
                return
            if not char.isspace():
                first_char = char
                break

        infile.seek(0)
        if first_char == "[":
            data = json.load(infile)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for index, record in enumerate(data, start=1):
                if not isinstance(record, dict):
                    raise ValueError(f"Record {index} is not a JSON object")
                yield record
            return

        if first_char == "{":
            # A file starting with an object can be a single JSON object or
            # JSONL. Trying JSONL first handles both without loading it all.
            for line_number, line in enumerate(infile, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record on line {line_number} is not a JSON object"
                    )
                yield record
            return

        raise ValueError(f"Unsupported JSON content in {path}")


def format_prompt(record: dict[str, Any], record_number: int) -> str:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"Record {record_number} is missing required fields: {', '.join(missing)}"
        )

    return (
        f"{record['context']}\n\n"
        f"Question: {record['question']}\n"
        f"A. {record['choice_A']}\n"
        f"B. {record['choice_B']}\n"
        f"C. {record['choice_C']}\n"
        f"D. {record['choice_D']}\n"
        "Answer:"
    )


def make_request(
    record: dict[str, Any],
    record_number: int,
    *,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = format_prompt(record, record_number)
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "n": 1,
        "tool_choice": "auto",
    }


def dumps_jsonl_record(record: dict[str, Any]) -> str:
    """Serialize one record without leaving Unicode line separators raw."""
    serialized = json.dumps(record, ensure_ascii=False)
    # Keep each request unambiguously on one physical/logical line even for
    # readers that treat NEL, LINE SEPARATOR, or PARAGRAPH SEPARATOR as a line
    # boundary. JSON decoding restores the original characters.
    return (
        serialized.replace("\u0085", "\\u0085")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LongBench v2 JSON/JSONL dataset to an openai-style "
            "OpenAI Chat Completions request JSONL file."
        )
    )
    parser.add_argument("-i", "--input-path", type=Path, required=True)
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Default: <input stem>.openai.jsonl",
    )
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Maximum generated tokens; LongBench v2's benchmark default is 10.",
    )
    parser.add_argument(
        "--max-request-num",
        type=int,
        default=None,
        help="Only convert the first N records (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero")
    if args.max_request_num is not None and args.max_request_num <= 0:
        raise ValueError("--max-request-num must be greater than zero")

    input_path = args.input_path
    output_path = args.output_path or input_path.with_name(
        f"{input_path.stem}.openai.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        for record_number, record in enumerate(load_records(input_path), start=1):
            if (
                args.max_request_num is not None
                and converted >= args.max_request_num
            ):
                break
            request = make_request(
                record,
                record_number,
                model=args.model,
                max_tokens=args.max_tokens,
            )
            outfile.write(dumps_jsonl_record(request) + "\n")
            converted += 1

    print(f"Converted {converted} requests: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
