import argparse
import json
import logging

"""
example useage:
python convert_request_evalscope.py \
    -i dsv4_data_10.jsonl \
    --max-request-num 10
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dumped requests to evalscope/openai dataset")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "-o", "--output-path", type=str, default=None, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--max-request-num",
        type=int,
        default=1000000000,
        help="max request num to process",
    )
    args = parser.parse_args()
    print(f"args: {args}")
    if args.output_path is None:
        args.output_path = args.input_path[:-6] + ".evalscope.jsonl"
    output_path = args.output_path

    line_cnt = 0
    failed_cnt = 0
    print(f"write output file to: {output_path}")
    with open(args.input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            line_cnt += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON on line {line_cnt}: {exc}")
                continue

            if "content" in record:
                payload = record["content"]
            elif "payload" in record:
                payload = record["payload"]
            else:
                raise ValueError(f"no valid content for line {line_cnt}")

            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception as err:
                    logging.error(f"load line {line_cnt} failed")
                    failed_cnt += 1
                    continue
            outfile.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if line_cnt >= args.max_request_num:
                break

    print(f"processed {line_cnt} lines, failed num: {failed_cnt}")
