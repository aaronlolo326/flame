from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple throughput result files from different environments into one CSV/JSONL pair."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input CSV or JSONL result files.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output path prefix, without suffix.")
    parser.add_argument(
        "--sort-keys",
        nargs="+",
        default=["benchmark", "model", "seq_len", "runtime_env"],
        help="Row keys used for sorting the merged output.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported input type: {path}")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.setdefault("runtime_env", "unknown")
        normalized.append(item)
    return normalized


def sort_value(row: dict[str, Any], key: str) -> tuple[int, Any]:
    value = row.get(key, "")
    if isinstance(value, (int, float)):
        return (0, value)
    if isinstance(value, str):
        try:
            return (0, int(value))
        except ValueError:
            try:
                return (0, float(value))
            except ValueError:
                return (1, value)
    return (1, str(value))


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        all_rows.extend(normalize_rows(read_rows(input_path)))

    if not all_rows:
        raise RuntimeError("No rows found in the provided input files.")

    all_fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for row in all_rows:
        for key in row.keys():
            if key not in seen_fields:
                seen_fields.add(key)
                all_fieldnames.append(key)

    all_rows.sort(key=lambda row: tuple(sort_value(row, key) for key in args.sort_keys))

    output_csv = args.output_prefix.with_suffix(".csv")
    output_jsonl = args.output_prefix.with_suffix(".jsonl")
    ensure_parent_dir(output_csv)
    ensure_parent_dir(output_jsonl)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Merged {len(all_rows)} rows into {output_csv} and {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
