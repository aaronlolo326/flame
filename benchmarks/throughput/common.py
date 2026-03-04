from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_SEQ_LENS = [
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
]


@dataclass
class BenchmarkRow:
    benchmark: str
    model: str
    runtime_env: str
    seq_len: int
    batch_size: int
    warmup_steps: int
    measured_steps: int
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    step_ms: float
    steps_per_second: float
    tokens_per_second: float
    peak_memory_gb: float
    status: str
    notes: str = ""


def now_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows_csv(path: Path, rows: Iterable[BenchmarkRow]) -> None:
    rows = list(rows)
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def write_rows_jsonl(path: Path, rows: Iterable[BenchmarkRow]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def append_row_jsonl(path: Path, row: BenchmarkRow) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def benchmark_timer(
    *,
    warmup_steps: int,
    measured_steps: int,
    sync_fn: Callable[[], None],
    step_fn: Callable[[], tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    for _ in range(warmup_steps):
        step_fn()
    sync_fn()

    forward_total = 0.0
    backward_total = 0.0
    optimizer_total = 0.0
    started = time.perf_counter()
    for _ in range(measured_steps):
        forward_ms, backward_ms, optimizer_ms = step_fn()
        forward_total += forward_ms
        backward_total += backward_ms
        optimizer_total += optimizer_ms
    sync_fn()
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    return (
        forward_total / measured_steps,
        backward_total / measured_steps,
        optimizer_total / measured_steps,
        elapsed_ms / measured_steps,
    )
def log(message: str) -> None:
    print(message, flush=True)
