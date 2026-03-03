from __future__ import annotations

import csv
import json
import os
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
    seq_len: int
    batch_size: int
    world_size: int
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


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(torch_module: object, backend: str = "nccl") -> tuple[int, int, int]:
    if not is_distributed():
        return get_rank(), get_local_rank(), get_world_size()
    if not torch_module.distributed.is_initialized():
        torch_module.distributed.init_process_group(backend=backend)
    if torch_module.cuda.is_available():
        torch_module.cuda.set_device(get_local_rank())
    return get_rank(), get_local_rank(), get_world_size()


def barrier(torch_module: object) -> None:
    if is_distributed() and torch_module.distributed.is_initialized():
        torch_module.distributed.barrier()


def destroy_distributed(torch_module: object) -> None:
    if is_distributed() and torch_module.distributed.is_initialized():
        torch_module.distributed.destroy_process_group()


def reduce_scalar(torch_module: object, value: float, op_name: str) -> float:
    if not is_distributed():
        return float(value)
    tensor = torch_module.tensor([value], device=f"cuda:{get_local_rank()}" if torch_module.cuda.is_available() else "cpu")
    op = getattr(torch_module.distributed.ReduceOp, op_name.upper())
    torch_module.distributed.all_reduce(tensor, op=op)
    return float(tensor.item())
