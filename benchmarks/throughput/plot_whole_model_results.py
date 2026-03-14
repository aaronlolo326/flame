from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MODEL_ORDER = [
    "lact",
    "e2e_ttt",
    "hybrid_lact",
    "full_attention",
    "hybrid_swa",
    "hybrid_gdn",
]

MODEL_LABELS = {
    "lact": "LaCT",
    "e2e_ttt": "E2E-TTT",
    "hybrid_lact": "75% LaCT + 25% FA",
    "full_attention": "Full Attention",
    "hybrid_swa": "75% SWA + 25% FA",
    "hybrid_gdn": "75% GDN + 25% FA",
}

STATUS_TO_SCORE = {
    "ok": 2,
    "oom": 1,
    "unsupported_backend": 1,
    "error": 1,
    "skipped_after_failure": 0,
    "skipped_after_oom": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot whole-model benchmark result figures.")
    parser.add_argument("--input", type=Path, required=True, help="Input whole-model CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "figures",
        help="Directory to write PNG figures.",
    )
    parser.add_argument("--title-suffix", default="", help="Optional title suffix.")
    return parser.parse_args()


def to_int(value: str) -> int:
    return int(float(value))


def to_float(value: str) -> float:
    return float(value)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    out = []
    for row in rows:
        out.append(
            {
                **row,
                "seq_len": to_int(row["seq_len"]),
                "forward_ms": to_float(row["forward_ms"]),
                "backward_ms": to_float(row["backward_ms"]),
                "optimizer_ms": to_float(row["optimizer_ms"]),
                "step_ms": to_float(row["step_ms"]),
                "tokens_per_second": to_float(row["tokens_per_second"]),
                "peak_memory_gb": to_float(row["peak_memory_gb"]),
            }
        )
    return out


def model_sort_key(model: str) -> tuple[int, str]:
    if model in MODEL_ORDER:
        return (MODEL_ORDER.index(model), model)
    return (len(MODEL_ORDER), model)


def title_with_suffix(base: str, suffix: str) -> str:
    return f"{base} {suffix}".strip()


def series_key(row: dict) -> tuple[str, str]:
    return row["model"], row.get("runtime_env", "unknown")


def series_label(model: str, runtime_env: str) -> str:
    base = MODEL_LABELS.get(model, model)
    return f"{base} [{runtime_env}]"


def make_throughput_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    by_series = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_series[series_key(row)].append(row)

    for (model, runtime_env) in sorted(by_series.keys(), key=lambda x: model_sort_key(x[0])):
        points = sorted(by_series[(model, runtime_env)], key=lambda x: x["seq_len"])
        ax.plot(
            [p["seq_len"] for p in points],
            [p["tokens_per_second"] for p in points],
            marker="o",
            linewidth=2,
            label=series_label(model, runtime_env),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tokens / second")
    ax.set_title(title_with_suffix("Whole-Model Throughput", title_suffix))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "whole_model_tokens_per_second.png", dpi=180)
    plt.close(fig)


def make_step_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    by_series = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_series[series_key(row)].append(row)

    for (model, runtime_env) in sorted(by_series.keys(), key=lambda x: model_sort_key(x[0])):
        points = sorted(by_series[(model, runtime_env)], key=lambda x: x["seq_len"])
        ax.plot(
            [p["seq_len"] for p in points],
            [p["step_ms"] for p in points],
            marker="o",
            linewidth=2,
            label=series_label(model, runtime_env),
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Step Time (ms)")
    ax.set_title(title_with_suffix("Whole-Model Step Time", title_suffix))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "whole_model_step_ms.png", dpi=180)
    plt.close(fig)


def make_memory_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    by_series = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_series[series_key(row)].append(row)

    for (model, runtime_env) in sorted(by_series.keys(), key=lambda x: model_sort_key(x[0])):
        points = sorted(by_series[(model, runtime_env)], key=lambda x: x["seq_len"])
        ax.plot(
            [p["seq_len"] for p in points],
            [p["peak_memory_gb"] for p in points],
            marker="o",
            linewidth=2,
            label=series_label(model, runtime_env),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title(title_with_suffix("Whole-Model Peak Memory", title_suffix))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "whole_model_peak_memory.png", dpi=180)
    plt.close(fig)


def make_breakdown_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    by_series = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_series[series_key(row)].append(row)

    for (model, runtime_env) in sorted(by_series.keys(), key=lambda x: model_sort_key(x[0])):
        points = sorted(by_series[(model, runtime_env)], key=lambda x: x["seq_len"])
        seq = [p["seq_len"] for p in points]
        fwd = [p["forward_ms"] for p in points]
        bwd = [p["backward_ms"] for p in points]
        opt = [p["optimizer_ms"] for p in points]
        label = series_label(model, runtime_env)
        axes[0].plot(seq, fwd, marker="o", linewidth=2, label=label)
        axes[1].plot(seq, bwd, marker="o", linewidth=2, label=label)
        axes[2].plot(seq, opt, marker="o", linewidth=2, label=label)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Sequence Length")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_title("Forward ms")
    axes[1].set_title("Backward ms")
    axes[2].set_title("Optimizer ms")
    axes[0].set_ylabel("ms")
    axes[2].legend()
    fig.suptitle(title_with_suffix("Whole-Model Forward/Backward/Optimizer Breakdown", title_suffix))
    fig.tight_layout()
    fig.savefig(output_dir / "whole_model_forward_backward_optimizer.png", dpi=180)
    plt.close(fig)


def make_status_map(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    series = sorted({series_key(row) for row in rows}, key=lambda x: model_sort_key(x[0]))
    seqs = sorted({row["seq_len"] for row in rows})
    matrix = np.zeros((len(series), len(seqs)), dtype=float)

    index = {(row["model"], row.get("runtime_env", "unknown"), row["seq_len"]): row for row in rows}
    for i, (model, runtime_env) in enumerate(series):
        for j, seq in enumerate(seqs):
            row = index.get((model, runtime_env, seq))
            if row is None:
                matrix[i, j] = -1
            else:
                matrix[i, j] = STATUS_TO_SCORE.get(row["status"], 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=-1, vmax=2)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([series_label(m, e) for m, e in series])
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels([str(s) for s in seqs], rotation=45, ha="right")
    ax.set_xlabel("Sequence Length")
    ax.set_title(title_with_suffix("Whole-Model Status Map (-1 missing, 0 skipped, 1 fail/oom, 2 ok)", title_suffix))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "whole_model_status_map.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    rows = [row for row in rows if row.get("benchmark") == "whole_model_train"]
    if not rows:
        raise RuntimeError("No whole_model_train rows found in input.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_throughput_plot(rows, args.output_dir, args.title_suffix)
    make_step_plot(rows, args.output_dir, args.title_suffix)
    make_memory_plot(rows, args.output_dir, args.title_suffix)
    make_breakdown_plot(rows, args.output_dir, args.title_suffix)
    make_status_map(rows, args.output_dir, args.title_suffix)
    print(f"Wrote figures to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
