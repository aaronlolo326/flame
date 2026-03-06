from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MODEL_ORDER = [
    "lact_full_layer",
    "lact_ttt_branch_only",
    "fa_branch_only",
    "swa_branch_only",
    "gdn_branch_only",
]

MODEL_LABELS = {
    "lact_full_layer": "LaCT Full",
    "lact_ttt_branch_only": "LaCT TTT Only",
    "fa_branch_only": "FA",
    "swa_branch_only": "SWA",
    "gdn_branch_only": "GDN",
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
    parser = argparse.ArgumentParser(description="Plot single-kernel benchmark result figures.")
    parser.add_argument("--input", type=Path, required=True, help="Input single-kernel CSV.")
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


def make_throughput_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    by_model = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_model[row["model"]].append(row)

    for model in sorted(by_model.keys(), key=model_sort_key):
        points = sorted(by_model[model], key=lambda x: x["seq_len"])
        ax.plot(
            [p["seq_len"] for p in points],
            [p["tokens_per_second"] for p in points],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tokens / second")
    ax.set_title(title_with_suffix("Single-Kernel Throughput", title_suffix))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "single_kernel_tokens_per_second.png", dpi=180)
    plt.close(fig)


def make_step_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    by_model = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_model[row["model"]].append(row)

    for model in sorted(by_model.keys(), key=model_sort_key):
        points = sorted(by_model[model], key=lambda x: x["seq_len"])
        ax.plot(
            [p["seq_len"] for p in points],
            [p["step_ms"] for p in points],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Step Time (ms)")
    ax.set_title(title_with_suffix("Single-Kernel Step Time", title_suffix))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "single_kernel_step_ms.png", dpi=180)
    plt.close(fig)


def make_memory_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    by_model = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_model[row["model"]].append(row)

    for model in sorted(by_model.keys(), key=model_sort_key):
        points = sorted(by_model[model], key=lambda x: x["seq_len"])
        ax.plot(
            [p["seq_len"] for p in points],
            [p["peak_memory_gb"] for p in points],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title(title_with_suffix("Single-Kernel Peak Memory", title_suffix))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "single_kernel_peak_memory.png", dpi=180)
    plt.close(fig)


def make_breakdown_plot(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    by_model = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_model[row["model"]].append(row)

    for model in sorted(by_model.keys(), key=model_sort_key):
        points = sorted(by_model[model], key=lambda x: x["seq_len"])
        seq = [p["seq_len"] for p in points]
        fwd = [p["forward_ms"] for p in points]
        bwd = [p["backward_ms"] for p in points]
        axes[0].plot(seq, fwd, marker="o", linewidth=2, label=MODEL_LABELS.get(model, model))
        axes[1].plot(seq, bwd, marker="o", linewidth=2, label=MODEL_LABELS.get(model, model))

    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=10)
    axes[1].set_yscale("log", base=10)
    axes[0].set_title("Forward ms")
    axes[1].set_title("Backward ms")
    axes[0].set_xlabel("Sequence Length")
    axes[1].set_xlabel("Sequence Length")
    axes[0].set_ylabel("ms")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()
    fig.suptitle(title_with_suffix("Single-Kernel Forward/Backward Breakdown", title_suffix))
    fig.tight_layout()
    fig.savefig(output_dir / "single_kernel_forward_backward.png", dpi=180)
    plt.close(fig)


def make_status_map(rows: list[dict], output_dir: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted({row["model"] for row in rows}, key=model_sort_key)
    seqs = sorted({row["seq_len"] for row in rows})
    matrix = np.zeros((len(models), len(seqs)), dtype=float)
    for i, model in enumerate(models):
        for j, seq in enumerate(seqs):
            chosen = None
            for row in rows:
                if row["model"] == model and row["seq_len"] == seq:
                    chosen = row
                    break
            if chosen is None:
                matrix[i, j] = -1
            else:
                matrix[i, j] = STATUS_TO_SCORE.get(chosen["status"], 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=-1, vmax=2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels([str(s) for s in seqs], rotation=45, ha="right")
    ax.set_xlabel("Sequence Length")
    ax.set_title(title_with_suffix("Single-Kernel Status Map (-1 missing, 0 skipped, 1 fail/oom, 2 ok)", title_suffix))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "single_kernel_status_map.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    rows = [row for row in rows if row.get("benchmark") == "single_kernel_train"]
    if not rows:
        raise RuntimeError("No single_kernel_train rows found in input.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_throughput_plot(rows, args.output_dir, args.title_suffix)
    make_step_plot(rows, args.output_dir, args.title_suffix)
    make_memory_plot(rows, args.output_dir, args.title_suffix)
    make_breakdown_plot(rows, args.output_dir, args.title_suffix)
    make_status_map(rows, args.output_dir, args.title_suffix)
    print(f"Wrote figures to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
