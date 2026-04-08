#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_BASELINE_LCC = Path(
    "/work/yufei/projects/flame/results/20260330_ttt_lora/lb/Qwen__Qwen3-4B/"
    "samples_longbench_lcc_2026-04-02T23-54-27.592922.jsonl"
)
DEFAULT_NEW_LCC = Path(
    "/work/yufei/projects/flame/results/20260330_ttt_lora/lb/Qwen__Qwen3-4B/"
    "samples_longbench_lcc_2026-04-06T07-43-20.284827.jsonl"
)
DEFAULT_BASELINE_REPO = Path(
    "/work/yufei/projects/flame/results/20260330_ttt_lora/lb/Qwen__Qwen3-4B/"
    "samples_longbench_repobench-p_2026-04-02T23-54-27.592922.jsonl"
)
DEFAULT_NEW_REPO = Path(
    "/work/yufei/projects/flame/results/20260330_ttt_lora/lb/Qwen__Qwen3-4B/"
    "samples_longbench_repobench-p_2026-04-06T07-43-20.284827.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/work/yufei/projects/flame/results/20260330_ttt_lora/coding_delta_analysis"
)


@dataclass
class SampleDelta:
    task: str
    doc_id: int
    base_score: float
    new_score: float
    delta: float
    target: str
    prompt: str
    base_resp: str
    new_resp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze coding-task gains between two LongBench runs.")
    parser.add_argument("--baseline-lcc", type=Path, default=DEFAULT_BASELINE_LCC)
    parser.add_argument("--new-lcc", type=Path, default=DEFAULT_NEW_LCC)
    parser.add_argument("--baseline-repobench", type=Path, default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--new-repobench", type=Path, default=DEFAULT_NEW_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            rows[int(obj["doc_id"])] = obj
    return rows


def build_deltas(task: str, baseline_path: Path, new_path: Path) -> List[SampleDelta]:
    baseline = load_jsonl(baseline_path)
    new = load_jsonl(new_path)
    deltas: List[SampleDelta] = []
    for doc_id in sorted(set(baseline) & set(new)):
        b = baseline[doc_id]
        n = new[doc_id]
        deltas.append(
            SampleDelta(
                task=task,
                doc_id=doc_id,
                base_score=float(b["score"]),
                new_score=float(n["score"]),
                delta=float(n["score"]) - float(b["score"]),
                target=str(b["target"]),
                prompt=str(b["arguments"]["gen_args_0"]["arg_0"]),
                base_resp=str(b["filtered_resps"][0]),
                new_resp=str(n["filtered_resps"][0]),
            )
        )
    return deltas


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(values: List[float]) -> dict:
    sv = sorted(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": sv[0],
        "p25": quantile(sv, 0.25),
        "median": quantile(sv, 0.5),
        "p75": quantile(sv, 0.75),
        "max": sv[-1],
        "positive_rate": sum(v > 0 for v in values) / len(values),
        "negative_rate": sum(v < 0 for v in values) / len(values),
    }


def histogram(values: List[float], bins: Iterable[float]) -> List[dict]:
    edges = list(bins)
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            left, right = edges[i], edges[i + 1]
            if (i < len(edges) - 2 and left <= v < right) or (i == len(edges) - 2 and left <= v <= right):
                counts[i] += 1
                break
    total = len(values)
    out = []
    for i, count in enumerate(counts):
        out.append(
            {
                "left": edges[i],
                "right": edges[i + 1],
                "count": count,
                "share": count / total if total else 0.0,
                "bar": "#" * max(1, round((count / total) * 30)) if count else "",
            }
        )
    return out


def categorize(sample: SampleDelta) -> str | None:
    if sample.base_score <= 0.2 and sample.new_score >= 0.9:
        return "exact_win_from_wrong"
    if sample.delta >= 0.5:
        return "strong_gain"
    if sample.delta >= 0.2:
        return "moderate_gain"
    if sample.base_score >= 0.9 and sample.new_score <= 0.2:
        return "exact_drop_from_correct"
    if sample.delta <= -0.5:
        return "strong_drop"
    if sample.delta <= -0.2:
        return "moderate_drop"
    return None


def choose_representative(samples: List[SampleDelta]) -> SampleDelta:
    median_delta = quantile(sorted(s.delta for s in samples), 0.5)
    return min(
        samples,
        key=lambda s: (abs(s.delta - median_delta), -abs(s.delta), s.doc_id),
    )


def short(text: str, limit: int = 500) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def write_example(path: Path, sample: SampleDelta, category: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"category: {category}",
                f"task: {sample.task}",
                f"doc_id: {sample.doc_id}",
                f"base_score: {sample.base_score:.4f}",
                f"new_score: {sample.new_score:.4f}",
                f"delta: {sample.delta:+.4f}",
                "",
                "target:",
                short(sample.target),
                "",
                "prompt_tail:",
                short(sample.prompt[-1200:], limit=1200),
                "",
                "baseline_response:",
                short(sample.base_resp, limit=1200),
                "",
                "new_response:",
                short(sample.new_resp, limit=1200),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_deltas = (
        build_deltas("lcc", args.baseline_lcc, args.new_lcc)
        + build_deltas("repobench-p", args.baseline_repobench, args.new_repobench)
    )
    task_to_deltas = {
        "lcc": [d for d in all_deltas if d.task == "lcc"],
        "repobench-p": [d for d in all_deltas if d.task == "repobench-p"],
        "overall": all_deltas,
    }

    bins = [-1.0, -0.8, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 0.8, 1.0]
    summary = {}
    lines = ["# Coding Delta Analysis", ""]

    for name, deltas in task_to_deltas.items():
        values = [d.delta for d in deltas]
        stats = summarize(values)
        hist = histogram(values, bins)
        summary[name] = {"stats": stats, "histogram": hist}

        lines.append(f"## {name}")
        lines.append(
            f"count={stats['count']} mean={stats['mean']:+.4f} std={stats['std']:.4f} "
            f"median={stats['median']:+.4f} p25={stats['p25']:+.4f} p75={stats['p75']:+.4f} "
            f"positive_rate={stats['positive_rate']:.3f} negative_rate={stats['negative_rate']:.3f}"
        )
        lines.append("")
        lines.append("Histogram:")
        for row in hist:
            lines.append(
                f"[{row['left']:+.2f}, {row['right']:+.2f}] "
                f"{row['count']:>4} {row['bar']}"
            )
        lines.append("")

    category_dir = args.output_dir / "representative_examples"
    category_dir.mkdir(parents=True, exist_ok=True)
    category_summary = {}
    category_names = [
        "exact_win_from_wrong",
        "strong_gain",
        "moderate_gain",
        "exact_drop_from_correct",
        "strong_drop",
        "moderate_drop",
    ]

    for scope_name, deltas in task_to_deltas.items():
        if scope_name == "overall":
            continue
        scoped = {name: [] for name in category_names}
        for sample in deltas:
            cat = categorize(sample)
            if cat is not None:
                scoped[cat].append(sample)
        category_summary[scope_name] = {
            cat: len(samples) for cat, samples in scoped.items() if samples
        }
        for cat, samples in scoped.items():
            if not samples:
                continue
            rep = choose_representative(samples)
            write_example(category_dir / f"{scope_name}__{cat}.txt", rep, cat)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "category_counts.json").write_text(json.dumps(category_summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(
        {
            "output_dir": str(args.output_dir),
            "summary_md": str(args.output_dir / "summary.md"),
            "summary_json": str(args.output_dir / "summary.json"),
            "category_counts_json": str(args.output_dir / "category_counts.json"),
            "representative_examples_dir": str(category_dir),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
