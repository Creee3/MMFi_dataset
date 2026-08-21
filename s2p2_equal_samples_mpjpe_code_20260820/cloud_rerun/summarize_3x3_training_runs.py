#!/usr/bin/env python3
"""Summarize completed 3x3 training runs into a readable grid.

The input is the manifest produced by
``run_3x3_equal_samples_three_models.py``.  This report is a training-stage
validation summary: MPJPE and PA-MPJPE are converted from meters to mm, while
the trainer's PCK values remain percentages.  It is intentionally separate
from the canonical exact-sample evaluator used for the final paper table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


CELLS = [
    "P1-S1", "P2-S1", "P3-S1",
    "P1-S2", "P2-S2", "P3-S2",
    "P1-S3", "P2-S3", "P3-S3",
]
MODEL_ORDER = ["metafi", "hpeli", "dtpose"]
MODEL_LABELS = {
    "metafi": "MetaFi++",
    "hpeli": "HPE-Li",
    "dtpose": "DT-Pose",
}
METRICS = [
    ("mpjpe_mm", "MPJPE (mm)"),
    ("pa_mpjpe_mm", "PA-MPJPE (mm)"),
    ("pck10", "PCK@10 (%)"),
    ("pck20", "PCK@20 (%)"),
    ("pck30", "PCK@30 (%)"),
    ("pck40", "PCK@40 (%)"),
    ("pck50", "PCK@50 (%)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a complete or partial 3x3 training manifest."
    )
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--allow_partial", action="store_true",
        help="Write a partial table instead of requiring all requested runs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_record_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def find_selection(run_dir: Path) -> Path:
    candidates = sorted(
        path for path in (run_dir / "weights").rglob("selection.json")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one selection.json below {run_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def load_run_record(record: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    cell = record.get("cell")
    model = record.get("model")
    seed = int(record.get("seed"))
    if cell not in CELLS or model not in MODEL_ORDER:
        raise RuntimeError(f"Unexpected run identity: {cell}/{model}")
    run_dir = resolve_record_path(record["run_dir"], manifest_path)
    selection_path = find_selection(run_dir)
    selection = read_json(selection_path)
    if selection.get("selection_metric") != "mpjpe":
        raise RuntimeError(
            f"{cell}/{model}/seed_{seed:02d} is not MPJPE-selected: "
            f"{selection.get('selection_metric')!r}"
        )
    if selection.get("selected_epoch") is None:
        raise RuntimeError(f"No selected epoch in {selection_path}")
    required = ["mpjpe", "pa_mpjpe", "pck10", "pck20", "pck30", "pck40", "pck50"]
    missing = [key for key in required if key not in selection]
    if missing:
        raise RuntimeError(f"Missing metrics in {selection_path}: {missing}")
    # train_pose.py evaluates coordinates in meters.  PCK is already emitted
    # as a percentage in [0, 100], so only the two distance metrics scale.
    return {
        "cell": cell,
        "model": model,
        "seed": seed,
        "selected_epoch": int(selection["selected_epoch"]),
        "selection_path": str(selection_path),
        "mpjpe_mm": float(selection["mpjpe"]) * 1000.0,
        "pa_mpjpe_mm": float(selection["pa_mpjpe"]) * 1000.0,
        "pck10": float(selection["pck10"]),
        "pck20": float(selection["pck20"]),
        "pck30": float(selection["pck30"]),
        "pck40": float(selection["pck40"]),
        "pck50": float(selection["pck50"]),
    }


def aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["seed"]))
    result: Dict[str, Any] = {
        "n_seeds": len(ordered),
        "seeds": [int(row["seed"]) for row in ordered],
        "selected_epochs": [int(row["selected_epoch"]) for row in ordered],
        "seed_results": ordered,
    }
    for key, _ in METRICS:
        values = [float(row[key]) for row in ordered]
        result[f"{key}_mean"] = mean(values)
        result[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return result


def fmt(result: Dict[str, Any], key: str) -> str:
    return f"{result[f'{key}_mean']:.1f} +/- {result[f'{key}_std']:.1f}"


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# MM-Fi 三模型 3×3 训练验证汇总",
        "",
        "本表读取每个 run 的 `selection.json`，只接受按 validation MPJPE 选择的 checkpoint。MPJPE 和 PA-MPJPE 已从米转换为毫米，PCK 为百分比。",
        "",
        "这是训练阶段的 validation 汇总，不等同于 canonical exact-sample evaluator 生成的最终论文表。",
        "",
        f"已完成格子：{len(report['available_cells'])}/9；已完成 run：{report['completed_runs']}/{report['requested_runs']}。",
        "",
    ]
    for setting_index in range(1, 4):
        setting = f"S{setting_index}"
        lines.extend([
            f"## {setting}",
            "",
            "| Protocol | 模型 | MPJPE (mm) | PA-MPJPE (mm) | PCK@10 (%) | PCK@20 (%) | PCK@30 (%) | PCK@40 (%) | PCK@50 (%) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for protocol_index in range(1, 4):
            cell = f"P{protocol_index}-{setting}"
            for model in MODEL_ORDER:
                aggregate_result = report["cells"].get(cell, {}).get("models", {}).get(model)
                if aggregate_result is None:
                    values = ["待完成"] * len(METRICS)
                else:
                    values = [fmt(aggregate_result, key) for key, _ in METRICS]
                lines.append(
                    f"| P{protocol_index} | {MODEL_LABELS[model]} | "
                    + " | ".join(values) + " |"
                )
        lines.append("")
    lines.extend([
        "## 说明",
        "",
        "- 每个格子目标帧为 `idx=62..296`，每段 235 个 target-time 样本。",
        "- 只有 `selection_metric: mpjpe` 的结果进入本表。",
        "- 论文最终对比前，建议用 canonical evaluator 对每个 MPJPE-best checkpoint 做精确 sample-key 对齐复核。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "cell", "model", "n_seeds", "seeds",
            *[f"{key}_mean" for key, _ in METRICS],
            *[f"{key}_std" for key, _ in METRICS],
        ])
        for cell in CELLS:
            for model in MODEL_ORDER:
                result = report["cells"].get(cell, {}).get("models", {}).get(model)
                if result is None:
                    continue
                writer.writerow([
                    cell, model, result["n_seeds"], ";".join(map(str, result["seeds"])),
                    *[result[f"{key}_mean"] for key, _ in METRICS],
                    *[result[f"{key}_std"] for key, _ in METRICS],
                ])


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    manifest_path = results_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    expected_runs = int(manifest.get("requested_run_count", -1))
    if manifest.get("checkpoint_selection") != "mpjpe":
        raise RuntimeError("The training manifest is not MPJPE-selected")
    expected_cells = list(manifest.get("cells", []))
    expected_models = list(manifest.get("models", []))
    expected_seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if expected_cells != CELLS:
        raise RuntimeError(
            "This report expects all nine cells; rerun the training launcher with --all_cells."
        )
    if expected_models != MODEL_ORDER or expected_seeds != [0, 1, 2]:
        raise RuntimeError(
            "This report expects metafi/hpeli/dtpose and seeds 0/1/2."
        )
    records = manifest.get("runs", [])
    completed_rows: List[Dict[str, Any]] = []
    failures = []
    for record in records:
        if record.get("status") != "completed":
            continue
        try:
            completed_rows.append(load_run_record(record, manifest_path))
        except Exception as error:
            failures.append(f"{record.get('cell')}/{record.get('model')}/seed_{record.get('seed')}: {error}")
    if failures:
        raise RuntimeError("Completed run validation failed:\n" + "\n".join(failures))
    expected_key_set = {
        (cell, model, seed)
        for cell in expected_cells for model in expected_models for seed in expected_seeds
    }
    actual_key_set = {(row["cell"], row["model"], row["seed"]) for row in completed_rows}
    missing = sorted(expected_key_set - actual_key_set)
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"Grid is incomplete: {len(missing)} runs missing. "
            "Use --allow_partial only for an interim report."
        )
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in completed_rows:
        grouped.setdefault(row["cell"], {}).setdefault(row["model"], []).append(row)
    cells: Dict[str, Any] = {}
    for cell, by_model in grouped.items():
        cells[cell] = {
            "models": {
                model: aggregate(rows)
                for model, rows in by_model.items()
            }
        }
    report = {
        "schema_version": 1,
        "status": "complete_grid" if not missing else "partial_grid",
        "source_kind": "trainer_selection_summary",
        "results_root": str(results_root),
        "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
        "evaluation_set": "validation",
        "checkpoint_selection": "mpjpe",
        "unit_conversion": "distance values from meters to millimeters; PCK unchanged",
        "requested_runs": expected_runs,
        "completed_runs": len(completed_rows),
        "available_cells": sorted(cells),
        "missing_runs": [list(item) for item in missing],
        "cells": cells,
    }
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "summary_3x3"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_3x3_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "training_3x3_summary_中文.md", report)
    write_csv(output_dir / "training_3x3_summary.csv", report)
    print(
        f"Training 3x3 summary written: {output_dir} | "
        f"completed={len(completed_rows)}/{expected_runs}"
    )


if __name__ == "__main__":
    main()
