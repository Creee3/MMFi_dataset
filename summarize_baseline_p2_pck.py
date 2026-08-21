#!/usr/bin/env python3
"""Rebuild P2 PCK@10/20/30/40/50 from archived baseline logs.

For each model and MM-Fi setting, the script selects the epoch with the lowest
validation MPJPE independently for each seed. All other metrics are read from
that same epoch, matching the checkpoint-selection policy used by the paper.
No model is trained or loaded.
"""

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


MODELS = ("metafi", "hpeli", "dtpose")
SETTINGS = ("S1P2", "S2P2", "S3P2")
METRICS = (
    "mpjpe",
    "pa_mpjpe",
    "pck10",
    "pck20",
    "pck30",
    "pck40",
    "pck50",
)

EPOCH_RE = re.compile(r"In epoch\s+(\d+),")
SEED_MARKER_RE = re.compile(r"Seed\s*=\s*(\d+)")
SEED_FILE_RE = re.compile(r"train_seed_(\d+)")
METRIC_RE = re.compile(
    r"test mpjpe:\s*(?P<mpjpe>[0-9.eE+-]+),\s*"
    r"test pa-mpjpe:\s*(?P<pa_mpjpe>[0-9.eE+-]+),\s*"
    r"test pck50:\s*(?P<pck50>[0-9.eE+-]+),\s*"
    r"test pck40:\s*(?P<pck40>[0-9.eE+-]+),\s*"
    r"test pck30:\s*(?P<pck30>[0-9.eE+-]+),\s*"
    r"test pck20:\s*(?P<pck20>[0-9.eE+-]+),\s*"
    r"test pck10:\s*(?P<pck10>[0-9.eE+-]+)"
)


def parse_metric_rows(text):
    current_epoch = None
    rows = {}
    for line in text.splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        metric_match = METRIC_RE.search(line)
        if metric_match and current_epoch is not None:
            row = {
                key: float(value.rstrip("."))
                for key, value in metric_match.groupdict().items()
            }
            row["epoch"] = current_epoch
            rows[current_epoch] = row
    return [rows[epoch] for epoch in sorted(rows)]


def split_seed_blocks(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = list(SEED_MARKER_RE.finditer(text))
    if markers:
        blocks = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            blocks.append((int(marker.group(1)), text[marker.end():end]))
        return blocks

    seed_match = SEED_FILE_RE.search(path.name)
    if not seed_match:
        return []

    seed = int(seed_match.group(1))
    starts = list(re.finditer(r"Training All Samples:", text))
    if len(starts) <= 1:
        return [(seed, text)]

    blocks = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        blocks.append((seed, text[start.start():end]))
    return blocks


def collect_runs(setting_dir, expected_seeds):
    candidates = {}
    for path in sorted(setting_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".log", ".txt"}:
            continue
        for seed, block in split_seed_blocks(path):
            rows = parse_metric_rows(block)
            if not rows:
                continue
            current = candidates.get(seed)
            if current is None or len(rows) > len(current[1]):
                candidates[seed] = (path, rows)

    expected = set(expected_seeds)
    found = set(candidates)
    if found != expected:
        raise RuntimeError(
            f"{setting_dir}: expected seeds {sorted(expected)}, found {sorted(found)}"
        )
    return candidates


def mean_std(values):
    return statistics.mean(values), statistics.pstdev(values)


def summarize_cell(model, setting, setting_dir, seeds, selection):
    runs = collect_runs(setting_dir, seeds)
    selected = []
    for seed in seeds:
        source_path, rows = runs[seed]
        best = min(rows, key=lambda row: row[selection])
        selected.append(
            {
                "seed": seed,
                "source_log": str(source_path),
                "epoch_count": len(rows),
                **best,
            }
        )

    aggregate = {}
    for metric in METRICS:
        scale = 1000.0 if metric in {"mpjpe", "pa_mpjpe"} else 1.0
        values = [row[metric] * scale for row in selected]
        mean, std = mean_std(values)
        output_name = f"{metric}_mm" if metric in {"mpjpe", "pa_mpjpe"} else metric
        aggregate[f"{output_name}_mean"] = mean
        aggregate[f"{output_name}_std"] = std

    return {
        "model": model,
        "setting": setting,
        "selection": selection,
        "n_seeds": len(selected),
        "seed_metrics": selected,
        **aggregate,
    }


def format_cell(cell, key, digits=1):
    return f"{cell[f'{key}_mean']:.{digits}f} +/- {cell[f'{key}_std']:.{digits}f}"


def build_markdown(cells):
    lines = [
        "# 三种基线模型 P2 的 PCK@10--PCK@50 汇总",
        "",
        "口径：每个 seed 选择 validation MPJPE 最低的 epoch，并从同一 epoch "
        "读取 PA-MPJPE 与全部 PCK；五个 seed 使用总体标准差（ddof=0）。",
        "",
        "| 模型 | 设置 | MPJPE (mm) | PA-MPJPE (mm) | PCK@10 (%) | PCK@20 (%) | PCK@30 (%) | PCK@40 (%) | PCK@50 (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| {model} | {setting} | {mpjpe} | {pa} | {p10} | {p20} | "
            "{p30} | {p40} | {p50} |".format(
                model=cell["model"],
                setting=cell["setting"],
                mpjpe=format_cell(cell, "mpjpe_mm"),
                pa=format_cell(cell, "pa_mpjpe_mm"),
                p10=format_cell(cell, "pck10"),
                p20=format_cell(cell, "pck20"),
                p30=format_cell(cell, "pck30"),
                p40=format_cell(cell, "pck40"),
                p50=format_cell(cell, "pck50"),
            )
        )
    lines.extend(
        [
            "",
            "注意：这些数值由已完成的正式训练日志重建，不是重新训练结果。",
            "归档中的单个代表性 checkpoint 可能被后续 seed 覆盖，不能替代五 seed 日志汇总。",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root", type=Path, default=Path("pose日志【整合】")
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument(
        "--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--selection", choices=("mpjpe", "pa_mpjpe"), default="mpjpe"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("baseline_p2_pck_10_50_summary.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("baseline_p2_pck_10_50_summary_中文.md")
    )
    args = parser.parse_args()

    cells = []
    for model in args.models:
        for setting in args.settings:
            setting_dir = args.log_root / model / setting
            if not setting_dir.is_dir():
                raise FileNotFoundError(setting_dir)
            cell = summarize_cell(
                model, setting, setting_dir, args.seeds, args.selection
            )
            cells.append(cell)
            print(
                f"{model:7s} {setting}: "
                f"MPJPE={format_cell(cell, 'mpjpe_mm')} mm | "
                f"PCK@10/20/30/40/50="
                f"{cell['pck10_mean']:.1f}/"
                f"{cell['pck20_mean']:.1f}/"
                f"{cell['pck30_mean']:.1f}/"
                f"{cell['pck40_mean']:.1f}/"
                f"{cell['pck50_mean']:.1f}"
            )

    payload = {
        "status": "complete",
        "source": "archived_formal_training_logs",
        "evaluation_set": "training-time validation loader",
        "protocol": "P2",
        "selection": args.selection,
        "seeds": args.seeds,
        "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5],
        "cells": cells,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(args.output, payload)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(build_markdown(cells), encoding="utf-8")
    print(f"Saved JSON: {args.output}")
    print(f"Saved Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
