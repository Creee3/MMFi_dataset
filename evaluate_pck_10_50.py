"""Post-evaluate formal 3x3 checkpoints with PCK@10/20/30/40/50.

The script reuses each run's best.pth and the same official validation split
used during checkpoint selection. It does not train or modify any model.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime


PROJECT_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_python_deps")
if os.path.isdir(PROJECT_DEPS) and PROJECT_DEPS not in sys.path:
    sys.path.insert(0, PROJECT_DEPS)

import numpy as np
import torch
import torch.nn as nn

from common import TemporalLUPICMC_CBAM, get_loaders, load_config
from mmfi_lib.evaluate import calulate_error, compute_pck


PROTOCOLS = ("protocol1", "protocol2", "protocol3")
SPLITS = ("random_split", "cross_subject_split", "cross_scene_split")
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_RESULTS_ROOT = os.path.join("strict_offline_runs", "S2P2_3x3_rank2")


class WiFiOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wifi_window):
        return self.model.forward_wifi(wifi_window)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def saved_arg(saved, name, default):
    value = saved.get(name, default)
    return default if value is None else value


def build_model(saved_args):
    model = TemporalLUPICMC_CBAM(
        in_ch=3,
        d_model=saved_arg(saved_args, "d_model", 512),
        d_proj=saved_arg(saved_args, "cmc_proj_dim", 128),
        window=saved_arg(saved_args, "window", 32),
        use_cbam=saved_arg(saved_args, "use_cbam", False),
        dropout=saved_arg(saved_args, "dropout", 0.15),
        num_layers=saved_arg(saved_args, "transformer_layers", 2),
        dim_feedforward=saved_arg(saved_args, "dim_feedforward", 1024),
        pose_head_hidden=saved_arg(saved_args, "pose_head_hidden", 512),
        pose_head_type=saved_arg(saved_args, "pose_head_type", "mlp"),
    )
    return WiFiOnlyWrapper(model), model


def collect_predictions(model, loader, device, use_root_relative=False):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            wifi_window = batch["input_wifi-csi_window"].to(device)
            target = batch["output"].to(device)
            output = model(wifi_window)
            prediction = output[0] if isinstance(output, tuple) else output

            prediction_np = prediction.cpu().numpy()
            if use_root_relative:
                root = ((target[:, 11:12, :] + target[:, 12:13, :]) / 2)
                prediction_np = prediction_np + root.cpu().numpy()

            predictions.append(prediction_np)
            targets.append(target.cpu().numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def checkpoint_signature(path):
    stat = os.stat(path)
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def reusable_result(path, checkpoint_path):
    if not os.path.isfile(path):
        return None
    try:
        result = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("status") != "complete":
        return None
    if result.get("thresholds") != list(THRESHOLDS):
        return None
    if result.get("checkpoint_signature") != checkpoint_signature(checkpoint_path):
        return None
    return result


def evaluate_checkpoint(checkpoint_path, loader, device, force=False):
    run_dir = os.path.dirname(checkpoint_path)
    output_path = os.path.join(run_dir, "pck_10_50_validation.json")
    if not force:
        existing = reusable_result(output_path, checkpoint_path)
        if existing is not None:
            print(f"  Reusing: {output_path}")
            return existing

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = dict(checkpoint.get("args", {}))
    eval_model, load_target = build_model(saved_args)
    load_target.load_state_dict(checkpoint["model"], strict=True)
    eval_model = eval_model.to(device)

    predictions, targets = collect_predictions(
        eval_model,
        loader,
        device,
        use_root_relative=bool(saved_arg(saved_args, "use_root_relative", False)),
    )
    mpjpe, pa_mpjpe, _ = calulate_error(predictions, targets)
    pck = compute_pck(predictions, targets, thresholds=THRESHOLDS)
    result = {
        "status": "complete",
        "evaluation_set": "official_validation",
        "checkpoint": checkpoint_path,
        "checkpoint_signature": checkpoint_signature(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "sample_count": int(targets.shape[0]),
        "thresholds": list(THRESHOLDS),
        "mpjpe_mm": float(mpjpe * 1000.0),
        "pa_mpjpe_mm": float(pa_mpjpe * 1000.0),
        **{name: float(value) for name, value in pck.items()},
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_path, result)
    del eval_model, load_target, checkpoint, predictions, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mean_and_std(rows, key):
    values = [float(row[key]) for row in rows]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std


def aggregate_cell(protocol, split, seed_rows):
    metric_keys = [
        "mpjpe_mm", "pa_mpjpe_mm",
        "pck@10", "pck@20", "pck@30", "pck@40", "pck@50",
    ]
    aggregate = {}
    for key in metric_keys:
        mean, std = mean_and_std(seed_rows, key)
        aggregate[f"{key}_mean"] = mean
        aggregate[f"{key}_std"] = std
    return {
        "protocol": protocol,
        "split": split,
        "n_seeds": len(seed_rows),
        "seed_metrics": seed_rows,
        **aggregate,
    }


def load_validation_loader(args, protocol, split, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = dict(checkpoint.get("args", {}))
    del checkpoint

    os.environ["MMFI_PROTOCOL"] = protocol
    os.environ["HELDOUT_SPLIT_ENABLED"] = "0"
    cfg = load_config(args.config_file, split)
    val_batch_size = args.val_batch_size or int(
        saved_arg(saved_args, "val_batch_size", 256))
    _, val_dataset, _, val_loader = get_loaders(
        args.dataset_root,
        cfg,
        window=int(saved_arg(saved_args, "window", 32)),
        stride=int(saved_arg(saved_args, "stride", 2)),
        batch_size=1,
        val_batch_size=val_batch_size,
        role="student",
        num_workers=0,
        eval_num_workers=args.eval_num_workers,
    )
    print(f"  Validation samples: {len(val_dataset)} | batch size: {val_batch_size}")
    return val_loader


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate formal best.pth files with PCK@10/20/30/40/50."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--protocols", nargs="+", choices=PROTOCOLS, default=["protocol2"])
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly if intended.")
    device = torch.device(args.device)
    cells = []

    for protocol in args.protocols:
        for split in args.splits:
            print("\n" + "=" * 78)
            print(f"{protocol} / {split}")
            checkpoint_rows = []
            for seed in args.seeds:
                checkpoint_path = os.path.join(
                    args.results_root,
                    protocol,
                    split,
                    "student",
                    f"seed_{seed:02d}",
                    "best.pth",
                )
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(checkpoint_path)
                checkpoint_rows.append((seed, checkpoint_path))
                print(f"  seed {seed}: {checkpoint_path}")
            if args.dry_run:
                continue

            val_loader = load_validation_loader(
                args, protocol, split, checkpoint_rows[0][1])
            seed_rows = []
            for seed, checkpoint_path in checkpoint_rows:
                print(f"  Evaluating seed {seed}...")
                result = evaluate_checkpoint(
                    checkpoint_path, val_loader, device, force=args.force)
                seed_rows.append({"seed": seed, **result})
                print(
                    f"    MPJPE={result['mpjpe_mm']:.1f} mm | "
                    f"PA-MPJPE={result['pa_mpjpe_mm']:.1f} mm | "
                    f"PCK@10/20/30/40/50="
                    f"{result['pck@10']:.1f}/"
                    f"{result['pck@20']:.1f}/"
                    f"{result['pck@30']:.1f}/"
                    f"{result['pck@40']:.1f}/"
                    f"{result['pck@50']:.1f}"
                )
            cell = aggregate_cell(protocol, split, seed_rows)
            cells.append(cell)

            partial_output = args.output or os.path.join(
                args.results_root, "pck_10_50_validation_summary.json")
            write_json(
                partial_output,
                {
                    "status": "complete" if len(cells) == (
                        len(args.protocols) * len(args.splits)) else "running",
                    "evaluation_set": "official_validation",
                    "thresholds": list(THRESHOLDS),
                    "cells": cells,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

    if args.dry_run:
        print("\nDry run complete; no checkpoint was evaluated.")
        return
    print(f"\nSaved summary: {partial_output}")


if __name__ == "__main__":
    main()
