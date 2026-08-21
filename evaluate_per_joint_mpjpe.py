"""Evaluate formal checkpoints with per-joint MPJPE and PA-MPJPE.

This is a post-evaluation script. It does not train or alter checkpoints.
It follows the same validation loader and checkpoint policy as
evaluate_pck_10_50.py, and reports the 17 MM-Fi joints in their canonical order.
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch

from evaluate_pck_10_50 import (
    DEFAULT_RESULTS_ROOT,
    SPLITS,
    build_model,
    checkpoint_signature,
    collect_predictions,
    load_validation_loader,
    saved_arg,
    write_json,
)
from mmfi_lib.evaluate import compute_similarity_transform


JOINT_NAMES = [
    "Bot Torso",
    "L.Hip",
    "L.Knee",
    "L.Foot",
    "R.Hip",
    "R.Knee",
    "R.Foot",
    "Center Torso",
    "Upper Torso",
    "Neck Base",
    "Center Head",
    "R.Shoulder",
    "R.Elbow",
    "R.Hand",
    "L.Shoulder",
    "L.Elbow",
    "L.Hand",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def per_joint_errors(predictions, targets):
    raw = np.linalg.norm(predictions - targets, axis=2).mean(axis=0)
    aligned = np.zeros((predictions.shape[0], predictions.shape[1]), dtype=np.float64)
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        _, aligned_prediction, transform, scale, translation = compute_similarity_transform(
            target, prediction, compute_optimal_scale=True
        )
        aligned_prediction = (scale * prediction.dot(transform)) + translation
        aligned[index] = np.linalg.norm(aligned_prediction - target, axis=1)
    return raw * 1000.0, aligned.mean(axis=0) * 1000.0


def evaluate_checkpoint(checkpoint_path, loader, device, force=False):
    output_path = os.path.join(
        os.path.dirname(checkpoint_path), "per_joint_mpjpe_validation.json"
    )
    signature = checkpoint_signature(checkpoint_path)
    if not force and os.path.isfile(output_path):
        try:
            existing = read_json(output_path)
            if (
                existing.get("status") == "complete"
                and existing.get("checkpoint_signature") == signature
                and existing.get("joint_names") == JOINT_NAMES
            ):
                print(f"  Reusing: {output_path}")
                return existing
        except (OSError, json.JSONDecodeError):
            pass

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
    raw, aligned = per_joint_errors(predictions, targets)
    result = {
        "status": "complete",
        "evaluation_set": "official_validation",
        "checkpoint": checkpoint_path,
        "checkpoint_signature": signature,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "sample_count": int(targets.shape[0]),
        "joint_names": JOINT_NAMES,
        "per_joint_mpjpe_mm": raw.tolist(),
        "per_joint_pa_mpjpe_mm": aligned.tolist(),
        "mpjpe_mm": float(raw.mean()),
        "pa_mpjpe_mm": float(aligned.mean()),
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_path, result)
    del eval_model, load_target, checkpoint, predictions, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mean_std(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0)


def aggregate_cell(protocol, split, rows):
    mpjpe_mean, mpjpe_std = mean_std(rows, "per_joint_mpjpe_mm")
    pa_mean, pa_std = mean_std(rows, "per_joint_pa_mpjpe_mm")
    return {
        "protocol": protocol,
        "split": split,
        "n_seeds": len(rows),
        "joint_names": JOINT_NAMES,
        "seed_metrics": rows,
        "per_joint_mpjpe_mm_mean": mpjpe_mean.tolist(),
        "per_joint_mpjpe_mm_std": mpjpe_std.tolist(),
        "per_joint_pa_mpjpe_mm_mean": pa_mean.tolist(),
        "per_joint_pa_mpjpe_mm_std": pa_std.tolist(),
        "mpjpe_mm_mean": float(mpjpe_mean.mean()),
        "mpjpe_mm_std_across_joint_means": float(mpjpe_std.mean()),
        "pa_mpjpe_mm_mean": float(pa_mean.mean()),
        "pa_mpjpe_mm_std_across_joint_means": float(pa_std.mean()),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate formal PrivPose checkpoints per joint."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--protocol", default="protocol2")
    parser.add_argument("--split", choices=SPLITS, default="cross_subject_split")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--output", default="per_joint_mpjpe_p2s2_summary.json")
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly if intended.")
    device = torch.device(args.device)
    checkpoint_rows = []
    for seed in args.seeds:
        checkpoint_path = os.path.join(
            args.results_root,
            args.protocol,
            args.split,
            "student",
            f"seed_{seed:02d}",
            "best.pth",
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        checkpoint_rows.append((seed, checkpoint_path))

    loader = load_validation_loader(
        args, args.protocol, args.split, checkpoint_rows[0][1]
    )
    rows = []
    for seed, checkpoint_path in checkpoint_rows:
        print(f"Evaluating seed {seed}: {checkpoint_path}")
        result = evaluate_checkpoint(checkpoint_path, loader, device, args.force)
        rows.append({"seed": seed, **result})
        print(
            f"  MPJPE={result['mpjpe_mm']:.2f} mm | "
            f"PA-MPJPE={result['pa_mpjpe_mm']:.2f} mm"
        )

    summary = {
        "status": "complete",
        "protocol": args.protocol,
        "split": args.split,
        "evaluation_set": "official_validation",
        "selection": "mpjpe",
        "thresholds_not_used": True,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **aggregate_cell(args.protocol, args.split, rows),
    }
    write_json(args.output, summary)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
