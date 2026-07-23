"""
Evaluate a checkpoint and break final-test errors down by group.

This script is meant for diagnosing validation-to-test generalization gaps. It
reports overall MPJPE / PA-MPJPE / PCK and per-subject, per-scene, and
per-action metrics for an existing best.pth.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from common import get_test_loader, load_config, save_json
from evaluate_final_test_mpjpe import as_namespace, build_model
from mmfi_lib.evaluate import calulate_error


def compute_metrics(preds, gts):
    mpjpe, pa, pck = calulate_error(preds, gts)
    return {
        "n": int(preds.shape[0]),
        "mpjpe": float(mpjpe * 1000.0),
        "pa_mpjpe": float(pa * 1000.0),
        "pck@20": float(pck.get("pck@20", 0.0)),
        "pck@50": float(pck.get("pck@50", 0.0)),
    }


def metric_table(values, preds, gts):
    groups = defaultdict(list)
    for idx, value in enumerate(values):
        groups[str(value)].append(idx)

    rows = []
    for name, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        item = {"group": name}
        item.update(compute_metrics(preds[idxs], gts[idxs]))
        rows.append(item)
    return sorted(rows, key=lambda x: x["mpjpe"], reverse=True)


def collect_predictions(model, loader, device, use_root_relative=False):
    model.eval()
    preds_all, gts_all = [], []
    subjects, scenes, actions = [], [], []

    with torch.no_grad():
        for batch in loader:
            wifi_window = batch["input_wifi-csi_window"].to(device)
            gt = batch["output"].to(device)
            out = model(wifi_window)
            pred3d = out[0] if isinstance(out, tuple) else out

            pred_np = pred3d.cpu().numpy()
            gt_np = gt.cpu().numpy()
            if use_root_relative:
                root = ((gt[:, 11:12, :] + gt[:, 12:13, :]) / 2).cpu().numpy()
                pred_np = pred_np + root

            preds_all.append(pred_np)
            gts_all.append(gt_np)
            subjects.extend([str(x) for x in batch.get("subject", [])])
            scenes.extend([str(x) for x in batch.get("scene", [])])
            actions.extend([str(x) for x in batch.get("action", [])])

    return (
        np.concatenate(preds_all, axis=0),
        np.concatenate(gts_all, axis=0),
        {"subject": subjects, "scene": scenes, "action": actions},
    )


def print_worst(title, rows, limit):
    print(f"\nWorst {title}:")
    for row in rows[:limit]:
        print(
            f"  {row['group']}: n={row['n']} "
            f"MPJPE={row['mpjpe']:.2f}mm PA={row['pa_mpjpe']:.2f}mm "
            f"PCK@20={row['pck@20']:.2f} PCK@50={row['pck@50']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Final-test MPJPE breakdown."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_type", choices=["baseline", "lupi"], default="baseline")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--split", default="four_way_split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = as_namespace(ckpt.get("args", {}), args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config_file, args.split)
    test_ds, test_loader = get_test_loader(
        args.dataset_root,
        cfg,
        train_args.window,
        train_args.stride,
        train_args.val_batch_size,
        num_workers=args.num_workers,
    )

    eval_model, load_target = build_model(args.model_type, train_args)
    load_target.load_state_dict(ckpt["model"], strict=True)
    eval_model = eval_model.to(device)

    preds, gts, meta = collect_predictions(
        eval_model,
        test_loader,
        device,
        use_root_relative=getattr(train_args, "use_root_relative", False),
    )

    result = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "n_test": int(len(test_ds)),
        "overall": compute_metrics(preds, gts),
        "by_subject": metric_table(meta["subject"], preds, gts),
        "by_scene": metric_table(meta["scene"], preds, gts),
        "by_action": metric_table(meta["action"], preds, gts),
    }

    out_path = args.output or os.path.join(args.run_dir, "final_test_breakdown.json")
    save_json(result, out_path)

    overall = result["overall"]
    print(f"Device: {device}")
    print(f"Run dir: {args.run_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples (windowed): {len(test_ds)}")
    print(
        f"Overall: MPJPE={overall['mpjpe']:.2f}mm "
        f"PA={overall['pa_mpjpe']:.2f}mm "
        f"PCK@20={overall['pck@20']:.2f} PCK@50={overall['pck@50']:.2f}"
    )
    print_worst("subjects", result["by_subject"], args.top_k)
    print_worst("scenes", result["by_scene"], args.top_k)
    print_worst("actions", result["by_action"], args.top_k)
    print(f"\nSaved breakdown: {out_path}")


if __name__ == "__main__":
    main()
