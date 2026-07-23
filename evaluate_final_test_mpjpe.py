"""
Evaluate an existing MPJPE checkpoint on the configured evaluation split.

This is useful when training finished successfully but Windows failed during
the final test DataLoader stage because of multiprocessing / page-file limits.
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from common import (
    TemporalLUPICMC_CBAM,
    TemporalWiFiStudentCBAM,
    eval_temporal_model,
    fmt_metrics,
    get_test_loader,
    load_config,
    save_json,
)


def as_namespace(saved_args: dict, cli_args):
    merged = dict(saved_args or {})
    merged["dataset_root"] = cli_args.dataset_root
    merged["config_file"] = cli_args.config_file
    merged["split"] = cli_args.split
    merged["device"] = cli_args.device
    merged["num_workers"] = cli_args.num_workers
    merged["val_batch_size"] = cli_args.val_batch_size or merged.get("val_batch_size", 128)
    return SimpleNamespace(**merged)


class WiFiOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wifi_window):
        return self.model.forward_wifi(wifi_window)


def build_model(model_type: str, args):
    if model_type == "lupi":
        model = TemporalLUPICMC_CBAM(
            in_ch=3,
            d_model=args.d_model,
            d_proj=getattr(args, "cmc_proj_dim", 128),
            window=args.window,
            use_cbam=args.use_cbam,
            dropout=args.dropout,
            num_layers=args.transformer_layers,
            dim_feedforward=args.dim_feedforward,
            pose_head_hidden=getattr(args, "pose_head_hidden", 512),
            pose_head_type=getattr(args, "pose_head_type", "mlp"),
        )
        return WiFiOnlyWrapper(model), model

    if model_type == "baseline":
        model = TemporalWiFiStudentCBAM(
            in_ch=3,
            d_model=args.d_model,
            window=args.window,
            use_cbam=args.use_cbam,
            dropout=args.dropout,
            pose_head_hidden=getattr(args, "pose_head_hidden", 512),
            num_layers=getattr(args, "transformer_layers", 2),
            dim_feedforward=getattr(args, "dim_feedforward", 1024),
            pose_head_type=getattr(args, "pose_head_type", "mlp"),
        )
        return model, model

    raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an existing best.pth on the configured evaluation split."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_type", choices=["lupi", "baseline"], default="lupi")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--split", default="four_way_split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = as_namespace(ckpt.get("args", {}), args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config_file, args.split)
    heldout_cfg = cfg.get("heldout_split", {})
    test_ds, test_loader = get_test_loader(
        args.dataset_root,
        cfg,
        train_args.window,
        train_args.stride,
        train_args.val_batch_size,
        num_workers=args.num_workers,
    )
    print(f"Device: {device}")
    print(f"Run dir: {args.run_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Protocol: {cfg.get('protocol', 'unknown_protocol')}")
    print("Held-out split: "
          f"enabled={heldout_cfg.get('enabled', False)} | "
          f"unit={heldout_cfg.get('unit', 'window')} | "
          f"test_size={heldout_cfg.get('test_size', 0.5)} | "
          f"seed={heldout_cfg.get('random_seed', 41)}")
    if not heldout_cfg.get("enabled", False):
        print("Evaluation set: MMFi original held-out validation/eval split.")
    print(f"Test samples (windowed): {len(test_ds)}")
    print(f"Window: {train_args.window}, Stride: {train_args.stride}")
    print(f"CBAM: {train_args.use_cbam}, Dropout: {train_args.dropout}")
    print(f"Pose head: {getattr(train_args, 'pose_head_type', 'mlp')}")
    print(f"num_workers: {args.num_workers}")

    eval_model, load_target = build_model(args.model_type, train_args)
    load_target.load_state_dict(ckpt["model"], strict=True)
    eval_model = eval_model.to(device)

    mpjpe, pa, pck = eval_temporal_model(
        eval_model,
        test_loader,
        device,
        use_root_relative=getattr(train_args, "use_root_relative", False),
    )
    result = {
        "mpjpe": mpjpe,
        "pa_mpjpe": pa,
        "pck@20": pck.get("pck@20", 0.0),
        "pck@50": pck.get("pck@50", 0.0),
        "checkpoint": args.checkpoint,
        "checkpoint_source": ckpt.get("checkpoint_source", "unknown"),
    }
    if args.output is not None:
        out_path = args.output
    elif args.checkpoint == "best.pth":
        out_path = os.path.join(args.run_dir, "final_test.json")
    else:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(args.run_dir, f"final_test_{ckpt_name}.json")
    save_json(result, out_path)
    print(f"Final test: {fmt_metrics(mpjpe, pa, pck)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
