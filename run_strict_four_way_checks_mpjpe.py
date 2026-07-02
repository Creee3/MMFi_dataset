"""
Run quick strict four-way MPJPE checks with the current S2P2-style settings.

This script is intentionally small: it only launches the already existing
training entrypoints with fixed, comparable arguments and isolated run dirs.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


RESULTS_ROOT = "experiment_results/strict_four_way_checks_mpjpe"
DEFAULT_TEACHER = r"strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth"


COMMON_ARGS = [
    "--split", "four_way_split",
    "--window", "32",
    "--stride", "2",
    "--batch_size", "64",
    "--val_batch_size", "128",
    "--lr", "1.03e-4",
    "--dropout", "0.15",
    "--no_cbam",
    "--epochs", "30",
    "--warmup_epochs", "0",
    "--weight_decay", "1e-3",
    "--lr_patience", "3",
    "--patience", "0",
    "--aug_noise", "0.0",
    "--aug_freq_mask", "0.0",
    "--aug_time_mask", "0.0",
    "--transformer_layers", "2",
    "--dim_feedforward", "1024",
    "--no_root_relative",
    "--device", "cuda",
    "--num_workers", "0",
]


VARIANTS = {
    "baseline_mixup": {
        "entry": "train_baseline_mpjpe.py",
        "extra": ["--mixup_alpha", "0.4"],
    },
    "baseline_nomixup": {
        "entry": "train_baseline_mpjpe.py",
        "extra": ["--mixup_alpha", "0.0"],
    },
    "cmc_mixup": {
        "entry": "train_lupi_rgb_teacher_mpjpe.py",
        "extra": [
            "--mixup_alpha", "0.4",
            "--lambda_cmc", "0.2",
            "--lambda_distill", "0.0",
            "--cmc_stopgrad_rgb",
            "--swa_start", "0",
        ],
    },
    "cmc_nomixup": {
        "entry": "train_lupi_rgb_teacher_mpjpe.py",
        "extra": [
            "--mixup_alpha", "0.0",
            "--lambda_cmc", "0.2",
            "--lambda_distill", "0.0",
            "--cmc_stopgrad_rgb",
            "--swa_start", "0",
        ],
    },
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_variant(name: str, cfg: dict, dataset_root: str, config_file: str,
                teacher_ckpt: str, seed: int, common_aux_args: list,
                cmc_aux_args: list):
    run_dir = os.path.join(RESULTS_ROOT, name, f"seed_{seed:02d}")
    ensure_dir(run_dir)

    cmd = [sys.executable, "-u", cfg["entry"], dataset_root, config_file]
    cmd += COMMON_ARGS + cfg["extra"] + common_aux_args

    if name.startswith("cmc_"):
        cmd += cmc_aux_args + ["--teacher_ckpt", teacher_ckpt]

    cmd += ["--run_dir", run_dir]

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"

    print("=" * 72)
    print(f"Variant: {name}")
    print(f"Seed:    {seed}")
    print(f"Run dir: {run_dir}")
    print("=" * 72)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with returncode={proc.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Run strict four-way MPJPE sanity checks."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "--variants", nargs="+",
        default=["baseline_mixup", "cmc_nomixup"],
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_rel_pose", type=float, default=0.0)
    parser.add_argument("--lambda_root", type=float, default=0.0)
    parser.add_argument("--subject_robust_weight", type=float, default=0.0)
    parser.add_argument("--cmc_start_epoch", type=int, default=1)
    parser.add_argument("--cmc_ramp_epochs", type=int, default=0)
    parser.add_argument("--cmc_encoder_grad_scale", type=float, default=1.0)
    args = parser.parse_args()

    ensure_dir(RESULTS_ROOT)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Variants: {args.variants}")

    if any(v.startswith("cmc_") for v in args.variants):
        if not os.path.isfile(args.teacher_ckpt):
            raise FileNotFoundError(args.teacher_ckpt)

    common_aux_args = [
        "--lambda_rel_pose", str(args.lambda_rel_pose),
        "--lambda_root", str(args.lambda_root),
        "--subject_robust_weight", str(args.subject_robust_weight),
    ]
    cmc_aux_args = [
        "--cmc_start_epoch", str(args.cmc_start_epoch),
        "--cmc_ramp_epochs", str(args.cmc_ramp_epochs),
        "--cmc_encoder_grad_scale", str(args.cmc_encoder_grad_scale),
    ]

    for name in args.variants:
        run_variant(
            name,
            VARIANTS[name],
            args.dataset_root,
            args.config_file,
            args.teacher_ckpt,
            args.seed,
            common_aux_args,
            cmc_aux_args,
        )

    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
