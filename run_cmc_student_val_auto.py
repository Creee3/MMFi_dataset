import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


RESULTS_ROOT = Path("experiment_results/cmc_student_val")
TEACHER_RUN_DIR = Path("strict_offline_runs/T1_rgb_teacher_student_split_mpjpe")
SPLIT = "teacher_student_split"


def run_command(cmd, env=None):
    print("\n" + "=" * 80, flush=True)
    print("Running:", flush=True)
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd), flush=True)
    print("=" * 80, flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def print_summary(results_root):
    summary_path = results_root / "summary.json"
    if not summary_path.is_file():
        print(f"No summary found yet: {summary_path}", flush=True)
        return
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    print("\nTop student-validation candidates:", flush=True)
    for item in summary[:10]:
        sources = sorted({
            metric.get("checkpoint_source", "unknown")
            for metric in item.get("metrics", [])
        })
        print(
            f"{item['mpjpe_mean']:.2f}+/-{item['mpjpe_std']:.2f} mm | "
            f"PA {item['pa_mpjpe_mean']:.2f}+/-{item['pa_mpjpe_std']:.2f} | "
            f"source={','.join(sources)} | {item['tag']}",
            flush=True,
        )
    print(f"Saved summary: {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain teacher and CMC student under the no-test student-validation protocol."
    )
    parser.add_argument("--dataset_root", default="data_base")
    parser.add_argument("--config_file", default="config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--teacher_run_dir", default=str(TEACHER_RUN_DIR))
    parser.add_argument("--teacher_epochs", type=int, default=30)
    parser.add_argument("--teacher_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--force_teacher", action="store_true")
    parser.add_argument("--lambda_cmc", type=float, default=0.02)
    parser.add_argument("--cmc_start_epoch", type=int, default=1)
    parser.add_argument("--cmc_ramp_epochs", type=int, default=0)
    parser.add_argument("--cmc_encoder_grad_scale", type=float, default=0.25)
    parser.add_argument("--tail_ckpts", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    teacher_run_dir = Path(args.teacher_run_dir)
    teacher_ckpt = teacher_run_dir / "best.pth"

    print(f"Protocol split: {SPLIT}", flush=True)
    print(f"Teacher checkpoint: {teacher_ckpt}", flush=True)
    print("Student validation uses the official cross-subject validation subjects.", flush=True)

    run_command([sys.executable, "check_project_integrity.py"], env=env)

    teacher_cmd = [
        sys.executable,
        "train_t1_rgb_teacher_mpjpe.py",
        args.dataset_root,
        args.config_file,
        "--split",
        SPLIT,
        "--run_dir",
        str(teacher_run_dir),
        "--epochs",
        str(args.teacher_epochs),
        "--batch_size",
        str(args.teacher_batch_size),
        "--val_batch_size",
        str(args.val_batch_size),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
    ]

    if args.dry_run:
        print("\nDry run teacher command:", flush=True)
        print(" ".join(teacher_cmd), flush=True)
    elif args.force_teacher or not teacher_ckpt.is_file():
        run_command(teacher_cmd, env=env)
    else:
        print(f"\nSkip teacher training; found existing checkpoint: {teacher_ckpt}", flush=True)

    grid_cmd = [
        sys.executable,
        "run_strict_mpjpe_grid_search.py",
        args.dataset_root,
        args.config_file,
        "--split",
        SPLIT,
        "--mode",
        "cmc",
        "--teacher_ckpt",
        str(teacher_ckpt),
        "--results_root",
        str(RESULTS_ROOT),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--lrs",
        "7e-5",
        "--mixup_alphas",
        "0.16",
        "--dropouts",
        "0.15",
        "--transformer_layers",
        "2",
        "--dim_feedforwards",
        "1024",
        "--pose_head_hiddens",
        "512",
        "--batch_sizes",
        "64",
        "--aug_noises",
        "0.02",
        "--aug_freq_masks",
        "0.05",
        "--aug_time_masks",
        "0.05",
        "--lambda_rel_poses",
        "0.0",
        "--lambda_roots",
        "0.0",
        "--lambda_bones",
        "0.005",
        "--pose_head_types",
        "graph_root",
        "--subject_robust_weights",
        "0.0",
        "--lambda_cmcs",
        str(args.lambda_cmc),
        "--cmc_starts",
        str(args.cmc_start_epoch),
        "--cmc_ramps",
        str(args.cmc_ramp_epochs),
        "--cmc_grad_scales",
        str(args.cmc_encoder_grad_scale),
        "--swa_start",
        "0",
        "--tail_ckpts",
        str(args.tail_ckpts),
        "--tail_soup",
        "--num_workers",
        str(args.num_workers),
        "--val_batch_size",
        str(args.val_batch_size),
        "--device",
        args.device,
    ]
    if args.dry_run:
        grid_cmd.append("--dry_run")

    run_command(grid_cmd, env=env)
    if not args.dry_run:
        print_summary(RESULTS_ROOT)


if __name__ == "__main__":
    main()
