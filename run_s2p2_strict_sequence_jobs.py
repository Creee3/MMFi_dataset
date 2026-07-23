"""
Run S2P2 strict non-overlap validation experiments.

This wrapper keeps the HPE-Li-compatible results untouched by writing to a
separate results root and setting HELDOUT_SPLIT_UNIT=sequence. In this mode,
all windows from the same scene-subject-action sequence are assigned wholly to
either validation or test.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from run_s2p2_wifi_cmc_mix20 import DEFAULT_TEACHER, ensure_dir


DEFAULT_RESULTS_ROOT = os.path.join(
    "strict_offline_runs", "S2P2_wifi_cmc_mix20_strict_seq")
DEFAULT_JOBS = "1,4,5,7,8"


def write_json(obj, path):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run strict sequence-level S2P2 selected jobs.")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--jobs", default=DEFAULT_JOBS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scheduler", default="plateau",
                        choices=["plateau", "cosine"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout_test_size", type=float, default=0.5)
    parser.add_argument("--heldout_seed", type=int, default=41)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.results_root)
    split_info = {
        "protocol": "S2P2 strict sequence-level held-out split",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "heldout_split_unit": "sequence",
        "heldout_split_test_size": args.heldout_test_size,
        "heldout_split_seed": args.heldout_seed,
        "jobs": args.jobs,
        "results_root": args.results_root,
        "note": (
            "All windows from the same scene-subject-action sequence are kept "
            "on the same validation/test side. This avoids temporal window "
            "overlap between validation and test."
        ),
    }
    write_json(split_info, os.path.join(args.results_root, "strict_split_info.json"))

    cmd = [
        sys.executable,
        "run_s2p2_selected_jobs.py",
        args.dataset_root,
        args.config_file,
        "--teacher_ckpt", args.teacher_ckpt,
        "--results_root", args.results_root,
        "--jobs", args.jobs,
        "--epochs", str(args.epochs),
        "--window", str(args.window),
        "--stride", str(args.stride),
        "--batch_size", str(args.batch_size),
        "--val_batch_size", str(args.val_batch_size),
        "--num_workers", str(args.num_workers),
        "--eval_num_workers", str(args.eval_num_workers),
        "--device", args.device,
        "--scheduler", args.scheduler,
        "--min_lr", str(args.min_lr),
        "--patience", str(args.patience),
        "--log_every", str(args.log_every),
        "--seed", str(args.seed),
    ]
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry_run")

    env = os.environ.copy()
    env["HELDOUT_SPLIT_ENABLED"] = "1"
    env["HELDOUT_SPLIT_UNIT"] = "sequence"
    env["HELDOUT_SPLIT_TEST_SIZE"] = str(args.heldout_test_size)
    env["HELDOUT_SPLIT_SEED"] = str(args.heldout_seed)
    env["PYTHONIOENCODING"] = "utf-8"

    print("=" * 80)
    print("S2P2 strict sequence-level selected jobs")
    print(f"Results root: {args.results_root}")
    print(f"Jobs: {args.jobs}")
    print("Held-out split: unit=sequence | "
          f"test_size={args.heldout_test_size} | seed={args.heldout_seed}")
    print(" ".join(cmd))
    print("=" * 80)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
