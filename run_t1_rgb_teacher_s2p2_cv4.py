"""
Run 4-fold subject-level RGB teacher selection inside S2P2 train subjects only.

The official S2P2 held-out subjects are never used here. Each fold uses 24 of
the 32 MMFi train subjects for RGB teacher training and 8 train subjects for
teacher-internal validation.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


RESULTS_ROOT = os.path.join("strict_offline_runs", "T1_rgb_teacher_s2p2_cv4")

FOLDS = [
    ("fold01", ["S01", "S02", "S11", "S12", "S21", "S22", "S31", "S32"]),
    ("fold02", ["S03", "S04", "S13", "S14", "S23", "S24", "S33", "S34"]),
    ("fold03", ["S06", "S07", "S16", "S17", "S26", "S27", "S36", "S37"]),
    ("fold04", ["S08", "S09", "S18", "S19", "S28", "S29", "S38", "S39"]),
]

TRAIN_SUBJECTS_32 = [
    "S01", "S02", "S03", "S04", "S06", "S07", "S08", "S09",
    "S11", "S12", "S13", "S14", "S16", "S17", "S18", "S19",
    "S21", "S22", "S23", "S24", "S26", "S27", "S28", "S29",
    "S31", "S32", "S33", "S34", "S36", "S37", "S38", "S39",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(obj, path):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_best_val(run_dir):
    hist_path = os.path.join(run_dir, "history.json")
    if not os.path.isfile(hist_path):
        return None
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return None
    best = min(history, key=lambda row: float(row.get("mpjpe", 1e9)))
    return {
        "epoch": int(best.get("epoch", -1)),
        "mpjpe": float(best["mpjpe"]) * 1000.0,
        "pa_mpjpe": float(best.get("pa_mpjpe", 0.0)) * 1000.0,
        "pck@20": float(best.get("pck@20", 0.0)),
        "pck@50": float(best.get("pck@50", 0.0)),
    }


def subject_csv(subjects):
    return ",".join(subjects)


def build_cmd(args, run_dir):
    cmd = [
        sys.executable,
        "-u",
        "train_t1_rgb_teacher_mpjpe.py",
        args.dataset_root,
        args.config_file,
        "--split", "teacher_student_split",
        "--epochs", str(args.epochs),
        "--window", str(args.window),
        "--stride", str(args.stride),
        "--batch_size", str(args.batch_size),
        "--val_batch_size", str(args.val_batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--mixup_alpha", str(args.mixup_alpha),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--log_every", str(args.log_every),
        "--skip_final_test",
        "--run_dir", run_dir,
    ]
    if args.temporal_rgb:
        cmd.append("--temporal_rgb")
    return cmd


def run_command(cmd, env):
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
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run S2P2 RGB teacher 4-fold subject-level internal CV.")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--results_root", default=RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--temporal_rgb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.results_root)
    split_info = {
        "protocol": "protocol2",
        "split": "teacher_student_split",
        "official_s2p2_heldout_subjects_not_used": [
            "S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"],
        "folds": [],
    }

    print("=" * 80)
    print("S2P2 RGB teacher 4-fold internal CV")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results root: {args.results_root}")
    print("Official held-out subjects are not used by teacher CV.")
    print("=" * 80)

    summaries = []
    for fold_name, val_subjects in FOLDS:
        train_subjects = [s for s in TRAIN_SUBJECTS_32 if s not in set(val_subjects)]
        run_dir = os.path.join(args.results_root, fold_name)
        ensure_dir(run_dir)

        fold_info = {
            "fold": fold_name,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "run_dir": run_dir,
        }
        split_info["folds"].append(fold_info)
        write_json(fold_info, os.path.join(run_dir, "fold_info.json"))

        metric = read_best_val(run_dir)
        if metric is not None and not args.force:
            print(f"[{fold_name}] skip existing MPJPE={metric['mpjpe']:.2f}mm")
            summaries.append({**fold_info, **metric, "status": "skipped_existing"})
            continue

        env = os.environ.copy()
        env["MMFI_PROTOCOL"] = "protocol2"
        env["HELDOUT_SPLIT_ENABLED"] = "0"
        env["TEACHER_CV_TRAIN_SUBJECTS"] = subject_csv(train_subjects)
        env["TEACHER_CV_VAL_SUBJECTS"] = subject_csv(val_subjects)
        env["TRAIN_SEED"] = str(args.seed)
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = build_cmd(args, run_dir)
        print("\n" + "=" * 80)
        print(f"[{fold_name}] val subjects: {subject_csv(val_subjects)}")
        print(f"[{fold_name}] train subjects: {subject_csv(train_subjects)}")
        print(" ".join(cmd))
        print("=" * 80)

        if args.dry_run:
            continue

        returncode = run_command(cmd, env)
        metric = read_best_val(run_dir)
        if metric is None:
            summaries.append({**fold_info, "returncode": returncode, "status": "failed_or_incomplete"})
            print(f"[{fold_name}] no validation metric, returncode={returncode}")
        else:
            summaries.append({**fold_info, **metric, "returncode": returncode})
            print(f"[{fold_name}] best MPJPE={metric['mpjpe']:.2f}mm "
                  f"PA={metric['pa_mpjpe']:.2f}mm epoch={metric['epoch']}")

        write_json(summaries, os.path.join(args.results_root, "summary_all.json"))

    write_json(split_info, os.path.join(args.results_root, "cv_split_info.json"))
    ranked = sorted([s for s in summaries if "mpjpe" in s], key=lambda row: row["mpjpe"])
    write_json(ranked, os.path.join(args.results_root, "summary_ranked.json"))

    if ranked:
        mean_mpjpe = sum(item["mpjpe"] for item in ranked) / len(ranked)
        print("\nTeacher CV summary:")
        for item in ranked:
            print(f"{item['fold']}: {item['mpjpe']:.2f}mm | "
                  f"PA {item['pa_mpjpe']:.2f}mm | epoch {item['epoch']}")
        print(f"Mean MPJPE across completed folds: {mean_mpjpe:.2f}mm")
    print(f"\nSaved CV summaries under: {args.results_root}")


if __name__ == "__main__":
    main()
