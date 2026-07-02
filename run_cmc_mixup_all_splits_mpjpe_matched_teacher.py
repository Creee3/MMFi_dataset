"""
run_cmc_mixup_all_splits_mpjpe_matched_teacher.py

CMC + Mixup 在 3×3 Protocol × Split 设置上的匹配 Teacher 实验：
每个 protocol × split 组合先训练 1 个对应 Teacher，
再使用该 Teacher 跑 2 个 student seeds。

Protocol:
  protocol1 — 仅日常活动
  protocol2 — 仅康复活动
  protocol3 — 全部活动

Split:
  random_split        — 80/20 随机划分
  cross_scene_split   — 跨场景
  cross_subject_split — 跨个体
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import yaml


SEEDS = [0, 1]
TEACHER_SEED = 0
RESULTS_ROOT = "experiment_results/cmc_mixup_all_splits_mpjpe_matched_teacher"
TEACHER_ROOT = "strict_offline_runs/teachers_3x3_mpjpe"
TEACHER_PATIENCE = 4

PROTOCOLS = ["protocol1", "protocol2", "protocol3"]
SPLITS = ["random_split", "cross_scene_split", "cross_subject_split"]

PROTOCOL_LABEL = {
    "protocol1": "Daily",
    "protocol2": "Rehab",
    "protocol3": "All",
}
SPLIT_LABEL = {
    "random_split": "Random",
    "cross_scene_split": "Cross-Scene",
    "cross_subject_split": "Cross-Subject",
}

# Student 最优超参数（来自 Bayesian HPO mixup 模式）
BEST_LR = 1.03e-04
BEST_BS = 64

STUDENT_COMMON_ARGS = [
    "--epochs", "30",
    "--window", "32", "--stride", "2",
    "--dropout", "0.15",
    "--aug_noise", "0.0", "--aug_freq_mask", "0.0", "--aug_time_mask", "0.0",
    "--mixup_alpha", "0.4",
    "--no_cbam",
    "--lr", str(BEST_LR),
    "--lr_patience", "3", "--patience", "0",
    "--warmup_epochs", "0",
    "--weight_decay", "1e-3",
    "--batch_size", str(BEST_BS),
    "--val_batch_size", "32",
    "--transformer_layers", "2", "--dim_feedforward", "1024",
    "--no_root_relative",
    "--device", "cuda",
    "--lambda_cmc", "0.2",
    "--lambda_distill", "0.0",
    "--cmc_stopgrad_rgb",
    "--swa_start", "0",
]

TEACHER_ENTRY = "train_t1_rgb_teacher_mpjpe.py"
STUDENT_ENTRY = "train_lupi_rgb_teacher_mpjpe.py"
STUDENT_PREFIX = "temporal_lupi_rgb_teacher_cmc_cbam_distill"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_abs(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def make_temp_config(original_config_file: str, protocol: str) -> str:
    with open(original_config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["protocol"] = protocol

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return tmp.name


def run_teacher(protocol: str, split: str, dataset_root: str, config_file: str):
    teacher_dir = norm_abs(os.path.join(TEACHER_ROOT, f"{protocol}_{split}"))
    best_path = os.path.join(teacher_dir, "best.pth")
    hist_path = os.path.join(teacher_dir, "history.json")
    if os.path.isfile(best_path) and os.path.isfile(hist_path):
        print(f"  ⏭  Teacher exists: {teacher_dir}, skipping.")
        return teacher_dir, True

    ensure_dir(teacher_dir)
    tmp_config = make_temp_config(config_file, protocol)
    cmd = [
        sys.executable, "-u", TEACHER_ENTRY,
        dataset_root, tmp_config,
        "--split", split,
        "--patience", str(TEACHER_PATIENCE),
        "--run_dir", teacher_dir,
    ]

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(TEACHER_SEED)
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n{'=' * 64}")
    print(f"🎓 Teacher | {protocol} × {split}")
    print(f"  protocol : {PROTOCOL_LABEL[protocol]}")
    print(f"  split    : {SPLIT_LABEL[split]}")
    print(f"  seed     : {TEACHER_SEED}")
    print(f"  run dir  : {teacher_dir}")
    print(f"{'=' * 64}")

    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
            encoding="utf-8", errors="replace",
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        finally:
            proc.wait()
    finally:
        os.unlink(tmp_config)

    ok = proc.returncode == 0 and os.path.isfile(best_path)
    if not ok:
        print(f"  ❌ Teacher training failed for {protocol} × {split}")
    return teacher_dir, ok


def run_student(protocol: str, split: str, seed: int,
                dataset_root: str, config_file: str, teacher_ckpt: str):
    variant_dir = f"{protocol}_{split}"
    dst = norm_abs(os.path.join(RESULTS_ROOT, variant_dir, f"seed_{seed:02d}"))
    hist_path = os.path.join(dst, "history.json")
    if os.path.isfile(hist_path):
        print(f"  ⏭  Already done: {dst}, skipping.")
        return True

    tmp_config = make_temp_config(config_file, protocol)
    cmd = [
        sys.executable, "-u", STUDENT_ENTRY,
        dataset_root, tmp_config,
        "--split", split,
        "--teacher_ckpt", norm_abs(teacher_ckpt),
        "--run_dir", dst,
    ] + STUDENT_COMMON_ARGS

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n{'=' * 64}")
    print(f"▶ Student | {protocol} × {split} | seed={seed}")
    print(f"  protocol : {PROTOCOL_LABEL[protocol]}")
    print(f"  split    : {SPLIT_LABEL[split]}")
    print(f"  teacher  : {teacher_ckpt}")
    print(f"  optimize : MPJPE")
    print(f"{'=' * 64}")

    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
            encoding="utf-8", errors="replace",
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        finally:
            proc.wait()
    finally:
        os.unlink(tmp_config)
    if proc.returncode != 0:
        print(f"  ❌ Student training failed: returncode={proc.returncode}")
        return False
    if not os.path.isfile(hist_path):
        print(f"  ❌ history.json not found after training: {hist_path}")
        return False
    print(f"  📁 Student result saved in place: {dst}")
    return True


def collect_results(protocols, splits):
    summary_path = os.path.join(RESULTS_ROOT, "summary.json")
    all_results = {}

    for protocol in protocols:
        for split in splits:
            variant_dir = f"{protocol}_{split}"
            pa_list, mpjpe_list, pck20_list, pck50_list = [], [], [], []

            for seed in SEEDS:
                hist_path = os.path.join(
                    RESULTS_ROOT, variant_dir, f"seed_{seed:02d}", "history.json")
                if not os.path.isfile(hist_path):
                    continue

                with open(hist_path, "r", encoding="utf-8") as f:
                    h = json.load(f)

                best_ep = min(h, key=lambda e: e["mpjpe"])
                pa_list.append(best_ep.get("pa_mpjpe", 0.0) * 1000.0)
                mpjpe_list.append(best_ep["mpjpe"] * 1000.0)
                pck20_list.append(best_ep.get("pck@20", 0.0))
                pck50_list.append(best_ep.get("pck@50", 0.0))

            if mpjpe_list:
                key = variant_dir
                all_results[key] = {
                    "n": len(mpjpe_list),
                    "protocol": PROTOCOL_LABEL[protocol],
                    "split": SPLIT_LABEL[split],
                    "teacher": norm_abs(os.path.join(TEACHER_ROOT, variant_dir, "best.pth")),
                    "optimize": "MPJPE",
                    "PA-MPJPE": f"{np.mean(pa_list):.2f} ± {np.std(pa_list):.2f}",
                    "MPJPE": f"{np.mean(mpjpe_list):.2f} ± {np.std(mpjpe_list):.2f}",
                    "PCK@20": f"{np.mean(pck20_list):.2f} ± {np.std(pck20_list):.2f}",
                    "PCK@50": f"{np.mean(pck50_list):.2f} ± {np.std(pck50_list):.2f}",
                    "pa_mpjpe_mean": float(np.mean(pa_list)),
                    "mpjpe_mean": float(np.mean(mpjpe_list)),
                    "pck20_mean": float(np.mean(pck20_list)),
                    "pck50_mean": float(np.mean(pck50_list)),
                    "pa_mpjpe_std": float(np.std(pa_list)),
                    "mpjpe_std": float(np.std(mpjpe_list)),
                    "pck20_std": float(np.std(pck20_list)),
                    "pck50_std": float(np.std(pck50_list)),
                }

    ensure_dir(RESULTS_ROOT)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Summary -> {summary_path}")
    print("\n" + "=" * 130)
    print("📊 CMC + Mixup 跨 Protocol × Split 对比（Matched Teacher, MPJPE 优化版）")
    print(f"   Student hparams: lr={BEST_LR:.2e}  bs={BEST_BS}  no CBAM")
    print(f"   Teacher config:  one matched teacher per protocol×split, shared by student seeds")
    print(f"   Teacher patience:{TEACHER_PATIENCE}")
    print(f"   Student seeds:   {SEEDS}  (n={len(SEEDS)})")
    print("=" * 130)
    header = (
        f"{'Protocol':<12s} {'Split':<20s} "
        f"{'MPJPE (mm)':<20s} {'PA-MPJPE (mm)':<20s} "
        f"{'PCK@20 (%)':<15s} {'PCK@50 (%)':<15s}"
    )
    print(header)
    print("-" * 130)

    for protocol in protocols:
        for split in splits:
            key = f"{protocol}_{split}"
            if key not in all_results:
                print(f"{PROTOCOL_LABEL[protocol]:<12s} {SPLIT_LABEL[split]:<20s} [missing]")
                continue
            r = all_results[key]
            print(
                f"{r['protocol']:<12s} {r['split']:<20s} "
                f"{r['MPJPE']:<20s} {r['PA-MPJPE']:<20s} "
                f"{r['PCK@20']:<15s} {r['PCK@50']:<15s}"
            )

    print("=" * 130)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CMC+Mixup across all Protocol×Split combinations with matched teachers")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        choices=PROTOCOLS,
        help="Run only one protocol (default: all 3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        dest="split_filter",
        choices=SPLITS,
        help="Run only one split (default: all 3)",
    )
    args_cli = parser.parse_args()

    dataset_root = args_cli.dataset_root
    config_file = args_cli.config_file
    ensure_dir(RESULTS_ROOT)
    ensure_dir(TEACHER_ROOT)

    protocols = [args_cli.protocol] if args_cli.protocol else PROTOCOLS
    splits = [args_cli.split_filter] if args_cli.split_filter else SPLITS

    n_combos = len(protocols) * len(splits)
    n_total_students = n_combos * len(SEEDS)

    print("=" * 76)
    print("🔬 CMC + Mixup — 3×3 Matched-Teacher Experiment (MPJPE optimized)")
    print(f"   Student params: lr={BEST_LR:.2e}  bs={BEST_BS}  no CBAM")
    print(f"   Teacher entry:  {TEACHER_ENTRY}")
    print(f"   Student entry:  {STUDENT_ENTRY}")
    print(f"   Protocols:      {[PROTOCOL_LABEL[p] for p in protocols]}")
    print(f"   Splits:         {[SPLIT_LABEL[s] for s in splits]}")
    print(f"   Teacher seed:   {TEACHER_SEED}  (1 teacher per combo)")
    print(f"   Teacher ES:     patience={TEACHER_PATIENCE}")
    print(f"   Student seeds:  {SEEDS}  (n={len(SEEDS)})")
    print(f"   Total:          {n_combos} teachers + {n_total_students} student runs")
    print(f"   Results dir:    {RESULTS_ROOT}")
    print(f"   Teacher root:   {TEACHER_ROOT}")
    print(f"   Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 76)

    failed_teachers = []
    failed_students = []

    for protocol in protocols:
        for split in splits:
            print(f"\n{'🔷' * 30}")
            print(f"📌 {PROTOCOL_LABEL[protocol]} × {SPLIT_LABEL[split]}")
            print(f"{'🔷' * 30}")

            teacher_dir, teacher_ok = run_teacher(
                protocol, split, dataset_root, config_file
            )
            if not teacher_ok:
                failed_teachers.append((protocol, split))
                continue

            teacher_ckpt = os.path.join(teacher_dir, "best.pth")
            for seed in SEEDS:
                ok = run_student(
                    protocol, split, seed,
                    dataset_root, config_file, teacher_ckpt
                )
                if not ok:
                    failed_students.append((protocol, split, seed))

    print(f"\n✅ All done.")
    print(f"   Failed teachers: {failed_teachers if failed_teachers else 'none'}")
    print(f"   Failed students: {failed_students if failed_students else 'none'}")
    collect_results(protocols, splits)


if __name__ == "__main__":
    main()
