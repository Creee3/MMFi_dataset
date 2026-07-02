"""
run_temporal_mixup_cmc_comparison.py

对比两组结果：
1. 时序 WiFi + mixup（baseline）
2. 时序 WiFi + mixup + CMC（训练时使用 RGB 2D pose 作为特权信息，不做 distillation）

设计目标：
- 在 run_temporal_only_cmc_comparison.py 的基础上仅启用 mixup
- 统一保存到 experiment_results/temporal_mixup_cmc_comparison/
- 每个 variant 支持多 seed 运行
- 自动把 strict_offline_runs 下最新一次训练目录移动到标准结果目录
- 自动汇总 summary.json

用法：
  python run_temporal_mixup_cmc_comparison.py <dataset_root> <config_file>
例如：
  python run_temporal_mixup_cmc_comparison.py data_base config.yaml
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
import re
import time
import numpy as np


SEEDS = list(range(10))
RESULTS_ROOT = "experiment_results/temporal_mixup_cmc_comparison_lr1e-3_bs32"

COMMON_ARGS = [
    "--split", "cross_subject_split",
    "--epochs", "30",
    "--window", "32", "--stride", "2",
    "--dropout", "0.15",
    "--aug_noise", "0.0", "--aug_freq_mask", "0.0", "--aug_time_mask", "0.0",
    "--mixup_alpha", "0.4",
    "--no_cbam",
    "--lr", "1e-3",
    "--lr_patience", "3", "--patience", "0",
    "--warmup_epochs", "0",
    "--weight_decay", "1e-3",
    "--batch_size", "32",
    "--val_batch_size", "32",
    "--transformer_layers", "2", "--dim_feedforward", "1024",
    "--no_root_relative",
    "--device", "cuda",
]

VARIANTS = {
    "A_temporal_wifi_mixup": {
        "label": "Temporal WiFi + Mixup",
        "entry": "train_baseline.py",
        "prefix": "temporal_baseline_cbam",
        "extra_args": [],
    },
    "B_temporal_wifi_mixup_cmc": {
        "label": "Temporal WiFi + Mixup + CMC",
        "entry": "train_lupi_rgb_teacher.py",
        "prefix": "temporal_lupi_rgb_teacher_cmc_cbam_distill",
        "extra_args": [
            "--teacher_ckpt", r"strict_offline_runs/T1_rgb_only_teacher/best.pth",
            "--lambda_cmc", "0.2",
            "--lambda_distill", "0.0",
            "--cmc_stopgrad_rgb",
            "--swa_start", "0",
        ],
    },
}


def latest_run_dir(prefix: str):
    runs_root = "strict_offline_runs"
    if not os.path.isdir(runs_root):
        return None
    dirs = [
        d for d in os.listdir(runs_root)
        if d.startswith(prefix) and os.path.isdir(os.path.join(runs_root, d))
    ]
    if not dirs:
        return None
    dirs = sorted(dirs)
    return os.path.join(runs_root, dirs[-1])



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def run_one(vname: str, variant: dict, seed: int, dataset_root: str, config_file: str):
    dst = os.path.join(RESULTS_ROOT, vname, f"seed_{seed:02d}")
    hist_path = os.path.join(dst, "history.json")
    if os.path.isfile(hist_path):
        print(f"  ⏭  Already done: {dst}, skipping.")
        return True

    cmd = [
        sys.executable,
        "-u",  # 关键:禁用 Python stdout 缓冲,保证实时解析
        variant["entry"],
        dataset_root,
        config_file,
    ] + COMMON_ARGS + variant.get("extra_args", [])

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n{'=' * 64}")
    print(f"▶ {vname} | seed={seed}")
    print(f"  label : {variant['label']}")
    print(f"  entry : {variant['entry']}")
    print(f"{'=' * 64}")

    # 启动前快照 prefix 匹配的目录(给 fallback 用)
    runs_root = "strict_offline_runs"
    existing_dirs = set()
    if os.path.isdir(runs_root):
        existing_dirs = {
            d for d in os.listdir(runs_root)
            if d.startswith(variant["prefix"])
            and os.path.isdir(os.path.join(runs_root, d))
        }

    # 边实时打印边解析 stdout,抓 "Run dir: <path>"
    run_dir_from_stdout = None
    run_dir_pattern = re.compile(r"Run dir:\s*(\S+)")

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
            m = run_dir_pattern.search(line)
            if m:
                run_dir_from_stdout = m.group(1).strip().rstrip(",.")
    finally:
        proc.wait()

    # 优先用 stdout 解析到的真实路径
    src = None
    if run_dir_from_stdout and os.path.isdir(run_dir_from_stdout):
        src = run_dir_from_stdout
        print(f"  ✓ Detected run dir from stdout: {src}")

    # Fallback:启动前后的目录差集
    if src is None and os.path.isdir(runs_root):
        current_dirs = {
            d for d in os.listdir(runs_root)
            if d.startswith(variant["prefix"])
            and os.path.isdir(os.path.join(runs_root, d))
        }
        new_dirs = sorted(current_dirs - existing_dirs)
        if len(new_dirs) == 1:
            src = os.path.join(runs_root, new_dirs[0])
            print(f"  ✓ Fallback: single new run dir -> {src}")
        elif len(new_dirs) > 1:
            src = os.path.join(runs_root, new_dirs[0])
            print(f"  ⚠️ Fallback: {len(new_dirs)} new dirs, picking earliest -> {src}")
        else:
            print(f"  ⚠️ Fallback: no new run dir for prefix '{variant['prefix']}'")

    # 移动(Windows 上偶发文件占用,加重试)
    if src and os.path.isdir(src):
        ensure_dir(os.path.dirname(dst))
        if os.path.isdir(dst):
            shutil.rmtree(dst)

        moved = False
        last_err = None
        for attempt in range(5):
            try:
                shutil.move(src, dst)
                moved = True
                break
            except PermissionError as e:
                last_err = e
                print(f"  ⚠️ Move attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        if moved:
            print(f"  📁 Moved {src} -> {dst}")
        else:
            print(f"  ❌ Move failed after retries: {last_err}")
            return False
    else:
        print(f"  ⚠️ No valid run dir to move (prefix: {variant['prefix']})")

    return proc.returncode == 0



def collect_results():
    summary_path = os.path.join(RESULTS_ROOT, "summary.json")
    all_results = {}

    order = [
        "A_temporal_wifi_mixup",
        "B_temporal_wifi_mixup_cmc",
    ]

    for vname in order:
        variant = VARIANTS[vname]
        pa_list, mpjpe_list, pck20_list, pck50_list = [], [], [], []

        for seed in SEEDS:
            hist_path = os.path.join(RESULTS_ROOT, vname, f"seed_{seed:02d}", "history.json")
            if not os.path.isfile(hist_path):
                print(f"  ⚠️ Missing: {hist_path}")
                continue

            with open(hist_path, "r", encoding="utf-8") as f:
                h = json.load(f)

            best_ep = min(h, key=lambda e: e["pa_mpjpe"])
            pa_list.append(best_ep["pa_mpjpe"] * 1000.0)
            mpjpe_list.append(best_ep.get("mpjpe", 0.0) * 1000.0)
            pck20_list.append(best_ep.get("pck@20", 0.0))
            pck50_list.append(best_ep.get("pck@50", 0.0))

        if pa_list:
            all_results[vname] = {
                "n": len(pa_list),
                "label": variant["label"],
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
    print("\n" + "=" * 120)
    print("📊 Temporal WiFi + Mixup  vs  Temporal WiFi + Mixup + CMC")
    print("   cross_subject_split / Protocol 2 / no root-relative / n=10")
    print("=" * 120)
    print(
        f"{'Variant':<30s} {'Description':<32s} {'PA-MPJPE (mm)':<20s} "
        f"{'MPJPE (mm)':<20s} {'PCK@20 (%)':<15s} {'PCK@50 (%)':<15s}"
    )
    print("-" * 120)

    for vname in order:
        if vname not in all_results:
            print(f"{vname:<30s} {'[missing]':<32s}")
            continue
        r = all_results[vname]
        print(
            f"{vname:<30s} {r['label']:<32s} {r['PA-MPJPE']:<20s} "
            f"{r['MPJPE']:<20s} {r['PCK@20']:<15s} {r['PCK@50']:<15s}"
        )

    print("=" * 120)

    if all(k in all_results for k in order):
        base = all_results["A_temporal_wifi_mixup"]["pa_mpjpe_mean"]
        cmc = all_results["B_temporal_wifi_mixup_cmc"]["pa_mpjpe_mean"]
        improve = base - cmc
        improve_pct = improve / base * 100.0 if base > 0 else 0.0
        tag = "improvement" if improve >= 0 else "drop"
        print(
            f"\n📌 +CMC vs Baseline: PA-MPJPE {'↓' if improve >= 0 else '↑'}{abs(improve):.2f}mm "
            f"({abs(improve_pct):.1f}% {tag})"
        )

    return all_results



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "baseline", "cmc"],
        help="Which experiment to run: all / baseline / cmc"
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    config_file = args.config_file
    ensure_dir(RESULTS_ROOT)

    print("=" * 72)
    print("🔬 Temporal WiFi + Mixup  vs  Temporal WiFi + Mixup + CMC")
    print(f"   Split:        cross_subject_split")
    print(f"   Epochs:       30")
    print(f"   Window:       32")
    print(f"   Stride:       2")
    print(f"   Mixup alpha:  0.4")
    print(f"   CBAM:         OFF")
    print(f"   Seeds:        {SEEDS}")
    print(f"   Device arg:   cuda")
    print(f"   Results dir:  {RESULTS_ROOT}")
    print(f"   Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    failed = []
    if args.variant == "baseline":
        order = ["A_temporal_wifi_mixup"]
    elif args.variant == "cmc":
        order = ["B_temporal_wifi_mixup_cmc"]
    else:
        order = [
            "A_temporal_wifi_mixup",
            "B_temporal_wifi_mixup_cmc",
        ]

    for vname in order:
        variant = VARIANTS[vname]
        print(f"\n{'🔷' * 30}")
        print(f"📌 Running {vname} ({variant['label']}) × {len(SEEDS)} seeds")
        print(f"{'🔷' * 30}")
        for seed in SEEDS:
            ok = run_one(vname, variant, seed, dataset_root, config_file)
            if not ok:
                failed.append((vname, seed))

    print(f"\n✅ All done. Failed: {failed if failed else 'none'}")
    collect_results()


if __name__ == "__main__":
    main()
