"""
run_ablation_mpjpe.py

消融实验（MPJPE 优化版）：基于 MPJPE Bayesian HPO 的 mixup 模式最优参数
对比四种配置：
  1. 纯时序 (WiFi Baseline)
  2. 时序 + Mixup
  3. 时序 + CMC
  4. 时序 + Mixup + CMC

优化目标: MPJPE（scheduler + best ckpt 均按 MPJPE 选取）
划分: cross_subject_split / protocol2

每个 variant 跑 5 seeds × 30 epochs。
结果保存到 experiment_results/ablation_mpjpe/

用法：
  python run_ablation_mpjpe.py <dataset_root> <config_file>
  python run_ablation_mpjpe.py data_base config.yaml --variant cmc
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


SEEDS = list(range(5))
RESULTS_ROOT = "experiment_results/ablation_mpjpe"
BAYES_MIXUP_SUMMARY = "experiment_results/bayesian_hparam_tuning_mpjpe/mixup/bo_summary.json"

# ── MPJPE HPO 最优超参数（优先读取 mixup/bo_summary.json，找不到时回退） ──
FALLBACK_LR = 1.03e-04
FALLBACK_BS = 64


def load_best_hparams():
    best_lr = FALLBACK_LR
    best_bs = FALLBACK_BS

    if os.path.isfile(BAYES_MIXUP_SUMMARY):
        try:
            with open(BAYES_MIXUP_SUMMARY, "r", encoding="utf-8") as f:
                summary = json.load(f)
            best_lr = float(summary["best_lr"])
            best_bs = int(summary["best_bs"])
            print(f"✅ Loaded tuned hparams from {BAYES_MIXUP_SUMMARY}")
            print(f"   Using mixup best_lr={best_lr:.6g}, best_bs={best_bs}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ Failed to parse {BAYES_MIXUP_SUMMARY}: {e}")
            print(f"   Falling back to lr={FALLBACK_LR:.6g}, bs={FALLBACK_BS}")
    else:
        print(f"⚠️ {BAYES_MIXUP_SUMMARY} not found")
        print(f"   Falling back to lr={FALLBACK_LR:.6g}, bs={FALLBACK_BS}")

    return best_lr, best_bs


BEST_LR, BEST_BS = load_best_hparams()

COMMON_ARGS = [
    "--split", "cross_subject_split",
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
]

VARIANTS = {
    "A_temporal_only": {
        "label": "WiFi Baseline (no Mixup)",
        "entry": "train_baseline_mpjpe.py",
        "prefix": "temporal_baseline_cbam",
        "extra_args": [
            "--mixup_alpha", "0.0",
        ],
    },
    "B_temporal_mixup": {
        "label": "WiFi + Mixup",
        "entry": "train_baseline_mpjpe.py",
        "prefix": "temporal_baseline_cbam",
        "extra_args": [],
    },
    "C_temporal_cmc": {
        "label": "WiFi + CMC (no Mixup)",
        "entry": "train_lupi_rgb_teacher_mpjpe.py",
        "prefix": "temporal_lupi_rgb_teacher_cmc_cbam_distill",
        "extra_args": [
            "--teacher_ckpt", r"strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth",
            "--lambda_cmc", "0.2",
            "--lambda_distill", "0.0",
            "--cmc_stopgrad_rgb",
            "--swa_start", "0",
            "--mixup_alpha", "0.0",
        ],
    },
    "D_temporal_mixup_cmc": {
        "label": "WiFi + Mixup + CMC",
        "entry": "train_lupi_rgb_teacher_mpjpe.py",
        "prefix": "temporal_lupi_rgb_teacher_cmc_cbam_distill",
        "extra_args": [
            "--teacher_ckpt", r"strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth",
            "--lambda_cmc", "0.2",
            "--lambda_distill", "0.0",
            "--cmc_stopgrad_rgb",
            "--swa_start", "0",
        ],
    },
}


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
        "-u",
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
    print(f"  optimize : MPJPE")
    print(f"{'=' * 64}")

    runs_root = "strict_offline_runs"
    existing_dirs = set()
    if os.path.isdir(runs_root):
        existing_dirs = {
            d for d in os.listdir(runs_root)
            if d.startswith(variant["prefix"])
            and os.path.isdir(os.path.join(runs_root, d))
        }

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

    src = None
    if run_dir_from_stdout and os.path.isdir(run_dir_from_stdout):
        src = run_dir_from_stdout
        print(f"  ✓ Detected run dir from stdout: {src}")

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
        "A_temporal_only",
        "B_temporal_mixup",
        "C_temporal_cmc",
        "D_temporal_mixup_cmc",
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

            best_ep = min(h, key=lambda e: e["mpjpe"])
            pa_list.append(best_ep.get("pa_mpjpe", 0.0) * 1000.0)
            mpjpe_list.append(best_ep["mpjpe"] * 1000.0)
            pck20_list.append(best_ep.get("pck@20", 0.0))
            pck50_list.append(best_ep.get("pck@50", 0.0))

        if mpjpe_list:
            all_results[vname] = {
                "n": len(mpjpe_list),
                "label": variant["label"],
                "optimize": "MPJPE",
                "MPJPE": f"{np.mean(mpjpe_list):.2f} ± {np.std(mpjpe_list):.2f}",
                "PA-MPJPE": f"{np.mean(pa_list):.2f} ± {np.std(pa_list):.2f}",
                "PCK@20": f"{np.mean(pck20_list):.2f} ± {np.std(pck20_list):.2f}",
                "PCK@50": f"{np.mean(pck50_list):.2f} ± {np.std(pck50_list):.2f}",
                "mpjpe_mean": float(np.mean(mpjpe_list)),
                "pa_mpjpe_mean": float(np.mean(pa_list)),
                "pck20_mean": float(np.mean(pck20_list)),
                "pck50_mean": float(np.mean(pck50_list)),
                "mpjpe_std": float(np.std(mpjpe_list)),
                "pa_mpjpe_std": float(np.std(pa_list)),
                "pck20_std": float(np.std(pck20_list)),
                "pck50_std": float(np.std(pck50_list)),
            }

    ensure_dir(RESULTS_ROOT)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Summary -> {summary_path}")
    print("\n" + "=" * 130)
    print(f"📊 Ablation Study (MPJPE Optimized)  |  lr={BEST_LR:.2e}  bs={BEST_BS}  no CBAM")
    print(f"   cross_subject_split / Protocol 2 / no root-relative / n={len(SEEDS)}")
    print("=" * 130)
    print(
        f"{'Variant':<26s} {'Description':<32s} {'MPJPE (mm)':<20s} "
        f"{'PA-MPJPE (mm)':<20s} {'PCK@20 (%)':<15s} {'PCK@50 (%)':<15s}"
    )
    print("-" * 130)

    for vname in order:
        if vname not in all_results:
            print(f"{vname:<26s} {'[missing]':<32s}")
            continue
        r = all_results[vname]
        print(
            f"{vname:<26s} {r['label']:<32s} {r['MPJPE']:<20s} "
            f"{r['PA-MPJPE']:<20s} {r['PCK@20']:<15s} {r['PCK@50']:<15s}"
        )

    print("=" * 130)

    # ── 消融对比分析（MPJPE） ──
    if all(k in all_results for k in order):
        A = all_results["A_temporal_only"]["mpjpe_mean"]
        B = all_results["B_temporal_mixup"]["mpjpe_mean"]
        C = all_results["C_temporal_cmc"]["mpjpe_mean"]
        D = all_results["D_temporal_mixup_cmc"]["mpjpe_mean"]

        print(f"\n📌 消融分析 (MPJPE):")
        print(f"   纯时序 (baseline):        {A:.2f} mm")
        print(f"   + Mixup:                  {B:.2f} mm  (Δ = {B - A:+.2f} mm)")
        print(f"   + CMC:                    {C:.2f} mm  (Δ = {C - A:+.2f} mm)")
        print(f"   + Mixup + CMC:            {D:.2f} mm  (Δ = {D - A:+.2f} mm)")

        mixup_effect_no_cmc = B - A
        mixup_effect_with_cmc = D - C
        print(f"\n   Mixup 边际效应 (no CMC):  {mixup_effect_no_cmc:+.2f} mm")
        print(f"   Mixup 边际效应 (with CMC): {mixup_effect_with_cmc:+.2f} mm")

        cmc_effect_no_mixup = C - A
        cmc_effect_with_mixup = D - B
        print(f"   CMC 边际效应 (no Mixup):   {cmc_effect_no_mixup:+.2f} mm")
        print(f"   CMC 边际效应 (with Mixup): {cmc_effect_with_mixup:+.2f} mm")

        # PA-MPJPE 也顺手打印
        A_pa = all_results["A_temporal_only"]["pa_mpjpe_mean"]
        B_pa = all_results["B_temporal_mixup"]["pa_mpjpe_mean"]
        C_pa = all_results["C_temporal_cmc"]["pa_mpjpe_mean"]
        D_pa = all_results["D_temporal_mixup_cmc"]["pa_mpjpe_mean"]
        print(f"\n📌 消融分析 (PA-MPJPE 参考):")
        print(f"   纯时序 (baseline):        {A_pa:.2f} mm")
        print(f"   + Mixup:                  {B_pa:.2f} mm  (Δ = {B_pa - A_pa:+.2f} mm)")
        print(f"   + CMC:                    {C_pa:.2f} mm  (Δ = {C_pa - A_pa:+.2f} mm)")
        print(f"   + Mixup + CMC:            {D_pa:.2f} mm  (Δ = {D_pa - A_pa:+.2f} mm)")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ablation Study (MPJPE Optimized)")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "temporal_only", "mixup", "cmc", "mixup_cmc"],
        help="Which variant to run (default: all)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    config_file = args.config_file
    ensure_dir(RESULTS_ROOT)

    print("=" * 72)
    print("🔬 Ablation Study — MPJPE Optimized (Best Hparams from MPJPE-HPO)")
    print(f"   Best params:  lr={BEST_LR:.2e}  bs={BEST_BS}  no CBAM")
    print(f"   Split:        cross_subject_split  |  Protocol: 2")
    print(f"   Epochs:       30")
    print(f"   Window:       32  |  Stride: 2")
    print(f"   Mixup alpha:  0.4 (variants B, D only)")
    print(f"   CBAM:         OFF")
    print(f"   CMC:          λ=0.2  τ=0.1  (variants C, D only)")
    print(f"   Distill:      OFF (λ=0.0)")
    print(f"   Optimize:     MPJPE")
    print(f"   Seeds:        {SEEDS}  (n={len(SEEDS)})")
    print(f"   Results dir:  {RESULTS_ROOT}")
    print(f"   Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    variant_map = {
        "temporal_only": ["A_temporal_only"],
        "mixup":         ["B_temporal_mixup"],
        "cmc":           ["C_temporal_cmc"],
        "mixup_cmc":     ["D_temporal_mixup_cmc"],
    }
    if args.variant == "all":
        order = [
            "A_temporal_only",
            "B_temporal_mixup",
            "C_temporal_cmc",
            "D_temporal_mixup_cmc",
        ]
    else:
        order = variant_map[args.variant]

    failed = []
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
