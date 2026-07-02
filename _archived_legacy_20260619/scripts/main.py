# main.py
# =============================================================================
# 一键训练入口：支持所有训练模式 + 消融实验 + 主实验
# =============================================================================

import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime

import numpy as np
import torch

from common import (
    Logger, make_run_dir, save_json, fmt_metrics,
    load_config, get_loaders, add_common_args, NumpyEncoder
)
from train_baseline import train_baseline
from train_teacher import train_teacher
from train_lupi import train_lupi_cmc


# =====================================================================
#  单次训练模式 (baseline / teacher / lupi / auto_cmc / auto_teacher_cmc)
# =====================================================================

def run_single_mode(args, device):
    cfg = load_config(args.config_file, args.split)
    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.dataset_root, cfg, args.window, args.stride,
        args.batch_size, args.val_batch_size)

    print("=" * 70)
    print(f"🎯 Mode: {args.mode}")
    print(f"📊 Device: {device}")
    print(f"📊 Split: {args.split}")
    print(f"📊 Window: {args.window}, Stride: {args.stride}")
    print(f"📊 Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Root-relative: {args.use_root_relative}")
    print(f"📊 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    # ---- baseline ----
    if args.mode == "baseline":
        metric, run_dir = train_baseline(
            args, train_loader, val_loader, device)
        results["baseline"] = {
            "pa_mpjpe_mm": metric * 1000, "run_dir": run_dir}

    # ---- teacher ----
    elif args.mode == "teacher":
        metric, ckpt, run_dir = train_teacher(
            args, train_loader, val_loader, device)
        results["teacher"] = {
            "pa_mpjpe_mm": metric * 1000, "run_dir": run_dir,
            "checkpoint": ckpt}

    # ---- lupi (需要 teacher_ckpt) ----
    elif args.mode == "lupi":
        if args.teacher_ckpt is None:
            print("❌ mode=lupi 需要指定 --teacher_ckpt")
            sys.exit(1)
        metric, run_dir = train_lupi_cmc(
            args, train_loader, val_loader, device,
            teacher_ckpt=args.teacher_ckpt)
        results["lupi"] = {
            "pa_mpjpe_mm": metric * 1000, "run_dir": run_dir}

    # ---- auto_teacher_cmc (teacher → LUPI) ----
    elif args.mode == "auto_teacher_cmc":
        print("\n" + "=" * 70)
        print("📌 STEP 1/2: Fusion Teacher")
        print("=" * 70)
        t_metric, t_ckpt, t_dir = train_teacher(
            args, train_loader, val_loader, device)
        results["teacher"] = {
            "pa_mpjpe_mm": t_metric * 1000, "run_dir": t_dir,
            "checkpoint": t_ckpt}

        print("\n" + "=" * 70)
        print("📌 STEP 2/2: LUPI + CMC + Distillation")
        print(f"   Using teacher: {t_ckpt}")
        print("=" * 70)
        l_metric, l_dir = train_lupi_cmc(
            args, train_loader, val_loader, device,
            teacher_ckpt=t_ckpt)
        results["lupi"] = {
            "pa_mpjpe_mm": l_metric * 1000, "run_dir": l_dir}

    # ---- auto_cmc (baseline → teacher → LUPI) ----
    elif args.mode == "auto_cmc":
        print("\n" + "=" * 70)
        print("📌 STEP 1/3: WiFi Baseline + CBAM")
        print("=" * 70)
        b_metric, b_dir = train_baseline(
            args, train_loader, val_loader, device)
        results["baseline"] = {
            "pa_mpjpe_mm": b_metric * 1000, "run_dir": b_dir}

        print("\n" + "=" * 70)
        print("📌 STEP 2/3: Fusion Teacher")
        print("=" * 70)
        t_metric, t_ckpt, t_dir = train_teacher(
            args, train_loader, val_loader, device)
        results["teacher"] = {
            "pa_mpjpe_mm": t_metric * 1000, "run_dir": t_dir,
            "checkpoint": t_ckpt}

        print("\n" + "=" * 70)
        print("📌 STEP 3/3: LUPI + CMC + Distillation")
        print(f"   Using teacher: {t_ckpt}")
        print("=" * 70)
        l_metric, l_dir = train_lupi_cmc(
            args, train_loader, val_loader, device,
            teacher_ckpt=t_ckpt)
        results["lupi"] = {
            "pa_mpjpe_mm": l_metric * 1000, "run_dir": l_dir}

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("🏁 FINAL SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:20s}: PA-MPJPE = {r['pa_mpjpe_mm']:.1f} mm  "
              f"| dir = {r['run_dir']}")
    print("=" * 70)

    summary_dir = "strict_offline_runs"
    os.makedirs(summary_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(results, os.path.join(
        summary_dir, f"summary_{args.mode}_{ts}.json"))

    return results


# =====================================================================
#  消融实验 (ablation)
# =====================================================================

ABLATION_VARIANTS = {
    "A_wifi_baseline": {
        "mode_args": ["--mode", "baseline"],
        "prefix": "temporal_baseline",
    },
    "B_distill_only": {
        "mode_args": [
            "--mode", "lupi",
            "--lambda_cmc", "0.0",
            "--lambda_distill", "1.0",
        ],
        "prefix": "temporal_lupi",
    },
    "C_cmc_only": {
        "mode_args": [
            "--mode", "lupi",
            "--lambda_cmc", "0.2",
            "--lambda_distill", "0.0",
        ],
        "prefix": "temporal_lupi",
    },
    "D_full_model": {
        "mode_args": [
            "--mode", "lupi",
            "--lambda_cmc", "0.2",
            "--lambda_distill", "1.0",
        ],
        "prefix": "temporal_lupi",
    },
}


def latest_run_dir(prefix):
    """返回 strict_offline_runs 里最新创建的匹配目录"""
    runs = "strict_offline_runs"
    if not os.path.isdir(runs):
        return None
    dirs = [d for d in os.listdir(runs)
            if d.startswith(prefix) and os.path.isdir(os.path.join(runs, d))]
    if not dirs:
        return None
    return os.path.join(runs, sorted(dirs)[-1])


def build_base_args(args):
    """从 args 构建基础命令行参数列表"""
    return [
        "--split", args.split,
        "--epochs", str(args.epochs),
        "--window", str(args.window),
        "--stride", str(args.stride),
        "--dropout", str(args.dropout),
        "--aug_noise", str(args.aug_noise),
        "--aug_freq_mask", str(args.aug_freq_mask),
        "--aug_time_mask", str(args.aug_time_mask),
        "--mixup_alpha", str(args.mixup_alpha),
        "--lr_patience", str(args.lr_patience),
        "--patience", str(args.ablation_patience),
        "--weight_decay", str(args.weight_decay),
        "--batch_size", str(args.batch_size),
        "--transformer_layers", str(args.transformer_layers),
        "--dim_feedforward", str(args.dim_feedforward),
        "--swa_start", str(args.swa_start),
        "--device", args.device,
    ] + (["--cmc_stopgrad_rgb"] if args.cmc_stopgrad_rgb else []) \
      + (["--no_root_relative"] if not args.use_root_relative else [])


def run_ablation_one(vname, variant, seed, args, base_cmd_args):
    """运行消融实验的单次训练"""
    results_root = os.path.join("experiment_results", "ablation")
    dst = os.path.join(results_root, vname, f"seed_{seed:02d}")

    if os.path.isfile(os.path.join(dst, "history.json")):
        print(f"  ⏭  Already done: {dst}, skipping.")
        return True

    cmd = [
        sys.executable, "main.py",
        args.dataset_root, args.config_file,
    ] + base_cmd_args + variant["mode_args"]

    # 需要 teacher 的变体加上 teacher_ckpt
    if "--mode" in variant["mode_args"]:
        mode_idx = variant["mode_args"].index("--mode")
        mode_val = variant["mode_args"][mode_idx + 1]
        if mode_val == "lupi" and args.teacher_ckpt:
            cmd += ["--teacher_ckpt", args.teacher_ckpt]

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)

    print(f"\n{'=' * 60}")
    print(f"▶ ABLATION: variant={vname} | seed={seed}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, env=env)

    src = latest_run_dir(variant["prefix"])
    if src and os.path.isdir(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"  📁 Moved {src} → {dst}")

    return result.returncode == 0


def collect_experiment_results(results_root, variants_or_splits, label=""):
    """通用结果收集函数"""
    summary_path = os.path.join(results_root, "summary.json")
    all_results = {}

    for name in variants_or_splits:
        pa_list, mpjpe_list, pck20_list, pck50_list = [], [], [], []
        seeds = sorted([
            d for d in os.listdir(os.path.join(results_root, name))
            if d.startswith("seed_")
        ]) if os.path.isdir(os.path.join(results_root, name)) else []

        for seed_dir in seeds:
            hist_path = os.path.join(results_root, name, seed_dir, "history.json")
            if not os.path.isfile(hist_path):
                print(f"  ⚠️  Missing: {hist_path}")
                continue
            with open(hist_path) as f:
                h = json.load(f)
            best_ep = min(h, key=lambda e: e["pa_mpjpe"])
            pa_list.append(best_ep["pa_mpjpe"] * 1000)
            mpjpe_list.append(best_ep.get("mpjpe", 0) * 1000)
            pck20_list.append(best_ep.get("pck@20", 0))
            pck50_list.append(best_ep.get("pck@50", 0))

        if pa_list:
            all_results[name] = {
                "n":        len(pa_list),
                "PA-MPJPE": f"{np.mean(pa_list):.2f} ± {np.std(pa_list):.2f}",
                "MPJPE":    f"{np.mean(mpjpe_list):.2f} ± {np.std(mpjpe_list):.2f}",
                "PCK@20":   f"{np.mean(pck20_list):.2f} ± {np.std(pck20_list):.2f}",
                "PCK@50":   f"{np.mean(pck50_list):.2f} ± {np.std(pck50_list):.2f}",
            }

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 {label} Summary → {summary_path}")
    print("\n" + "=" * 60)
    for name, res in all_results.items():
        print(f"\n{name}  (n={res['n']})")
        for k, v in res.items():
            if k != "n":
                print(f"  {k:12s}: {v}")
    print("=" * 60)

    return all_results


def run_ablation(args):
    """运行完整消融实验"""
    results_root = os.path.join("experiment_results", "ablation")
    seeds = list(range(args.num_seeds))
    base_cmd_args = build_base_args(args)
    failed = []

    print("=" * 70)
    print(f"🔬 ABLATION EXPERIMENT")
    print(f"   Variants: {list(ABLATION_VARIANTS.keys())}")
    print(f"   Seeds: {seeds}")
    print(f"   Total runs: {len(ABLATION_VARIANTS) * len(seeds)}")
    if args.teacher_ckpt:
        print(f"   Teacher: {args.teacher_ckpt}")
    print("=" * 70)

    for vname, variant in ABLATION_VARIANTS.items():
        for seed in seeds:
            ok = run_ablation_one(vname, variant, seed, args, base_cmd_args)
            if not ok:
                failed.append((vname, seed))

    print(f"\n✅ Ablation done. Failed: {failed if failed else 'none'}")
    collect_experiment_results(
        results_root, list(ABLATION_VARIANTS.keys()), label="Ablation")


# =====================================================================
#  主实验 (main_experiments): 3个split × N次
# =====================================================================

def run_main_experiment_one(split, seed, args, base_cmd_args):
    """运行主实验的单次训练"""
    results_root = os.path.join("experiment_results", "main_experiments")
    dst = os.path.join(results_root, split, f"seed_{seed:02d}")

    if os.path.isfile(os.path.join(dst, "history.json")):
        print(f"  ⏭  Already done: {dst}, skipping.")
        return True

    teacher_ckpt = args.teacher_ckpts.get(split, args.teacher_ckpt)
    if teacher_ckpt is None:
        print(f"  ❌ No teacher checkpoint for split={split}")
        return False

    cmd = [
        sys.executable, "main.py",
        args.dataset_root, args.config_file,
        "--mode", "lupi",
        "--split", split,
        "--teacher_ckpt", teacher_ckpt,
    ] + [a for a in base_cmd_args if not a.startswith("--split")]

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)

    print(f"\n{'=' * 60}")
    print(f"▶ MAIN EXP: split={split} | seed={seed}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, env=env)

    src = latest_run_dir("temporal_lupi")
    if src and os.path.isdir(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"  📁 Moved {src} → {dst}")

    return result.returncode == 0


def run_main_experiments(args):
    """运行完整主实验 (3个split)"""
    results_root = os.path.join("experiment_results", "main_experiments")
    splits = ["cross_subject_split", "cross_scene_split", "random_split"]
    seeds  = list(range(args.num_seeds))
    base_cmd_args = build_base_args(args)
    failed = []

    print("=" * 70)
    print(f"🧪 MAIN EXPERIMENTS")
    print(f"   Splits: {splits}")
    print(f"   Seeds: {seeds}")
    print(f"   Total runs: {len(splits) * len(seeds)}")
    print(f"   Teacher checkpoints:")
    for s, c in args.teacher_ckpts.items():
        print(f"     {s}: {c}")
    print("=" * 70)

    for split in splits:
        for seed in seeds:
            ok = run_main_experiment_one(split, seed, args, base_cmd_args)
            if not ok:
                failed.append((split, seed))

    print(f"\n✅ Main experiments done. Failed: {failed if failed else 'none'}")
    collect_experiment_results(
        results_root, splits, label="Main Experiments")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WiFi 3D HPE — Unified Training Entry (All Modes)")
    add_common_args(parser)

    # ---- 模式选择 ----
    parser.add_argument("--mode", type=str, default="auto_teacher_cmc",
                        choices=[
                            # 单次训练模式
                            "baseline",
                            "teacher",
                            "lupi",
                            "auto_cmc",
                            "auto_teacher_cmc",
                            # 批量实验模式
                            "ablation",
                            "main_experiments",
                        ])

    # ---- Teacher 相关 ----
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to teacher checkpoint (.pth)")

    # 主实验用：每个split对应的teacher路径
    parser.add_argument("--teacher_ckpt_subject", type=str, default=None,
                        help="Teacher ckpt for cross_subject_split")
    parser.add_argument("--teacher_ckpt_scene", type=str, default=None,
                        help="Teacher ckpt for cross_scene_split")
    parser.add_argument("--teacher_ckpt_random", type=str, default=None,
                        help="Teacher ckpt for random_split")

    # ---- SWA ----
    parser.add_argument("--swa_start", type=int,   default=10)
    parser.add_argument("--swa_lr",    type=float, default=5e-5)

    # ---- CMC ----
    parser.add_argument("--lambda_cmc",       type=float, default=0.2)
    parser.add_argument("--cmc_tau",          type=float, default=0.1)
    parser.add_argument("--cmc_proj_dim",     type=int,   default=128)
    parser.add_argument("--cmc_stopgrad_rgb", action="store_true")
    parser.add_argument("--cmc_freeze_rgb",   action="store_true")

    # ---- 蒸馏 ----
    parser.add_argument("--lambda_distill", type=float, default=1.0)

    # ---- 批量实验专用 ----
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds for ablation/main_experiments")
    parser.add_argument("--ablation_patience", type=int, default=0,
                        help="Early stopping patience for ablation (0=off)")

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")

    # 构建 teacher_ckpts 字典 (主实验用)
    args.teacher_ckpts = {}
    if args.teacher_ckpt_subject:
        args.teacher_ckpts["cross_subject_split"] = args.teacher_ckpt_subject
    elif args.teacher_ckpt:
        args.teacher_ckpts["cross_subject_split"] = args.teacher_ckpt

    if args.teacher_ckpt_scene:
        args.teacher_ckpts["cross_scene_split"] = args.teacher_ckpt_scene
    elif args.teacher_ckpt:
        args.teacher_ckpts["cross_scene_split"] = args.teacher_ckpt

    if args.teacher_ckpt_random:
        args.teacher_ckpts["random_split"] = args.teacher_ckpt_random
    elif args.teacher_ckpt:
        args.teacher_ckpts["random_split"] = args.teacher_ckpt

    # ================================================================
    #  分发到对应模式
    # ================================================================

    if args.mode == "ablation":
        # ---- 消融实验 ----
        if args.teacher_ckpt is None:
            print("⚠️  消融实验的 B/C/D 变体需要 --teacher_ckpt")
            print("   A_wifi_baseline 仍然可以运行")
        run_ablation(args)

    elif args.mode == "main_experiments":
        # ---- 主实验 ----
        if not args.teacher_ckpts:
            print("❌ 主实验需要指定 teacher checkpoints:")
            print("   --teacher_ckpt_subject xxx.pth")
            print("   --teacher_ckpt_scene   xxx.pth")
            print("   --teacher_ckpt_random  xxx.pth")
            print("   或者用 --teacher_ckpt 统一指定")
            sys.exit(1)
        run_main_experiments(args)

    else:
        # ---- 单次训练模式 ----
        run_single_mode(args, device)


if __name__ == "__main__":
    main()