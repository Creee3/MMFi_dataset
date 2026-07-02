"""
run_cmc_mixup_all_splits_mpjpe.py

CMC + Mixup 变体在 9 种数据划分上的对比实验（MPJPE 优化版）：
  3 种 Protocol × 3 种 Split = 9 种组合

  Protocol:
    protocol1 — 仅日常活动
    protocol2 — 仅康复活动
    protocol3 — 全部活动

  Split:
    random_split      — 80/20 随机划分
    cross_scene_split — 跨场景 (E01/E02/E03 → E04)
    cross_subject_split — 跨个体 (32 → 8 subjects)

每个组合: 2 seeds × 30 epochs, 使用 Bayesian HPO 最优参数
优化目标: MPJPE（而非 PA-MPJPE）

用法:
  python run_cmc_mixup_all_splits_mpjpe.py <dataset_root> <config_file>
  python run_cmc_mixup_all_splits_mpjpe.py data_base config.yaml
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


SEEDS = list(range(2))
RESULTS_ROOT = "experiment_results/cmc_mixup_all_splits_mpjpe"

PROTOCOLS = ["protocol1", "protocol2", "protocol3"]
SPLITS = ["random_split", "cross_scene_split", "cross_subject_split"]

PROTOCOL_LABEL = {
    "protocol1": "Daily",
    "protocol2": "Rehab",
    "protocol3": "All",
}
SPLIT_LABEL = {
    "random_split":      "Random",
    "cross_scene_split": "Cross-Scene",
    "cross_subject_split": "Cross-Subject",
}

# ── 最优超参数（来自 Bayesian HPO mixup 模式） ──
BEST_LR  = 1.03e-04
BEST_BS  = 64

COMMON_ARGS = [
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

CMC_ARGS = [
    "--teacher_ckpt", r"strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth",
    "--lambda_cmc", "0.2",
    "--lambda_distill", "0.0",
    "--cmc_stopgrad_rgb",
    "--swa_start", "0",
]

ENTRY = "train_lupi_rgb_teacher_mpjpe.py"
PREFIX = "temporal_lupi_rgb_teacher_cmc_cbam_distill"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_temp_config(original_config_file: str, protocol: str) -> str:
    """生成临时 config yaml，替换 protocol 字段"""
    with open(original_config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["protocol"] = protocol

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return tmp.name


def run_one(protocol: str, split: str, seed: int,
            dataset_root: str, config_file: str):
    variant_dir = f"{protocol}_{split}"
    dst = os.path.join(RESULTS_ROOT, variant_dir, f"seed_{seed:02d}")
    hist_path = os.path.join(dst, "history.json")
    if os.path.isfile(hist_path):
        print(f"  ⏭  Already done: {dst}, skipping.")
        return True

    # 生成临时 config（替换 protocol）
    tmp_config = make_temp_config(config_file, protocol)

    cmd = [
        sys.executable, "-u", ENTRY,
        dataset_root, tmp_config,
        "--split", split,
    ] + COMMON_ARGS + CMC_ARGS

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n{'=' * 64}")
    print(f"▶ {protocol} × {split} | seed={seed}")
    print(f"  protocol : {PROTOCOL_LABEL[protocol]}")
    print(f"  split    : {SPLIT_LABEL[split]}")
    print(f"  optimize : MPJPE")
    print(f"{'=' * 64}")

    runs_root = "strict_offline_runs"
    existing_dirs = set()
    if os.path.isdir(runs_root):
        existing_dirs = {
            d for d in os.listdir(runs_root)
            if d.startswith(PREFIX)
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

    # 清理临时 config
    os.unlink(tmp_config)

    src = None
    if run_dir_from_stdout and os.path.isdir(run_dir_from_stdout):
        src = run_dir_from_stdout
        print(f"  ✓ Detected run dir from stdout: {src}")

    if src is None and os.path.isdir(runs_root):
        current_dirs = {
            d for d in os.listdir(runs_root)
            if d.startswith(PREFIX)
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
            print(f"  ⚠️ Fallback: no new run dir for prefix '{PREFIX}'")

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
        print(f"  ⚠️ No valid run dir to move (prefix: {PREFIX})")

    return proc.returncode == 0


def collect_results():
    summary_path = os.path.join(RESULTS_ROOT, "summary.json")
    all_results = {}

    for protocol in PROTOCOLS:
        for split in SPLITS:
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

    # ── 打印汇总表格 ──
    print(f"\n📄 Summary -> {summary_path}")
    print("\n" + "=" * 130)
    print("📊 CMC + Mixup 跨 Protocol × Split 对比（MPJPE 优化版）")
    print(f"   Best hparams: lr={BEST_LR:.2e}  bs={BEST_BS}  no CBAM  |  n={len(SEEDS)}")
    print("=" * 130)
    header = (
        f"{'Protocol':<12s} {'Split':<20s} "
        f"{'MPJPE (mm)':<20s} {'PA-MPJPE (mm)':<20s} "
        f"{'PCK@20 (%)':<15s} {'PCK@50 (%)':<15s}"
    )
    print(header)
    print("-" * 130)

    for protocol in PROTOCOLS:
        for split in SPLITS:
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
    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CMC+Mixup across all Protocol×Split combinations (MPJPE-optimized)")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        choices=["protocol1", "protocol2", "protocol3"],
        help="Run only one protocol (default: all 3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        dest="split_filter",
        choices=["random_split", "cross_scene_split", "cross_subject_split"],
        help="Run only one split (default: all 3)",
    )
    args_cli = parser.parse_args()

    dataset_root = args_cli.dataset_root
    config_file = args_cli.config_file
    ensure_dir(RESULTS_ROOT)

    # 筛选
    protocols = [args_cli.protocol] if args_cli.protocol else PROTOCOLS
    splits = [args_cli.split_filter] if args_cli.split_filter else SPLITS

    n_combos = len(protocols) * len(splits)
    n_total = n_combos * len(SEEDS)

    print("=" * 72)
    print("🔬 CMC + Mixup — 跨 3 Protocol × 3 Split 实验（MPJPE 优化版）")
    print(f"   Best params:  lr={BEST_LR:.2e}  bs={BEST_BS}  no CBAM")
    print(f"   Model:        TemporalLUPICMC_CBAM")
    print(f"   CMC:          λ=0.2  τ=0.1")
    print(f"   Distill:      OFF (λ=0.0)")
    print(f"   Optimize:     MPJPE")
    print(f"   Entry:        {ENTRY}")
    print(f"   Protocols:    {[PROTOCOL_LABEL[p] for p in protocols]}")
    print(f"   Splits:       {[SPLIT_LABEL[s] for s in splits]}")
    print(f"   Seeds:        {SEEDS}  (n={len(SEEDS)})")
    print(f"   Total runs:   {n_combos} combos × {len(SEEDS)} seeds = {n_total}")
    print(f"   Results dir:  {RESULTS_ROOT}")
    print(f"   Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    failed = []
    for protocol in protocols:
        for split in splits:
            print(f"\n{'🔷' * 30}")
            print(f"📌 {PROTOCOL_LABEL[protocol]} × {SPLIT_LABEL[split]} "
                  f"× {len(SEEDS)} seeds")
            print(f"{'🔷' * 30}")
            for seed in SEEDS:
                ok = run_one(protocol, split, seed, dataset_root, config_file)
                if not ok:
                    failed.append((protocol, split, seed))

    print(f"\n✅ All done. Failed: {failed if failed else 'none'}")
    collect_results()


if __name__ == "__main__":
    main()
