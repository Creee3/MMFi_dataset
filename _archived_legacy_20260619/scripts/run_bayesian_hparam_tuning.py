"""
run_bayesian_hparam_tuning.py

贝叶斯超参数调优（Optuna TPE）

对应四种训练模式（与上游四个脚本一一对应）：
  temporal_only  ←→  run_temporal_only_cmc_comparison_new.py   (no CBAM, no mixup)
  mixup          ←→  run_temporal_mixup_cmc_comparison.py      (no CBAM, mixup ON)
  cbam           ←→  run_temporal_cbam_cmc_comparison.py       (CBAM ON, no mixup)
  cbam_mixup     ←→  run_temporal_cbam_mixup_cmc_comparison.py (CBAM ON, mixup ON)

每种模式：
  - 只跑 CMC variant（+CMC），不跑 baseline
  - 贝叶斯优化直接在 CMC variant 上搜最优 lr + batch_size

搜索空间：
  lr          : log-uniform [1e-4, 5e-3]
  batch_size  : categorical [16, 32, 64, 128, 256]

用法：
  pip install optuna
  python run_bayesian_hparam_tuning.py <dataset_root> <config_file>
  python run_bayesian_hparam_tuning.py data_base config.yaml --modes temporal_only mixup
  python run_bayesian_hparam_tuning.py data_base config.yaml --n_trials 20 --seeds 3
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# Optuna 依赖检查
# ─────────────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ 缺少 optuna，请先安装：pip install optuna")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────
RESULTS_ROOT  = "experiment_results/bayesian_hparam_tuning"
RUNS_ROOT     = "strict_offline_runs"
DEFAULT_SEEDS = 3
DEFAULT_TRIALS = 15  # 每种模式的 BO 轮数；显存充裕可调高

# ── 搜索空间 ─────────────────────────────────────────────────
LR_LOW,  LR_HIGH = 1e-4, 5e-3          # log-uniform
BS_CHOICES       = [16, 32, 64, 128, 256]       # categorical

# ── 四种模式的差异配置 ───────────────────────────────────────
# 每种模式只有 cmc variant，不跑 baseline
CMC_VARIANT = {
    "entry": "train_lupi_rgb_teacher.py",
    "prefix": "temporal_lupi_rgb_teacher_cmc_cbam_distill",
    "extra_args": [
        "--teacher_ckpt", r"strict_offline_runs/T1_rgb_only_teacher/best.pth",
        "--lambda_cmc", "0.2",
        "--lambda_distill", "0.0",
        "--cmc_stopgrad_rgb",
        "--swa_start", "0",
    ],
}

MODES = {
    "temporal_only": {
        "desc": "No CBAM, No Mixup",
        "fixed_override": {
            "--mixup_alpha": "0.0",
            "--no_cbam": None,
        },
        "variant": CMC_VARIANT,
    },
    "mixup": {
        "desc": "No CBAM, Mixup ON (alpha=0.4)",
        "fixed_override": {
            "--mixup_alpha": "0.4",
            "--no_cbam": None,
        },
        "variant": CMC_VARIANT,
    },
    "cbam": {
        "desc": "CBAM ON, No Mixup",
        "fixed_override": {
            "--mixup_alpha": "0.0",
        },
        "variant": CMC_VARIANT,
    },
    "cbam_mixup": {
        "desc": "CBAM ON, Mixup ON (alpha=0.4)",
        "fixed_override": {
            "--mixup_alpha": "0.4",
        },
        "variant": CMC_VARIANT,
    },
}

# ── 固定不变的基础训练参数 ────────────────────────────────────
BASE_FIXED_ARGS = {
    "--split": "cross_subject_split",
    "--epochs": "30",
    "--window": "32",
    "--stride": "2",
    "--dropout": "0.15",
    "--aug_noise": "0.0",
    "--aug_freq_mask": "0.0",
    "--aug_time_mask": "0.0",
    "--lr_patience": "3",
    "--patience": "0",
    "--warmup_epochs": "0",
    "--weight_decay": "1e-3",
    "--val_batch_size": "128",
    "--transformer_layers": "2",
    "--dim_feedforward": "1024",
    "--no_root_relative": None,
    "--device": "cuda",
}


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_args(fixed_base: dict, fixed_override: dict, tune_args: dict, extra: list) -> list:
    """合并所有参数为命令行列表。None 值表示 flag（不带值）。"""
    merged = {**fixed_base, **fixed_override, **tune_args}
    result = []
    for k, v in merged.items():
        result.append(k)
        if v is not None:
            result.append(str(v))
    result.extend(extra)
    return result


def run_single_seed(
    entry: str,
    prefix: str,
    args_list: list,
    seed: int,
    dataset_root: str,
    config_file: str,
    dst: str,
) -> "float | None":
    """
    跑一个 seed，返回最优 epoch 的 PA-MPJPE（单位 mm）；失败返回 None。
    如果 dst/history.json 已存在则直接读取（断点续跑）。
    """
    hist_path = os.path.join(dst, "history.json")
    if os.path.isfile(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            h = json.load(f)
        best = min(h, key=lambda e: e["pa_mpjpe"])
        return best["pa_mpjpe"] * 1000.0

    cmd = [sys.executable, "-u", entry, dataset_root, config_file] + args_list

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"

    # 记录启动前目录（fallback 用）
    existing_dirs: set = set()
    if os.path.isdir(RUNS_ROOT):
        existing_dirs = {
            d for d in os.listdir(RUNS_ROOT)
            if d.startswith(prefix) and os.path.isdir(os.path.join(RUNS_ROOT, d))
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

    if proc.returncode != 0:
        print(f"  ❌ Training failed (seed={seed}, returncode={proc.returncode})")
        return None

    # 定位 run dir
    src = None
    if run_dir_from_stdout and os.path.isdir(run_dir_from_stdout):
        src = run_dir_from_stdout
    elif os.path.isdir(RUNS_ROOT):
        current_dirs = {
            d for d in os.listdir(RUNS_ROOT)
            if d.startswith(prefix) and os.path.isdir(os.path.join(RUNS_ROOT, d))
        }
        new_dirs = sorted(current_dirs - existing_dirs)
        if new_dirs:
            src = os.path.join(RUNS_ROOT, new_dirs[0])

    if src and os.path.isdir(src):
        ensure_dir(os.path.dirname(dst))
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        for attempt in range(5):
            try:
                shutil.move(src, dst)
                break
            except PermissionError:
                time.sleep(2)

    if not os.path.isfile(hist_path):
        print(f"  ❌ history.json not found after training (seed={seed})")
        return None

    with open(hist_path, "r", encoding="utf-8") as f:
        h = json.load(f)
    best = min(h, key=lambda e: e["pa_mpjpe"])
    return best["pa_mpjpe"] * 1000.0


# ─────────────────────────────────────────────────────────────
# 贝叶斯优化目标函数（每次 trial 跑 n_seeds 个 seed 取均值）
# ─────────────────────────────────────────────────────────────
def make_objective(mode_name: str, mode_cfg: dict, dataset_root: str, config_file: str, seeds: list):
    """
    返回一个 Optuna objective 函数。
    trial 提议 lr 和 batch_size，跑 CMC variant，返回 mean PA-MPJPE。
    """
    variant_cfg = mode_cfg["variant"]

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
        bs = trial.suggest_categorical("batch_size", BS_CHOICES)

        tune_args = {"--lr": str(lr), "--batch_size": str(bs)}
        args_list = build_args(
            BASE_FIXED_ARGS,
            mode_cfg["fixed_override"],
            tune_args,
            variant_cfg["extra_args"],
        )

        tag = f"lr{lr:.2e}_bs{bs}".replace("e-0", "e-").replace("e+0", "e")
        print(f"\n  🔍 Trial {trial.number:03d} | {tag}")

        pa_scores = []
        for seed in seeds:
            dst = os.path.join(
                RESULTS_ROOT, mode_name, "bo_trials",
                f"trial_{trial.number:03d}_{tag}", f"seed_{seed:02d}"
            )
            score = run_single_seed(
                variant_cfg["entry"],
                variant_cfg["prefix"],
                args_list,
                seed,
                dataset_root,
                config_file,
                dst,
            )
            if score is not None:
                pa_scores.append(score)
                print(f"    seed={seed}  PA-MPJPE={score:.2f} mm")
            else:
                print(f"    seed={seed}  FAILED")

        if not pa_scores:
            return float("inf")

        mean_score = float(np.mean(pa_scores))
        print(f"  ✅ Trial {trial.number:03d} mean PA-MPJPE = {mean_score:.2f} mm")
        return mean_score

    return objective


# ─────────────────────────────────────────────────────────────
# 用最优超参数跑完整评估（只跑 CMC variant）
# ─────────────────────────────────────────────────────────────
def run_final_eval(mode_name: str, mode_cfg: dict, best_params: dict,
                   dataset_root: str, config_file: str, seeds: list) -> dict:
    print(f"\n{'=' * 64}")
    print(f"🏁 Final Evaluation  |  mode={mode_name}  (CMC only)")
    print(f"   Best params: lr={best_params['lr']:.2e}  bs={best_params['batch_size']}")
    print(f"{'=' * 64}")

    tune_args = {
        "--lr": str(best_params["lr"]),
        "--batch_size": str(best_params["batch_size"]),
    }
    results = {}
    variant_cfg = mode_cfg["variant"]

    args_list = build_args(
        BASE_FIXED_ARGS,
        mode_cfg["fixed_override"],
        tune_args,
        variant_cfg["extra_args"],
    )
    pa_list, mpjpe_list, pck20_list, pck50_list = [], [], [], []

    print(f"\n  ▶ variant=cmc")
    for seed in seeds:
        dst = os.path.join(RESULTS_ROOT, mode_name, "final_eval", "cmc", f"seed_{seed:02d}")
        score = run_single_seed(
            variant_cfg["entry"],
            variant_cfg["prefix"],
            args_list,
            seed,
            dataset_root,
            config_file,
            dst,
        )
        hist_path = os.path.join(dst, "history.json")
        if os.path.isfile(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                h = json.load(f)
            best_ep = min(h, key=lambda e: e["pa_mpjpe"])
            pa_list.append(best_ep["pa_mpjpe"] * 1000.0)
            mpjpe_list.append(best_ep.get("mpjpe", 0.0) * 1000.0)
            pck20_list.append(best_ep.get("pck@20", 0.0))
            pck50_list.append(best_ep.get("pck@50", 0.0))
            print(f"    seed={seed}  PA-MPJPE={pa_list[-1]:.2f} mm")
        else:
            print(f"    seed={seed}  FAILED")

    if pa_list:
        results["cmc"] = {
            "PA-MPJPE": f"{np.mean(pa_list):.2f} ± {np.std(pa_list):.2f}",
            "MPJPE":    f"{np.mean(mpjpe_list):.2f} ± {np.std(mpjpe_list):.2f}",
            "PCK@20":   f"{np.mean(pck20_list):.2f} ± {np.std(pck20_list):.2f}",
            "PCK@50":   f"{np.mean(pck50_list):.2f} ± {np.std(pck50_list):.2f}",
            "pa_mpjpe_mean": float(np.mean(pa_list)),
            "pa_mpjpe_std":  float(np.std(pa_list)),
        }
    return results


# ─────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────
def print_summary(all_summary: dict):
    W = 120
    print("\n" + "=" * W)
    print("📊 Bayesian HPO Summary (CMC only)  |  cross_subject_split / Protocol 2")
    print("=" * W)
    header = (
        f"{'Mode':<16s} {'Description':<32s} "
        f"{'Best lr':>10s} {'Best bs':>8s}  "
        f"{'PA-MPJPE (mm)':>22s}  {'MPJPE (mm)':>22s}  "
        f"{'PCK@20':>12s}  {'PCK@50':>12s}"
    )
    print(header)
    print("-" * W)

    for mode_name, info in all_summary.items():
        best_lr = info.get("best_lr", "—")
        best_bs = info.get("best_bs", "—")
        desc    = info.get("desc", "")
        r       = info.get("final_eval", {}).get("cmc", None)
        lr_show = f"{best_lr:.2e}" if isinstance(best_lr, float) else str(best_lr)

        if r:
            print(
                f"{mode_name:<16s} {desc:<32s} "
                f"{lr_show:>10s} {str(best_bs):>8s}  "
                f"{r['PA-MPJPE']:>22s}  {r['MPJPE']:>22s}  "
                f"{r['PCK@20']:>12s}  {r['PCK@50']:>12s}"
            )
        else:
            print(f"{mode_name:<16s} {desc:<32s}  [missing]")
        print("-" * W)
    print("=" * W)


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="贝叶斯超参数调优 (Optuna TPE)")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "--modes", nargs="+",
        default=list(MODES.keys()),
        choices=list(MODES.keys()),
        help="要跑哪些模式（默认全部）",
    )
    parser.add_argument(
        "--n_trials", type=int, default=DEFAULT_TRIALS,
        help=f"每种模式的 BO 轮数（默认 {DEFAULT_TRIALS}）",
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help=f"每个 trial/最终评估的 seed 数（默认 {DEFAULT_SEEDS}）",
    )
    parser.add_argument(
        "--skip_bo", action="store_true",
        help="跳过 BO 搜索，直接用 --best_lr / --best_bs 做最终评估",
    )
    parser.add_argument("--best_lr", type=float, default=None,
                        help="配合 --skip_bo 使用，指定最优 lr")
    parser.add_argument("--best_bs", type=int, default=None,
                        help="配合 --skip_bo 使用，指定最优 batch_size")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    ensure_dir(RESULTS_ROOT)

    print("=" * 72)
    print("🔬 Bayesian Hyperparameter Optimization (Optuna TPE)")
    print(f"   Modes:        {args.modes}")
    print(f"   Search space: lr ∈ [{LR_LOW:.0e}, {LR_HIGH:.0e}] (log-uniform)")
    print(f"                 batch_size ∈ {BS_CHOICES}")
    print(f"   Trials/mode:  {args.n_trials}")
    print(f"   Seeds/trial:  {len(seeds)}  {seeds}")
    print(f"   Results dir:  {RESULTS_ROOT}")
    print(f"   Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    all_summary = {}

    for mode_name in args.modes:
        mode_cfg = MODES[mode_name]
        print(f"\n{'🔷' * 25}")
        print(f"📌 Mode: {mode_name}  ({mode_cfg['desc']})")
        print(f"{'🔷' * 25}")

        summary_path = os.path.join(RESULTS_ROOT, mode_name, "bo_summary.json")
        best_params  = None

        # ── 1. 贝叶斯搜索 ────────────────────────────────────
        if not args.skip_bo:
            storage_path = os.path.join(RESULTS_ROOT, mode_name, "optuna_study.db")
            storage_url  = f"sqlite:///{storage_path}"
            ensure_dir(os.path.dirname(storage_path))

            study = optuna.create_study(
                study_name=f"hpo_{mode_name}",
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=storage_url,
                load_if_exists=True,  # 断点续跑：已完成的 trial 不重跑
            )

            already_done = len(study.trials)
            remaining    = max(0, args.n_trials - already_done)
            print(f"\n  📈 Optuna study: {already_done} trials already done, "
                  f"{remaining} to go (target={args.n_trials})")

            if remaining > 0:
                study.optimize(
                    make_objective(mode_name, mode_cfg, args.dataset_root, args.config_file, seeds),
                    n_trials=remaining,
                    show_progress_bar=False,
                )

            best_trial  = study.best_trial
            best_params = best_trial.params
            print(f"\n  🏆 Best trial #{best_trial.number}")
            print(f"     lr          = {best_params['lr']:.6f}")
            print(f"     batch_size  = {best_params['batch_size']}")
            print(f"     PA-MPJPE   = {best_trial.value:.2f} mm")

            # 打印 top-5
            all_trials = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value
            )
            print(f"\n  📋 Top-5 trials:")
            for rank, t in enumerate(all_trials[:5], 1):
                print(
                    f"     #{rank} trial={t.number:03d}  "
                    f"lr={t.params['lr']:.2e}  "
                    f"bs={t.params['batch_size']}  "
                    f"PA-MPJPE={t.value:.2f} mm"
                )

        else:
            # 跳过 BO，直接用命令行指定的参数
            if args.best_lr is None or args.best_bs is None:
                print("  ❌ --skip_bo 需要同时指定 --best_lr 和 --best_bs")
                sys.exit(1)
            best_params = {"lr": args.best_lr, "batch_size": args.best_bs}
            print(f"  ⚡ Skipping BO, using lr={args.best_lr:.2e} bs={args.best_bs}")

        # ── 2. 用最优参数跑 baseline + CMC 完整评估 ──────────
        final_results = run_final_eval(
            mode_name, mode_cfg, best_params,
            args.dataset_root, args.config_file, seeds
        )

        # ── 3. 保存本模式结果 ─────────────────────────────────
        mode_summary = {
            "mode": mode_name,
            "desc": mode_cfg["desc"],
            "best_lr":  best_params["lr"],
            "best_bs":  best_params["batch_size"],
            "n_trials": args.n_trials,
            "n_seeds":  len(seeds),
            "final_eval": final_results,
        }
        ensure_dir(os.path.dirname(summary_path))
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(mode_summary, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Saved: {summary_path}")

        all_summary[mode_name] = mode_summary

    # ── 4. 全局汇总 ───────────────────────────────────────────
    global_summary_path = os.path.join(RESULTS_ROOT, "global_summary.json")
    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print_summary(all_summary)
    print(f"\n📄 Global summary -> {global_summary_path}")
    print(f"✅ All done  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")


if __name__ == "__main__":
    main()