"""
run_bayesian_hparam_tuning_mpjpe.py

贝叶斯超参数调优（Optuna TPE）— MPJPE 优化版

与 run_bayesian_hparam_tuning.py 完全相同的逻辑，唯一区别：
  优化目标从 PA-MPJPE 改为 MPJPE

四种训练模式：
  temporal_only  ←→  纯时序 + CMC (no CBAM, no mixup)
  mixup          ←→  时序 + Mixup + CMC (no CBAM)
  cbam           ←→  时序 + CBAM + CMC (no mixup)
  cbam_mixup     ←→  时序 + CBAM + Mixup + CMC

每种模式只跑 CMC variant（不跑 baseline），贝叶斯优化直接在 CMC 上搜最优 lr + bs。

搜索空间：
  lr          : log-uniform [1e-4, 5e-3]
  batch_size  : categorical [16, 32, 64, 128, 256]
  optional    : mixup_alpha / lambda_cmc / dropout

用法：
  pip install optuna
  python run_bayesian_hparam_tuning_mpjpe.py <dataset_root> <config_file>
  python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes temporal_only mixup
  python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --n_trials 20 --seeds 3
  python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --tune_mixup_alpha --tune_lambda_cmc
"""

import copy
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import yaml

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
RESULTS_ROOT  = "experiment_results/bayesian_hparam_tuning_mpjpe"
STRICT_RESULTS_ROOT = "experiment_results/bayesian_hparam_tuning_mpjpe_four_way_s2p2"
DEFAULT_SEEDS = 3
DEFAULT_TRIALS = 15
DEFAULT_TEACHER_CKPT = r"strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth"

# ── 搜索空间 ─────────────────────────────────────────────────
LR_LOW,  LR_HIGH = 1e-4, 5e-3
BS_CHOICES       = [16, 32, 64, 128, 256]
MIXUP_ALPHA_CHOICES = [0.0, 0.1, 0.2, 0.4]
LAMBDA_CMC_CHOICES  = [0.05, 0.1, 0.2, 0.4]
DROPOUT_CHOICES     = [0.1, 0.15, 0.2, 0.25]
LAMBDA_REL_POSE_CHOICES = [0.0, 0.25, 0.5, 1.0]
LAMBDA_ROOT_CHOICES     = [0.0, 0.25, 0.5, 1.0]
SUBJECT_ROBUST_CHOICES  = [0.0, 0.2, 0.4]
CMC_START_CHOICES       = [1, 5, 10]
CMC_RAMP_CHOICES        = [0, 5, 10]
CMC_GRAD_SCALE_CHOICES  = [0.0, 0.25, 0.5, 1.0]

# ── CMC 配置（使用 MPJPE 优化版训练脚本） ─────────────────────
CMC_VARIANT = {
    "entry": "train_lupi_rgb_teacher_mpjpe.py",
    "prefix": "temporal_lupi_rgb_teacher_cmc_cbam_distill",
    "extra_args": [
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


def read_protocol(config_file: str) -> str:
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("protocol", "unknown")


def resolve_results_root(split: str, explicit_root: str | None) -> str:
    if explicit_root:
        return explicit_root
    if split == "four_way_split":
        return STRICT_RESULTS_ROOT
    return RESULTS_ROOT


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


def replace_arg(args_list: list, key: str, value: str | None) -> list:
    """Return a copy with all existing key occurrences removed, then append key/value."""
    cleaned = []
    i = 0
    while i < len(args_list):
        if args_list[i] == key:
            i += 2 if i + 1 < len(args_list) and not args_list[i + 1].startswith("--") else 1
            continue
        cleaned.append(args_list[i])
        i += 1
    cleaned.append(key)
    if value is not None:
        cleaned.append(str(value))
    return cleaned


def run_single_seed(
    entry: str,
    args_list: list,
    seed: int,
    dataset_root: str,
    config_file: str,
    dst: str,
) -> "float | None":
    """
    跑一个 seed，返回最优 epoch 的 MPJPE（单位 mm）；失败返回 None。
    如果 dst/history.json 已存在则直接读取（断点续跑）。
    """
    ensure_dir(dst)
    hist_path = os.path.join(dst, "history.json")
    if os.path.isfile(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            h = json.load(f)
        best = min(h, key=lambda e: e["mpjpe"])
        return best["mpjpe"] * 1000.0

    cmd = [sys.executable, "-u", entry, dataset_root, config_file] + args_list + ["--run_dir", dst]

    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"

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

    if proc.returncode != 0:
        print(f"  ❌ Training failed (seed={seed}, returncode={proc.returncode})")
        return None

    if not os.path.isfile(hist_path):
        print(f"  ❌ history.json not found after training (seed={seed})")
        return None

    with open(hist_path, "r", encoding="utf-8") as f:
        h = json.load(f)
    best = min(h, key=lambda e: e["mpjpe"])
    return best["mpjpe"] * 1000.0


def load_eval_metrics(dst: str, prefer_test_metrics: bool) -> dict | None:
    if prefer_test_metrics:
        final_test_path = os.path.join(dst, "final_test.json")
        if os.path.isfile(final_test_path):
            with open(final_test_path, "r", encoding="utf-8") as f:
                test_result = json.load(f)
            return {
                "source": "test",
                "mpjpe": float(test_result["mpjpe"]) * 1000.0,
                "pa_mpjpe": float(test_result.get("pa_mpjpe", 0.0)) * 1000.0,
                "pck@20": float(test_result.get("pck@20", 0.0)),
                "pck@50": float(test_result.get("pck@50", 0.0)),
            }

    hist_path = os.path.join(dst, "history.json")
    if not os.path.isfile(hist_path):
        return None
    with open(hist_path, "r", encoding="utf-8") as f:
        h = json.load(f)
    best_ep = min(h, key=lambda e: e["mpjpe"])
    return {
        "source": "val",
        "mpjpe": float(best_ep["mpjpe"]) * 1000.0,
        "pa_mpjpe": float(best_ep.get("pa_mpjpe", 0.0)) * 1000.0,
        "pck@20": float(best_ep.get("pck@20", 0.0)),
        "pck@50": float(best_ep.get("pck@50", 0.0)),
    }


# ─────────────────────────────────────────────────────────────
# 贝叶斯优化目标函数（每次 trial 跑 n_seeds 个 seed 取均值）
# ─────────────────────────────────────────────────────────────
def make_objective(
    mode_name: str,
    mode_cfg: dict,
    dataset_root: str,
    config_file: str,
    seeds: list,
    results_root: str,
    fixed_base_args: dict,
    tune_mixup_alpha: bool = False,
    tune_lambda_cmc: bool = False,
    tune_dropout: bool = False,
    tune_aux_losses: bool = False,
    tune_subject_robust: bool = False,
    tune_cmc_schedule: bool = False,
):
    """
    返回一个 Optuna objective 函数。
    trial 提议超参数，跑 CMC variant，返回 mean MPJPE。
    """
    variant_cfg = mode_cfg["variant"]

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
        bs = trial.suggest_categorical("batch_size", BS_CHOICES)

        tune_args = {"--lr": str(lr), "--batch_size": str(bs)}
        fixed_override = dict(mode_cfg["fixed_override"])
        extra_args = list(variant_cfg["extra_args"])

        if tune_mixup_alpha:
            mixup_alpha = trial.suggest_categorical("mixup_alpha", MIXUP_ALPHA_CHOICES)
            fixed_override["--mixup_alpha"] = str(mixup_alpha)

        if tune_lambda_cmc:
            lambda_cmc = trial.suggest_categorical("lambda_cmc", LAMBDA_CMC_CHOICES)
            extra_args = replace_arg(extra_args, "--lambda_cmc", str(lambda_cmc))

        if tune_dropout:
            dropout = trial.suggest_categorical("dropout", DROPOUT_CHOICES)
            tune_args["--dropout"] = str(dropout)

        if tune_aux_losses:
            rel_pose = trial.suggest_categorical("lambda_rel_pose", LAMBDA_REL_POSE_CHOICES)
            root = trial.suggest_categorical("lambda_root", LAMBDA_ROOT_CHOICES)
            tune_args["--lambda_rel_pose"] = str(rel_pose)
            tune_args["--lambda_root"] = str(root)

        if tune_subject_robust:
            robust = trial.suggest_categorical("subject_robust_weight", SUBJECT_ROBUST_CHOICES)
            tune_args["--subject_robust_weight"] = str(robust)

        if tune_cmc_schedule:
            cmc_start = trial.suggest_categorical("cmc_start_epoch", CMC_START_CHOICES)
            cmc_ramp = trial.suggest_categorical("cmc_ramp_epochs", CMC_RAMP_CHOICES)
            cmc_grad = trial.suggest_categorical("cmc_encoder_grad_scale", CMC_GRAD_SCALE_CHOICES)
            tune_args["--cmc_start_epoch"] = str(cmc_start)
            tune_args["--cmc_ramp_epochs"] = str(cmc_ramp)
            tune_args["--cmc_encoder_grad_scale"] = str(cmc_grad)

        args_list = build_args(
            fixed_base_args,
            fixed_override,
            tune_args,
            extra_args,
        )

        tag_parts = [f"lr{lr:.2e}", f"bs{bs}"]
        if tune_mixup_alpha:
            tag_parts.append(f"a{trial.params['mixup_alpha']}")
        if tune_lambda_cmc:
            tag_parts.append(f"cmc{trial.params['lambda_cmc']}")
        if tune_dropout:
            tag_parts.append(f"do{trial.params['dropout']}")
        if tune_aux_losses:
            tag_parts.append(f"rel{trial.params['lambda_rel_pose']}")
            tag_parts.append(f"root{trial.params['lambda_root']}")
        if tune_subject_robust:
            tag_parts.append(f"rob{trial.params['subject_robust_weight']}")
        if tune_cmc_schedule:
            tag_parts.append(f"cs{trial.params['cmc_start_epoch']}")
            tag_parts.append(f"cr{trial.params['cmc_ramp_epochs']}")
            tag_parts.append(f"cg{trial.params['cmc_encoder_grad_scale']}")
        tag = "_".join(tag_parts).replace("e-0", "e-").replace("e+0", "e")
        print(f"\n  🔍 Trial {trial.number:03d} | {tag}")

        mpjpe_scores = []
        for seed in seeds:
            dst = os.path.join(
                results_root, mode_name, "bo_trials",
                f"trial_{trial.number:03d}_{tag}", f"seed_{seed:02d}"
            )
            score = run_single_seed(
                variant_cfg["entry"],
                args_list,
                seed,
                dataset_root,
                config_file,
                dst,
            )
            if score is not None:
                mpjpe_scores.append(score)
                print(f"    seed={seed}  MPJPE={score:.2f} mm")
            else:
                print(f"    seed={seed}  FAILED")

        if not mpjpe_scores:
            return float("inf")

        mean_score = float(np.mean(mpjpe_scores))
        print(f"  ✅ Trial {trial.number:03d} mean MPJPE = {mean_score:.2f} mm")
        return mean_score

    return objective


# ─────────────────────────────────────────────────────────────
# 用最优超参数跑完整评估
# ─────────────────────────────────────────────────────────────
def run_final_eval(mode_name: str, mode_cfg: dict, best_params: dict,
                   dataset_root: str, config_file: str, seeds: list,
                   results_root: str, split: str, fixed_base_args: dict) -> dict:
    print(f"\n{'=' * 64}")
    print(f"🏁 Final Evaluation  |  mode={mode_name}  (CMC only, optimize MPJPE)")
    print(f"   Best params: lr={best_params['lr']:.2e}  bs={best_params['batch_size']}")
    print(f"{'=' * 64}")

    tune_args = {
        "--lr": str(best_params["lr"]),
        "--batch_size": str(best_params["batch_size"]),
    }
    fixed_override = dict(mode_cfg["fixed_override"])
    extra_args = list(mode_cfg["variant"]["extra_args"])
    if "mixup_alpha" in best_params:
        fixed_override["--mixup_alpha"] = str(best_params["mixup_alpha"])
    if "lambda_cmc" in best_params:
        extra_args = replace_arg(extra_args, "--lambda_cmc", str(best_params["lambda_cmc"]))
    if "dropout" in best_params:
        tune_args["--dropout"] = str(best_params["dropout"])
    if "lambda_rel_pose" in best_params:
        tune_args["--lambda_rel_pose"] = str(best_params["lambda_rel_pose"])
    if "lambda_root" in best_params:
        tune_args["--lambda_root"] = str(best_params["lambda_root"])
    if "subject_robust_weight" in best_params:
        tune_args["--subject_robust_weight"] = str(best_params["subject_robust_weight"])
    if "cmc_start_epoch" in best_params:
        tune_args["--cmc_start_epoch"] = str(best_params["cmc_start_epoch"])
    if "cmc_ramp_epochs" in best_params:
        tune_args["--cmc_ramp_epochs"] = str(best_params["cmc_ramp_epochs"])
    if "cmc_encoder_grad_scale" in best_params:
        tune_args["--cmc_encoder_grad_scale"] = str(best_params["cmc_encoder_grad_scale"])
    results = {}
    args_list = build_args(
        fixed_base_args,
        fixed_override,
        tune_args,
        extra_args,
    )
    pa_list, mpjpe_list, pck20_list, pck50_list = [], [], [], []

    print(f"\n  ▶ variant=cmc")
    prefer_test_metrics = (split == "four_way_split")
    for seed in seeds:
        dst = os.path.join(results_root, mode_name, "final_eval", "cmc", f"seed_{seed:02d}")
        run_single_seed(
            variant_cfg["entry"],
            args_list,
            seed,
            dataset_root,
            config_file,
            dst,
        )
        metrics = load_eval_metrics(dst, prefer_test_metrics=prefer_test_metrics)
        if metrics is not None:
            pa_list.append(metrics["pa_mpjpe"])
            mpjpe_list.append(metrics["mpjpe"])
            pck20_list.append(metrics["pck@20"])
            pck50_list.append(metrics["pck@50"])
            print(
                f"    seed={seed}  [{metrics['source']}]  "
                f"MPJPE={mpjpe_list[-1]:.2f} mm  PA-MPJPE={pa_list[-1]:.2f} mm"
            )
        else:
            print(f"    seed={seed}  FAILED")

    if mpjpe_list:
        results["cmc"] = {
            "MPJPE":    f"{np.mean(mpjpe_list):.2f} ± {np.std(mpjpe_list):.2f}",
            "PA-MPJPE": f"{np.mean(pa_list):.2f} ± {np.std(pa_list):.2f}",
            "PCK@20":   f"{np.mean(pck20_list):.2f} ± {np.std(pck20_list):.2f}",
            "PCK@50":   f"{np.mean(pck50_list):.2f} ± {np.std(pck50_list):.2f}",
            "mpjpe_mean": float(np.mean(mpjpe_list)),
            "mpjpe_std":  float(np.std(mpjpe_list)),
        }
    return results


# ─────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────
def print_summary(all_summary: dict, split: str, protocol: str):
    W = 120
    print("\n" + "=" * W)
    print(f"📊 Bayesian HPO Summary — MPJPE Optimized  |  {split} / {protocol}")
    print("=" * W)
    header = (
        f"{'Mode':<16s} {'Description':<32s} "
        f"{'Best lr':>10s} {'Best bs':>8s}  "
        f"{'MPJPE (mm)':>22s}  {'PA-MPJPE (mm)':>22s}  "
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
                f"{r['MPJPE']:>22s}  {r['PA-MPJPE']:>22s}  "
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

    parser = argparse.ArgumentParser(description="贝叶斯超参数调优 — MPJPE 优化版 (Optuna TPE)")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument(
        "--modes", nargs="+",
        default=["mixup"],
        choices=list(MODES.keys()),
        help="要跑哪些模式（默认 mixup）",
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
    parser.add_argument(
        "--split", type=str, default="four_way_split",
        choices=["random_split", "cross_subject_split", "cross_scene_split", "four_way_split"],
        help="训练/调优所使用的数据划分（默认 four_way_split）",
    )
    parser.add_argument(
        "--teacher_ckpt", type=str, default=DEFAULT_TEACHER_CKPT,
        help="RGB teacher checkpoint 路径",
    )
    parser.add_argument(
        "--results_root", type=str, default=None,
        help="可选：显式指定结果目录，默认旧 split 用旧目录，four_way_split 用独立 strict 目录",
    )
    parser.add_argument(
        "--tune_mixup_alpha", action="store_true",
        help=f"同时搜索 mixup_alpha，候选值 {MIXUP_ALPHA_CHOICES}",
    )
    parser.add_argument(
        "--tune_lambda_cmc", action="store_true",
        help=f"同时搜索 lambda_cmc，候选值 {LAMBDA_CMC_CHOICES}",
    )
    parser.add_argument(
        "--tune_dropout", action="store_true",
        help=f"同时搜索 dropout，候选值 {DROPOUT_CHOICES}",
    )
    parser.add_argument(
        "--tune_aux_losses", action="store_true",
        help=f"同时搜索 lambda_rel_pose/lambda_root，候选值 {LAMBDA_REL_POSE_CHOICES}",
    )
    parser.add_argument(
        "--tune_subject_robust", action="store_true",
        help=f"同时搜索 subject_robust_weight，候选值 {SUBJECT_ROBUST_CHOICES}",
    )
    parser.add_argument(
        "--tune_cmc_schedule", action="store_true",
        help="同时搜索 CMC start/ramp/encoder gradient scale",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    protocol = read_protocol(args.config_file)
    results_root = resolve_results_root(args.split, args.results_root)
    ensure_dir(results_root)

    if not os.path.isfile(args.teacher_ckpt):
        print("❌ Teacher checkpoint 不存在：")
        print(f"   {args.teacher_ckpt}")
        print("   请先训练对应划分下的 teacher，或通过 --teacher_ckpt 指向正确的 best.pth")
        sys.exit(1)

    print("=" * 72)
    print("🔬 Bayesian Hyperparameter Optimization (Optuna TPE) — MPJPE 优化版")
    print(f"   Modes:        {args.modes}")
    print(f"   Protocol:     {protocol}")
    print(f"   Split:        {args.split}")
    print(f"   Search space: lr ∈ [{LR_LOW:.0e}, {LR_HIGH:.0e}] (log-uniform)")
    print(f"                 batch_size ∈ {BS_CHOICES}")
    if args.tune_mixup_alpha:
        print(f"                 mixup_alpha ∈ {MIXUP_ALPHA_CHOICES}")
    if args.tune_lambda_cmc:
        print(f"                 lambda_cmc ∈ {LAMBDA_CMC_CHOICES}")
    if args.tune_dropout:
        print(f"                 dropout ∈ {DROPOUT_CHOICES}")
    if args.tune_aux_losses:
        print(f"                 lambda_rel_pose ∈ {LAMBDA_REL_POSE_CHOICES}")
        print(f"                 lambda_root ∈ {LAMBDA_ROOT_CHOICES}")
    if args.tune_subject_robust:
        print(f"                 subject_robust_weight ∈ {SUBJECT_ROBUST_CHOICES}")
    if args.tune_cmc_schedule:
        print(f"                 cmc_start_epoch ∈ {CMC_START_CHOICES}")
        print(f"                 cmc_ramp_epochs ∈ {CMC_RAMP_CHOICES}")
        print(f"                 cmc_encoder_grad_scale ∈ {CMC_GRAD_SCALE_CHOICES}")
    print(f"   Trials/mode:  {args.n_trials}")
    print(f"   Seeds/trial:  {len(seeds)}  {seeds}")
    print(f"   Entry:        {CMC_VARIANT['entry']}")
    print(f"   Teacher:      {args.teacher_ckpt}")
    print(f"   Optimize:     MPJPE")
    print(f"   Results dir:  {results_root}")
    print(f"   Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    all_summary = {}
    fixed_base_args = dict(BASE_FIXED_ARGS)
    fixed_base_args["--split"] = args.split

    for mode_name in args.modes:
        mode_cfg = copy.deepcopy(MODES[mode_name])
        mode_cfg["variant"]["extra_args"].extend(["--teacher_ckpt", args.teacher_ckpt])
        print(f"\n{'🔷' * 25}")
        print(f"📌 Mode: {mode_name}  ({mode_cfg['desc']})  — optimize MPJPE")
        print(f"{'🔷' * 25}")

        summary_path = os.path.join(results_root, mode_name, "bo_summary.json")
        best_params  = None

        # ── 1. 贝叶斯搜索 ────────────────────────────────────
        if not args.skip_bo:
            storage_path = os.path.join(results_root, mode_name, "optuna_study.db")
            storage_url  = f"sqlite:///{storage_path}"
            ensure_dir(os.path.dirname(storage_path))

            study = optuna.create_study(
                study_name=f"hpo_{mode_name}_{args.split}_{protocol}_mpjpe",
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=storage_url,
                load_if_exists=True,
            )

            already_done = len(study.trials)
            remaining    = max(0, args.n_trials - already_done)
            print(f"\n  📈 Optuna study: {already_done} trials already done, "
                  f"{remaining} to go (target={args.n_trials})")

            if remaining > 0:
                study.optimize(
                    make_objective(
                        mode_name, mode_cfg, args.dataset_root, args.config_file,
                        seeds, results_root, fixed_base_args,
                        tune_mixup_alpha=args.tune_mixup_alpha,
                        tune_lambda_cmc=args.tune_lambda_cmc,
                        tune_dropout=args.tune_dropout,
                        tune_aux_losses=args.tune_aux_losses,
                        tune_subject_robust=args.tune_subject_robust,
                        tune_cmc_schedule=args.tune_cmc_schedule,
                    ),
                    n_trials=remaining,
                    show_progress_bar=False,
                )

            best_trial  = study.best_trial
            best_params = best_trial.params
            print(f"\n  🏆 Best trial #{best_trial.number}")
            print(f"     lr          = {best_params['lr']:.6f}")
            print(f"     batch_size  = {best_params['batch_size']}")
            print(f"     MPJPE      = {best_trial.value:.2f} mm")

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
                    f"MPJPE={t.value:.2f} mm"
                )

        else:
            if args.best_lr is None or args.best_bs is None:
                print("  ❌ --skip_bo 需要同时指定 --best_lr 和 --best_bs")
                sys.exit(1)
            best_params = {"lr": args.best_lr, "batch_size": args.best_bs}
            print(f"  ⚡ Skipping BO, using lr={args.best_lr:.2e} bs={args.best_bs}")

        # ── 2. 用最优参数跑 CMC 完整评估 ─────────────────────
        final_results = run_final_eval(
            mode_name, mode_cfg, best_params,
            args.dataset_root, args.config_file, seeds,
            results_root, args.split, fixed_base_args
        )

        # ── 3. 保存本模式结果 ─────────────────────────────────
        mode_summary = {
            "mode": mode_name,
            "desc": mode_cfg["desc"],
            "optimize": "MPJPE",
            "protocol": protocol,
            "split": args.split,
            "teacher_ckpt": args.teacher_ckpt,
            "tuned_params": {
                "lr": True,
                "batch_size": True,
                "mixup_alpha": args.tune_mixup_alpha,
                "lambda_cmc": args.tune_lambda_cmc,
                "dropout": args.tune_dropout,
                "lambda_rel_pose": args.tune_aux_losses,
                "lambda_root": args.tune_aux_losses,
                "subject_robust_weight": args.tune_subject_robust,
                "cmc_schedule": args.tune_cmc_schedule,
            },
            "best_lr":  best_params["lr"],
            "best_bs":  best_params["batch_size"],
            "best_mixup_alpha": best_params.get("mixup_alpha"),
            "best_lambda_cmc": best_params.get("lambda_cmc"),
            "best_dropout": best_params.get("dropout"),
            "best_lambda_rel_pose": best_params.get("lambda_rel_pose"),
            "best_lambda_root": best_params.get("lambda_root"),
            "best_subject_robust_weight": best_params.get("subject_robust_weight"),
            "best_cmc_start_epoch": best_params.get("cmc_start_epoch"),
            "best_cmc_ramp_epochs": best_params.get("cmc_ramp_epochs"),
            "best_cmc_encoder_grad_scale": best_params.get("cmc_encoder_grad_scale"),
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
    global_summary_path = os.path.join(results_root, "global_summary.json")
    ensure_dir(os.path.dirname(global_summary_path))
    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print_summary(all_summary, args.split, protocol)
    print(f"\n📄 Global summary -> {global_summary_path}")
    print(f"✅ All done  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")


if __name__ == "__main__":
    main()
