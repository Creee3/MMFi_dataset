"""Run the S2P2 temporal/CMC/Mixup component ablation."""

import argparse
import os
import sys
from datetime import datetime


PROJECT_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_python_deps")
if os.path.isdir(PROJECT_DEPS):
    if PROJECT_DEPS not in sys.path:
        sys.path.insert(0, PROJECT_DEPS)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    inherited_paths = [path for path in inherited_pythonpath.split(os.pathsep) if path]
    if PROJECT_DEPS not in inherited_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [PROJECT_DEPS, *inherited_paths]
        )

from run_s2p2_3x3_rank2 import (
    average,
    base_env,
    population_std,
    read_json,
    run_with_retries,
    worker_attempts,
    write_json,
)
from run_s2p2_cmc_hparam_tuning import completed_metric, ensure_dir


PROTOCOL = "protocol2"
SPLIT = "cross_subject_split"
EXPECTED_TEACHER_FOLD = "fold02"

DEFAULT_FORMAL_PLAN = os.path.join(
    "strict_offline_runs",
    "S2P2_3x3_rank2",
    "experiment_plan.json",
)
DEFAULT_TEACHER_SELECTION = os.path.join(
    "strict_offline_runs",
    "S2P2_3x3_rank2",
    PROTOCOL,
    SPLIT,
    "selected_teacher.json",
)
DEFAULT_FULL_RESULT = os.path.join(
    "strict_offline_runs",
    "S2P2_3x3_rank2",
    PROTOCOL,
    SPLIT,
    "student_result.json",
)
DEFAULT_RESULTS_ROOT = os.path.join(
    "strict_offline_runs",
    "S2P2_temporal_cmc_mixup_ablation",
)

VARIANTS = (
    {
        "name": "temporal_only",
        "description": "Temporal WiFi estimator without CMC or Mixup.",
        "use_cmc": False,
        "use_mixup": False,
    },
    {
        "name": "temporal_cmc",
        "description": "Temporal WiFi estimator with CMC only.",
        "use_cmc": True,
        "use_mixup": False,
    },
    {
        "name": "temporal_mixup",
        "description": "Temporal WiFi estimator with Mixup only.",
        "use_cmc": False,
        "use_mixup": True,
    },
)

EXPECTED_FIXED_VALUES = {
    "epochs": 30,
    "window": 32,
    "stride": 2,
    "lr": 0.00020724687211082514,
    "batch_size": 128,
    "val_batch_size": 256,
    "aug_noise": 0.08,
    "aug_freq_mask": 0.15,
    "aug_time_mask": 0.15,
    "mixup_alpha": 0.2,
    "lambda_cmc": 0.1,
}


def require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_fixed_config(fixed):
    missing = [key for key in EXPECTED_FIXED_VALUES if key not in fixed]
    if missing:
        raise ValueError(f"Formal config is missing keys: {', '.join(missing)}")
    for key, expected in EXPECTED_FIXED_VALUES.items():
        actual = fixed[key]
        if isinstance(expected, float):
            matches = abs(float(actual) - expected) <= 1e-12
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Formal config has {key}={actual}, expected {expected}. "
                "Refusing to run an ablation with a changed baseline."
            )
    if list(fixed.get("seeds", [])) != [0, 1, 2]:
        raise ValueError("Formal config must use student seeds 0, 1, and 2.")
    if fixed.get("adaptive_cmc", False):
        raise ValueError("The formal S2P2 config must keep adaptive CMC disabled.")


def variant_config(fixed, variant):
    config = dict(fixed)
    config["lambda_cmc"] = fixed["lambda_cmc"] if variant["use_cmc"] else 0.0
    config["mixup_alpha"] = fixed["mixup_alpha"] if variant["use_mixup"] else 0.0
    return config


def student_command(
        args, fixed, variant, teacher_checkpoint, run_dir, workers):
    command = [
        sys.executable,
        "-u",
        "train_lupi_rgb_teacher_mpjpe.py",
        args.dataset_root,
        args.config_file,
        "--split", SPLIT,
        "--epochs", str(fixed["epochs"]),
        "--window", str(fixed["window"]),
        "--stride", str(fixed["stride"]),
        "--batch_size", str(fixed["batch_size"]),
        "--val_batch_size", str(fixed["val_batch_size"]),
        "--num_workers", str(workers),
        "--eval_num_workers", str(args.eval_num_workers),
        "--device", args.device,
        "--no_cbam",
        "--lr", str(fixed["lr"]),
        "--min_lr", str(fixed["min_lr"]),
        "--lr_patience", str(fixed["lr_patience"]),
        "--warmup_epochs", str(fixed["warmup_epochs"]),
        "--weight_decay", str(fixed["weight_decay"]),
        "--dropout", str(fixed["dropout"]),
        "--aug_noise", str(fixed["aug_noise"]),
        "--aug_freq_mask", str(fixed["aug_freq_mask"]),
        "--aug_time_mask", str(fixed["aug_time_mask"]),
        "--mixup_alpha", str(fixed["mixup_alpha"]),
        "--lambda_cmc", str(fixed["lambda_cmc"]),
        "--cmc_tau", str(fixed["cmc_tau"]),
        "--cmc_proj_dim", str(fixed["cmc_proj_dim"]),
        "--cmc_start_epoch", str(fixed["cmc_start_epoch"]),
        "--cmc_ramp_epochs", str(fixed["cmc_ramp_epochs"]),
        "--cmc_encoder_grad_scale", str(fixed["cmc_encoder_grad_scale"]),
        "--adaptive_cmc_factor", str(fixed["adaptive_cmc_factor_inactive"]),
        "--adaptive_cmc_min_scale", str(fixed["adaptive_cmc_min_scale_inactive"]),
        "--patience", str(fixed["early_stopping_patience"]),
        "--log_every", str(args.progress_log_every),
        "--skip_final_test",
        "--resume_last",
        "--run_dir", run_dir,
    ]
    if variant["use_cmc"]:
        command.extend(["--teacher_ckpt", teacher_checkpoint])
    return command


def summarize_metrics(seed_metrics):
    mpjpes = [row["mpjpe_mm"] for row in seed_metrics]
    pa_mpjpes = [row["pa_mpjpe_mm"] for row in seed_metrics]
    pck20 = [row["pck@20"] for row in seed_metrics]
    pck50 = [row["pck@50"] for row in seed_metrics]
    return {
        "n_completed": len(seed_metrics),
        "mpjpe_mean_mm": average(mpjpes),
        "mpjpe_std_mm": population_std(mpjpes),
        "pa_mpjpe_mean_mm": average(pa_mpjpes),
        "pa_mpjpe_std_mm": population_std(pa_mpjpes),
        "pck@20_mean_pct": average(pck20),
        "pck@20_std_pct": population_std(pck20),
        "pck@50_mean_pct": average(pck50),
        "pck@50_std_pct": population_std(pck50),
    }


def trained_variant_result(variant, fixed, teacher_checkpoint, seed_metrics):
    result = {
        "variant": variant["name"],
        "description": variant["description"],
        "source": "independent_ablation_runs",
        "use_cmc": variant["use_cmc"],
        "use_mixup": variant["use_mixup"],
        "lambda_cmc": fixed["lambda_cmc"],
        "mixup_alpha": fixed["mixup_alpha"],
        "teacher_checkpoint": teacher_checkpoint if variant["use_cmc"] else None,
        "seed_metrics": seed_metrics,
    }
    result.update(summarize_metrics(seed_metrics))
    return result


def reused_full_result(full_result, full_result_path, fixed, teacher_checkpoint):
    if full_result.get("status") != "complete":
        raise ValueError("The formal Full result is not marked complete.")
    if full_result.get("protocol") != PROTOCOL or full_result.get("split") != SPLIT:
        raise ValueError("The formal Full result does not match S2P2.")
    if full_result.get("teacher_fold") != EXPECTED_TEACHER_FOLD:
        raise ValueError("The formal Full result does not use fold02.")
    source_checkpoint = str(full_result.get("teacher_checkpoint", ""))
    if source_checkpoint.replace("/", "\\").lower() != (
            teacher_checkpoint.replace("/", "\\").lower()):
        raise ValueError("The formal Full result and Teacher manifest disagree.")
    if abs(float(full_result.get("lr", -1.0)) - float(fixed["lr"])) > 1e-12:
        raise ValueError("The formal Full result uses a different learning rate.")
    if int(full_result.get("batch_size", -1)) != int(fixed["batch_size"]):
        raise ValueError("The formal Full result uses a different batch size.")
    seed_metrics = []
    for row in full_result.get("seed_metrics", []):
        seed_metrics.append({
            "seed": int(row["seed"]),
            "best_epoch": int(row["epoch"]),
            "mpjpe_mm": float(row["mpjpe"]),
            "pa_mpjpe_mm": float(row["pa_mpjpe"]),
            "pck@20": float(row["pck@20"]),
            "pck@50": float(row["pck@50"]),
            "run_dir": row["run_dir"],
        })
    seed_metrics.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in seed_metrics] != [0, 1, 2]:
        raise ValueError("The formal Full result must contain seeds 0, 1, and 2.")
    computed = summarize_metrics(seed_metrics)
    for stored_key, computed_key in (
        ("mpjpe_mean_mm", "mpjpe_mean_mm"),
        ("mpjpe_std_mm", "mpjpe_std_mm"),
        ("pa_mpjpe_mean_mm", "pa_mpjpe_mean_mm"),
        ("pa_mpjpe_std_mm", "pa_mpjpe_std_mm"),
    ):
        if abs(float(full_result[stored_key]) - computed[computed_key]) > 1e-9:
            raise ValueError(f"The formal Full result has inconsistent {stored_key}.")
    result = {
        "variant": "full",
        "description": "Temporal WiFi estimator with both CMC and Mixup.",
        "source": "reused_formal_result",
        "source_result": full_result_path,
        "use_cmc": True,
        "use_mixup": True,
        "lambda_cmc": fixed["lambda_cmc"],
        "mixup_alpha": fixed["mixup_alpha"],
        "teacher_checkpoint": teacher_checkpoint,
        "seed_metrics": seed_metrics,
    }
    result.update(computed)
    return result


def add_deltas(results):
    reference = next(row for row in results if row["variant"] == "temporal_only")
    for row in results:
        row["delta_vs_temporal_only"] = {
            "mpjpe_mean_mm": row["mpjpe_mean_mm"] - reference["mpjpe_mean_mm"],
            "pa_mpjpe_mean_mm": (
                row["pa_mpjpe_mean_mm"] - reference["pa_mpjpe_mean_mm"]
            ),
            "pck@20_mean_pct": (
                row["pck@20_mean_pct"] - reference["pck@20_mean_pct"]
            ),
            "pck@50_mean_pct": (
                row["pck@50_mean_pct"] - reference["pck@50_mean_pct"]
            ),
        }


def print_summary(results):
    print("\n" + "=" * 104)
    print("S2P2 temporal/CMC/Mixup ablation (three seeds)")
    print("Variant          CMC Mixup | MPJPE mm       | PA-MPJPE mm    | PCK@20       | PCK@50")
    print("-" * 104)
    for row in results:
        print(
            f"{row['variant']:<16} "
            f"{str(row['use_cmc']):>3} "
            f"{str(row['use_mixup']):>5} | "
            f"{row['mpjpe_mean_mm']:>7.3f} +/- {row['mpjpe_std_mm']:<6.3f} | "
            f"{row['pa_mpjpe_mean_mm']:>7.3f} +/- {row['pa_mpjpe_std_mm']:<6.3f} | "
            f"{row['pck@20_mean_pct']:>6.3f} +/- {row['pck@20_std_pct']:<6.3f} | "
            f"{row['pck@50_mean_pct']:>6.3f} +/- {row['pck@50_std_pct']:<6.3f}"
        )
    print("=" * 104)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the S2P2 temporal/CMC/Mixup component ablation."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--formal_plan", default=DEFAULT_FORMAL_PLAN)
    parser.add_argument("--teacher_selection", default=DEFAULT_TEACHER_SELECTION)
    parser.add_argument("--full_result", default=DEFAULT_FULL_RESULT)
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--worker_fallbacks", nargs="*", type=int, default=[4, 2, 1, 0])
    parser.add_argument("--attempt_no_output_timeout", type=int, default=600)
    parser.add_argument("--progress_log_every", type=int, default=100)
    parser.add_argument("--min_free_gpu_mib", type=int, default=0)
    parser.add_argument("--gpu_poll_seconds", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    require_file(args.formal_plan, "formal S2P2 experiment plan")
    require_file(args.teacher_selection, "selected S2P2 teacher manifest")
    require_file(args.full_result, "formal S2P2 Full result")
    if args.num_workers < 0 or any(value < 0 for value in args.worker_fallbacks):
        raise ValueError("Worker counts must be non-negative.")
    if args.eval_num_workers < 0:
        raise ValueError("--eval_num_workers must be non-negative.")
    if args.progress_log_every <= 0:
        raise ValueError("--progress_log_every must be positive.")

    formal_plan = read_json(args.formal_plan)
    if not any(
            cell.get("protocol") == PROTOCOL and cell.get("split") == SPLIT
            for cell in formal_plan.get("cells", [])):
        raise ValueError("Formal experiment plan does not contain the S2P2 cell.")
    fixed = dict(formal_plan["fixed_student_config"])
    validate_fixed_config(fixed)

    teacher = read_json(args.teacher_selection)
    if teacher.get("protocol") != PROTOCOL or teacher.get("split") != SPLIT:
        raise ValueError("Teacher manifest does not match S2P2.")
    if teacher.get("selected_fold") != EXPECTED_TEACHER_FOLD:
        raise ValueError("Teacher manifest does not select fold02.")
    teacher_checkpoint = teacher.get("checkpoint")
    if not teacher_checkpoint:
        raise ValueError("Teacher manifest has no checkpoint path.")
    if not args.dry_run:
        require_file(teacher_checkpoint, "selected fold02 teacher checkpoint")

    full_result = read_json(args.full_result)
    full = reused_full_result(
        full_result,
        args.full_result,
        fixed,
        teacher_checkpoint,
    )

    ensure_dir(args.results_root)
    plan = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": PROTOCOL,
        "split": SPLIT,
        "formal_plan": args.formal_plan,
        "teacher_selection": args.teacher_selection,
        "teacher_fold": teacher.get("selected_fold"),
        "teacher_checkpoint": teacher_checkpoint,
        "full_result": args.full_result,
        "full_result_policy": "reuse_without_retraining",
        "fixed_student_config": fixed,
        "worker_attempts": worker_attempts(args),
        "eval_num_workers": args.eval_num_workers,
        "minimum_free_gpu_mib": args.min_free_gpu_mib,
        "variants_to_train": [
            {**variant, **{
                "lambda_cmc": variant_config(fixed, variant)["lambda_cmc"],
                "mixup_alpha": variant_config(fixed, variant)["mixup_alpha"],
            }}
            for variant in VARIANTS
        ],
    }
    write_json(os.path.join(args.results_root, "ablation_plan.json"), plan)

    print("=" * 80)
    print("S2P2 temporal/CMC/Mixup ablation")
    print(f"Protocol/split: {PROTOCOL}/{SPLIT}")
    print(f"Seeds: {fixed['seeds']}")
    print(f"Workers: {' -> '.join(map(str, worker_attempts(args)))}")
    print(f"Teacher reused: {teacher_checkpoint}")
    print(f"Full result reused: {args.full_result}")
    print(f"Results: {args.results_root}")
    print("=" * 80)

    results = []
    for variant in VARIANTS:
        config = variant_config(fixed, variant)
        seed_metrics = []
        print("\n" + "#" * 80)
        print(
            f"VARIANT {variant['name']} | "
            f"lambda_cmc={config['lambda_cmc']} | "
            f"mixup_alpha={config['mixup_alpha']}"
        )
        print("#" * 80)
        for seed in fixed["seeds"]:
            run_dir = os.path.join(args.results_root, variant["name"], f"seed_{seed:02d}")
            env = base_env(PROTOCOL, seed)
            metric = run_with_retries(
                args,
                f"S2P2/{variant['name']}/seed_{seed:02d}",
                run_dir,
                seed,
                env,
                lambda workers, directory=run_dir, current=config, item=variant: (
                    student_command(
                        args,
                        current,
                        item,
                        teacher_checkpoint,
                        directory,
                        workers,
                    )
                ),
                lambda directory=run_dir: completed_metric(directory),
            )
            if args.dry_run:
                continue
            seed_metrics.append({
                "seed": seed,
                "best_epoch": metric["epoch"],
                "mpjpe_mm": metric["mpjpe"],
                "pa_mpjpe_mm": metric["pa_mpjpe"],
                "pck@20": metric["pck@20"],
                "pck@50": metric["pck@50"],
                "run_dir": run_dir,
            })

        if args.dry_run:
            continue
        result = trained_variant_result(
            variant,
            config,
            teacher_checkpoint,
            seed_metrics,
        )
        results.append(result)
        write_json(
            os.path.join(args.results_root, variant["name"], "variant_result.json"),
            result,
        )

    if args.dry_run:
        print("Dry run complete; no training process was started.")
        return

    results.append(full)
    add_deltas(results)
    summary = {
        "status": "complete",
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": PROTOCOL,
        "split": SPLIT,
        "selection_metric": "minimum validation MPJPE within each seed run",
        "comparison_policy": "three seeds per variant; Full reused from formal run",
        "results": results,
        "ranking_by_mpjpe": sorted(results, key=lambda row: row["mpjpe_mean_mm"]),
    }
    write_json(os.path.join(args.results_root, "ablation_result.json"), summary)
    print_summary(results)


if __name__ == "__main__":
    main()
