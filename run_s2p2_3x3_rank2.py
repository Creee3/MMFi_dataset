"""Run the final protocol x split RGB-Teacher/CMC-Student experiment grid."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

from mmfi_lib.mmfi import decode_config
from run_s2p2_cmc_hparam_tuning import (
    completed_metric,
    ensure_dir,
    has_nonretryable_error,
    run_command,
    write_command,
)


PROTOCOLS = ("protocol1", "protocol2", "protocol3")
SPLITS = ("random_split", "cross_scene_split", "cross_subject_split")
DEFAULT_SELECTION = os.path.join(
    "strict_offline_runs",
    "S2P2_cmc_lr_bs_tuning_w84200_resume_clean2",
    "selected_for_3x3.json",
)
DEFAULT_RESULTS_ROOT = os.path.join("strict_offline_runs", "S2P2_3x3_rank2")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def average(values):
    return sum(values) / len(values)


def population_std(values):
    center = average(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def normalized_form(data_form):
    return {
        subject: sorted(dict.fromkeys(actions))
        for subject, actions in sorted(data_form.items())
    }


def form_pairs(data_form):
    return {
        (subject, action)
        for subject, actions in data_form.items()
        for action in actions
    }


def official_outer_forms(config_file, protocol, split):
    with open(config_file, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["protocol"] = protocol
    cfg["split_to_use"] = split
    cfg["data_unit"] = "frame"
    cfg["modality"] = "wifi-csi|rgb"
    decoded = decode_config(cfg)
    train_form = normalized_form(decoded["train_dataset"]["data_form"])
    validation_form = normalized_form(decoded["val_dataset"]["data_form"])
    if not train_form or not validation_form:
        raise RuntimeError(f"{protocol}/{split} has an empty official outer split.")
    overlap = form_pairs(train_form) & form_pairs(validation_form)
    if overlap:
        raise RuntimeError(
            f"{protocol}/{split} outer train/validation overlap in "
            f"{len(overlap)} subject-action pairs."
        )
    return train_form, validation_form


def balanced_chunks(values, count):
    quotient, remainder = divmod(len(values), count)
    chunks = []
    cursor = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        chunks.append(values[cursor:cursor + size])
        cursor += size
    return chunks


def subject_number(subject):
    return int(subject[1:])


def make_subject_folds(train_form, fold_count=4):
    by_environment = {}
    for subject in sorted(train_form, key=subject_number):
        environment = (subject_number(subject) - 1) // 10
        by_environment.setdefault(environment, []).append(subject)

    folds = [[] for _ in range(fold_count)]
    for subjects in by_environment.values():
        for index, chunk in enumerate(balanced_chunks(subjects, fold_count)):
            folds[index].extend(chunk)

    flattened = [subject for fold in folds for subject in fold]
    if sorted(flattened) != sorted(train_form) or len(flattened) != len(set(flattened)):
        raise RuntimeError("Teacher CV folds do not partition the outer train subjects.")
    if any(not fold for fold in folds):
        raise RuntimeError("Teacher CV produced an empty validation fold.")
    return [sorted(fold, key=subject_number) for fold in folds]


def split_form_by_subject(train_form, validation_subjects):
    validation_set = set(validation_subjects)
    fold_train = {
        subject: actions for subject, actions in train_form.items()
        if subject not in validation_set
    }
    fold_validation = {
        subject: actions for subject, actions in train_form.items()
        if subject in validation_set
    }
    if not fold_train or not fold_validation:
        raise RuntimeError("Teacher fold train/validation data_form is empty.")
    if set(fold_train) & set(fold_validation):
        raise RuntimeError("Teacher fold train and validation subjects overlap.")
    return fold_train, fold_validation


def validate_teacher_folds(outer_train, outer_validation, folds):
    outer_train_pairs = form_pairs(outer_train)
    outer_validation_pairs = form_pairs(outer_validation)
    seen_validation_pairs = set()
    for validation_subjects in folds:
        fold_train, fold_validation = split_form_by_subject(
            outer_train, validation_subjects)
        train_pairs = form_pairs(fold_train)
        validation_pairs = form_pairs(fold_validation)
        if train_pairs & validation_pairs:
            raise RuntimeError("Teacher fold train/validation pairs overlap.")
        if train_pairs | validation_pairs != outer_train_pairs:
            raise RuntimeError("Teacher fold does not cover the exact outer train mapping.")
        if validation_pairs & outer_validation_pairs:
            raise RuntimeError("Teacher CV leaks official outer validation pairs.")
        if seen_validation_pairs & validation_pairs:
            raise RuntimeError("Teacher validation folds overlap.")
        seen_validation_pairs.update(validation_pairs)
    if seen_validation_pairs != outer_train_pairs:
        raise RuntimeError("Teacher validation folds do not partition outer train pairs.")


def sequence_count(data_form):
    return sum(len(actions) for actions in data_form.values())


def base_env(protocol, seed):
    env = os.environ.copy()
    env["MMFI_PROTOCOL"] = protocol
    env["HELDOUT_SPLIT_ENABLED"] = "0"
    env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    return env


def teacher_env(protocol, train_form, validation_form):
    env = base_env(protocol, seed=0)
    env["TEACHER_CV_TRAIN_FORM_JSON"] = json.dumps(
        train_form, separators=(",", ":"))
    env["TEACHER_CV_VAL_FORM_JSON"] = json.dumps(
        validation_form, separators=(",", ":"))
    return env


def gpu_status():
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        if result.returncode == 0 and len(parts) >= 4:
            return {
                "free_mib": int(float(parts[0])),
                "used_mib": int(float(parts[1])),
                "utilization": int(float(parts[2])),
                "temperature": int(float(parts[3])),
            }
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def wait_for_gpu(minimum_free_mib, poll_seconds):
    if minimum_free_mib <= 0:
        return
    while True:
        status = gpu_status()
        if status is None:
            print("[gpu-wait] nvidia-smi unavailable; proceeding without an idle check.")
            return
        if status["free_mib"] >= minimum_free_mib:
            print(
                f"[gpu-wait] ready: free={status['free_mib']}MiB "
                f"used={status['used_mib']}MiB util={status['utilization']}%"
            )
            return
        print(
            f"[gpu-wait] waiting: free={status['free_mib']}MiB < "
            f"{minimum_free_mib}MiB; used={status['used_mib']}MiB "
            f"util={status['utilization']}% temp={status['temperature']}C"
        )
        sys.stdout.flush()
        time.sleep(poll_seconds)


def teacher_metric(run_dir, required_epochs):
    history_path = os.path.join(run_dir, "history.json")
    best_path = os.path.join(run_dir, "best.pth")
    complete_path = os.path.join(run_dir, "train_complete.json")
    if not all(os.path.isfile(path) for path in (history_path, best_path, complete_path)):
        return None
    try:
        history = read_json(history_path)
        complete = read_json(complete_path)
    except (OSError, ValueError):
        return None
    if not history or complete.get("status") != "complete":
        return None
    if int(history[-1].get("epoch", 0)) < required_epochs:
        return None
    best = min(history, key=lambda row: float(row.get("mpjpe", 1e9)))
    return {
        "best_epoch": int(best["epoch"]),
        "mpjpe_mm": float(best["mpjpe"]) * 1000.0,
        "pa_mpjpe_mm": float(best.get("pa_mpjpe", 0.0)) * 1000.0,
        "pck@20": float(best.get("pck@20", 0.0)),
        "pck@50": float(best.get("pck@50", 0.0)),
        "last_epoch": int(history[-1]["epoch"]),
        "checkpoint": best_path,
    }


def worker_attempts(args):
    return [args.num_workers] + list(args.worker_fallbacks)


def run_with_retries(
        args, label, run_dir, seed, env, build_command, read_metric):
    existing = read_metric()
    if existing is not None:
        print(f"[{label}] existing complete result reused.")
        return existing

    ensure_dir(run_dir)
    process_log = os.path.join(run_dir, "process_output.log")
    attempts = worker_attempts(args)
    for attempt, workers in enumerate(attempts):
        command = build_command(workers)
        write_command(
            os.path.join(run_dir, f"command_attempt_{attempt:02d}_workers{workers}.txt"),
            command,
        )
        print("\n" + "=" * 80)
        print(f"[{label}] attempt {attempt + 1}/{len(attempts)} | workers={workers}")
        print(" ".join(command))
        print("=" * 80)
        if args.dry_run:
            return {"status": "dry_run"}

        wait_for_gpu(args.min_free_gpu_mib, args.gpu_poll_seconds)

        attempt_log = os.path.join(
            run_dir, f"process_output_attempt_{attempt:02d}_workers{workers}.log")
        returncode = run_command(
            command,
            seed,
            process_log,
            attempt_log,
            no_output_timeout=args.attempt_no_output_timeout,
            env=env,
        )
        metric = read_metric()
        if returncode == 0 and metric is not None:
            return metric
        print(
            f"[{label}] attempt did not complete: returncode={returncode}, "
            f"complete_marker={metric is not None}"
        )
        if has_nonretryable_error(attempt_log):
            raise RuntimeError(f"{label} hit a non-retryable data/runtime error.")
    raise RuntimeError(f"{label} failed after {len(attempts)} attempts.")


def teacher_command(args, run_dir, workers):
    return [
        sys.executable, "-u", "train_t1_rgb_teacher_mpjpe.py",
        args.dataset_root, args.config_file,
        "--split", "teacher_student_split",
        "--epochs", str(args.epochs),
        "--window", "32",
        "--stride", "2",
        "--batch_size", str(args.teacher_batch_size),
        "--val_batch_size", "256",
        "--num_workers", str(workers),
        "--eval_num_workers", str(args.eval_num_workers),
        "--device", args.device,
        "--mixup_alpha", "0.2",
        "--dropout", "0.25",
        "--lr", "0.00005",
        "--weight_decay", "0.01",
        "--warmup_epochs", "3",
        "--lr_patience", "3",
        "--patience", "0",
        "--aug_noise", "0.10",
        "--aug_freq_mask", "0.20",
        "--aug_time_mask", "0.20",
        "--log_every", str(args.progress_log_every),
        "--skip_final_test",
        "--resume_last",
        "--run_dir", run_dir,
    ]


def run_teacher_cv(args, protocol, split, cell_dir):
    outer_train, outer_validation = official_outer_forms(
        args.config_file, protocol, split)
    folds = make_subject_folds(outer_train)
    validate_teacher_folds(outer_train, outer_validation, folds)
    cv_dir = os.path.join(cell_dir, "teacher_cv")
    ensure_dir(cv_dir)
    write_json(
        os.path.join(cell_dir, "outer_split_info.json"),
        {
            "protocol": protocol,
            "split": split,
            "train_subject_count": len(outer_train),
            "validation_subject_count": len(outer_validation),
            "train_sequence_count": sequence_count(outer_train),
            "validation_sequence_count": sequence_count(outer_validation),
            "train_form": outer_train,
            "validation_form": outer_validation,
            "teacher_cv_validation_subjects": folds,
            "validation_status": "no subject-action leakage",
        },
    )
    fold_rows = []

    for index, validation_subjects in enumerate(folds, 1):
        fold_name = f"fold{index:02d}"
        run_dir = os.path.join(cv_dir, fold_name)
        train_form, validation_form = split_form_by_subject(
            outer_train, validation_subjects)
        fold_info = {
            "fold": fold_name,
            "protocol": protocol,
            "outer_split": split,
            "train_subjects": sorted(train_form, key=subject_number),
            "validation_subjects": validation_subjects,
            "train_sequence_count": sequence_count(train_form),
            "validation_sequence_count": sequence_count(validation_form),
            "train_form": train_form,
            "validation_form": validation_form,
        }
        write_json(os.path.join(run_dir, "fold_info.json"), fold_info)
        env = teacher_env(protocol, train_form, validation_form)
        metric = run_with_retries(
            args,
            f"{protocol}/{split}/teacher/{fold_name}",
            run_dir,
            0,
            env,
            lambda workers, directory=run_dir: teacher_command(
                args, directory, workers),
            lambda directory=run_dir: teacher_metric(directory, args.epochs),
        )
        if args.dry_run:
            continue
        fold_rows.append({**fold_info, **metric})
        write_json(os.path.join(cv_dir, "summary_all.json"), fold_rows)

    if args.dry_run:
        return {
            "selected_fold": "selected_after_teacher_cv",
            "checkpoint": os.path.join(
                cv_dir, "selected_after_teacher_cv", "best.pth"),
        }
    ranked = sorted(fold_rows, key=lambda row: row["mpjpe_mm"])
    if len(ranked) != 4:
        raise RuntimeError(f"{protocol}/{split} Teacher CV is incomplete.")
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    write_json(os.path.join(cv_dir, "summary_ranked.json"), ranked)
    selected = {
        "protocol": protocol,
        "split": split,
        "selection_objective": "minimum internal-fold validation MPJPE",
        "selected_fold": ranked[0]["fold"],
        "checkpoint": ranked[0]["checkpoint"],
        "mpjpe_mm": ranked[0]["mpjpe_mm"],
        "best_epoch": ranked[0]["best_epoch"],
        "ranked_folds": ranked,
    }
    write_json(os.path.join(cell_dir, "selected_teacher.json"), selected)
    return selected


def student_command(args, fixed, split, teacher_checkpoint, run_dir, workers):
    return [
        sys.executable, "-u", "train_lupi_rgb_teacher_mpjpe.py",
        args.dataset_root, args.config_file,
        "--split", split,
        "--teacher_ckpt", teacher_checkpoint,
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


def run_students(args, protocol, split, cell_dir, teacher, fixed):
    metrics = []
    for seed in fixed["seeds"]:
        run_dir = os.path.join(cell_dir, "student", f"seed_{seed:02d}")
        env = base_env(protocol, seed)
        metric = run_with_retries(
            args,
            f"{protocol}/{split}/student/seed_{seed:02d}",
            run_dir,
            seed,
            env,
            lambda workers, directory=run_dir: student_command(
                args, fixed, split, teacher["checkpoint"], directory, workers),
            lambda directory=run_dir: completed_metric(directory),
        )
        if args.dry_run:
            continue
        metrics.append({**metric, "seed": seed, "run_dir": run_dir})

    if args.dry_run:
        return None
    mpjpes = [row["mpjpe"] for row in metrics]
    pa_mpjpes = [row["pa_mpjpe"] for row in metrics]
    result = {
        "status": "complete",
        "protocol": protocol,
        "split": split,
        "teacher_fold": teacher["selected_fold"],
        "teacher_checkpoint": teacher["checkpoint"],
        "lr": fixed["lr"],
        "batch_size": fixed["batch_size"],
        "n_completed": len(metrics),
        "mpjpe_mean_mm": average(mpjpes),
        "mpjpe_std_mm": population_std(mpjpes),
        "pa_mpjpe_mean_mm": average(pa_mpjpes),
        "pa_mpjpe_std_mm": population_std(pa_mpjpes),
        "seed_metrics": metrics,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(cell_dir, "student_result.json"), result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Run the final S2P2 3x3 experiment grid.")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--selected_file", default=DEFAULT_SELECTION)
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--teacher_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--worker_fallbacks", nargs="*", type=int, default=[2, 1, 0, 0])
    parser.add_argument("--attempt_no_output_timeout", type=int, default=600)
    parser.add_argument("--progress_log_every", type=int, default=100)
    parser.add_argument("--min_free_gpu_mib", type=int, default=20000)
    parser.add_argument("--gpu_poll_seconds", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not os.path.isfile(args.selected_file):
        raise FileNotFoundError(f"Selection manifest not found: {args.selected_file}")
    selection = read_json(args.selected_file)
    fixed = dict(selection["fixed_student_config"])
    fixed["epochs"] = args.epochs
    if fixed.get("adaptive_cmc", False):
        raise ValueError("The selected tuning configuration must keep adaptive CMC disabled.")
    if args.num_workers < 0 or any(value < 0 for value in args.worker_fallbacks):
        raise ValueError("Worker counts must be non-negative.")
    if args.progress_log_every <= 0:
        raise ValueError("--progress_log_every must be positive.")

    ensure_dir(args.results_root)
    cells = [
        {"protocol": protocol, "split": split}
        for protocol in args.protocols for split in args.splits
    ]
    plan = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_manifest": args.selected_file,
        "selected_trial": selection["selected"],
        "fixed_student_config": fixed,
        "teacher_config": {
            "folds": 4,
            "seed": 0,
            "epochs": args.epochs,
            "batch_size": args.teacher_batch_size,
            "selection_objective": "minimum internal validation MPJPE",
        },
        "worker_attempts": worker_attempts(args),
        "progress_log_every": args.progress_log_every,
        "minimum_free_gpu_mib": args.min_free_gpu_mib,
        "cells": cells,
    }
    write_json(os.path.join(args.results_root, "experiment_plan.json"), plan)

    print("=" * 80)
    print("S2P2 final 3x3 experiment grid")
    print(f"Cells: {len(cells)} | workers: {' -> '.join(map(str, worker_attempts(args)))}")
    print(f"Student lr={fixed['lr']} batch_size={fixed['batch_size']}")
    print(f"GPU gate: minimum free memory {args.min_free_gpu_mib} MiB")
    print(f"Results: {args.results_root}")
    print("=" * 80)

    results = []
    for cell in cells:
        protocol = cell["protocol"]
        split = cell["split"]
        cell_dir = os.path.join(args.results_root, protocol, split)
        ensure_dir(cell_dir)
        print("\n" + "#" * 80)
        print(f"CELL {protocol} x {split}")
        print("#" * 80)
        teacher = run_teacher_cv(args, protocol, split, cell_dir)
        result = run_students(args, protocol, split, cell_dir, teacher, fixed)
        if args.dry_run:
            continue
        results.append(result)
        write_json(os.path.join(args.results_root, "summary_all.json"), results)

    if args.dry_run:
        print("Dry run complete; no training process was started.")
        return
    write_json(
        os.path.join(args.results_root, "run_complete.json"),
        {
            "status": "complete",
            "completed_cells": len(results),
            "expected_cells": len(cells),
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"Completed {len(results)}/{len(cells)} cells.")


if __name__ == "__main__":
    main()
