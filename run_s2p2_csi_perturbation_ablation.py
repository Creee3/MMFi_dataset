"""Run a compact leave-one-out CSI perturbation ablation on S1P1."""

import argparse
import os
import subprocess
import time
from datetime import datetime

from run_s2p2_3x3_rank2 import (
    base_env,
    read_json,
    run_with_retries,
    student_command,
    write_json,
)
from run_s2p2_cmc_hparam_tuning import completed_metric


DEFAULT_SELECTION = os.path.join(
    "strict_offline_runs",
    "S2P2_cmc_lr_bs_tuning_w84200_resume_clean2",
    "selected_for_3x3.json",
)
DEFAULT_TEACHER_SELECTION = os.path.join(
    "strict_offline_runs",
    "S2P2_3x3_rank2",
    "protocol1",
    "random_split",
    "selected_teacher.json",
)
DEFAULT_RESULTS_ROOT = os.path.join(
    "strict_offline_runs",
    "S2P2_csi_perturbation_ablation_s1p1_seed0",
)

PROTOCOL = "protocol1"
SPLIT = "random_split"
SEED = 0
FULL_AUGMENTATION = {
    "aug_noise": 0.08,
    "aug_freq_mask": 0.15,
    "aug_time_mask": 0.15,
}

VARIANTS = (
    {
        "name": "no_noise",
        "description": "Disable Gaussian noise only.",
        "aug_noise": 0.0,
        "aug_freq_mask": 0.15,
        "aug_time_mask": 0.15,
    },
    {
        "name": "no_freq",
        "description": "Disable frequency masking only.",
        "aug_noise": 0.08,
        "aug_freq_mask": 0.0,
        "aug_time_mask": 0.15,
    },
    {
        "name": "no_time",
        "description": "Disable temporal masking only.",
        "aug_noise": 0.08,
        "aug_freq_mask": 0.15,
        "aug_time_mask": 0.0,
    },
    {
        "name": "all_zero",
        "description": "Disable all CSI perturbations.",
        "aug_noise": 0.0,
        "aug_freq_mask": 0.0,
        "aug_time_mask": 0.0,
    },
)


def process_is_running(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def wait_for_process(pid, poll_seconds):
    if pid <= 0:
        return
    while process_is_running(pid):
        print(
            f"[queue] waiting for process {pid} to exit before starting the ablation...",
            flush=True,
        )
        time.sleep(poll_seconds)
    print(f"[queue] process {pid} has exited; starting the ablation.", flush=True)


def require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def result_row(variant, metric, run_dir, source):
    return {
        "variant": variant["name"],
        "description": variant["description"],
        "aug_noise": variant["aug_noise"],
        "aug_freq_mask": variant["aug_freq_mask"],
        "aug_time_mask": variant["aug_time_mask"],
        "seed": SEED,
        "best_epoch": metric["epoch"],
        "mpjpe_mm": metric["mpjpe"],
        "pa_mpjpe_mm": metric["pa_mpjpe"],
        "pck@20": metric["pck@20"],
        "pck@50": metric["pck@50"],
        "run_dir": run_dir,
        "source": source,
    }


def add_deltas(rows):
    reference = next(row for row in rows if row["variant"] == "all_zero")
    for row in rows:
        row["delta_vs_all_zero"] = {
            "mpjpe_mm": row["mpjpe_mm"] - reference["mpjpe_mm"],
            "pa_mpjpe_mm": row["pa_mpjpe_mm"] - reference["pa_mpjpe_mm"],
            "pck@20": row["pck@20"] - reference["pck@20"],
            "pck@50": row["pck@50"] - reference["pck@50"],
        }


def print_summary(rows):
    print("\n" + "=" * 104)
    print("S1P1 seed-0 CSI perturbation ablation")
    print("Variant      noise  freq  time | MPJPE mm | PA-MPJPE mm | PCK@20 | PCK@50 | dMPJPE vs all-zero")
    print("-" * 104)
    for row in rows:
        print(
            f"{row['variant']:<12} "
            f"{row['aug_noise']:>5.2f} "
            f"{row['aug_freq_mask']:>5.2f} "
            f"{row['aug_time_mask']:>5.2f} | "
            f"{row['mpjpe_mm']:>8.3f} | "
            f"{row['pa_mpjpe_mm']:>11.3f} | "
            f"{row['pck@20']:>6.3f} | "
            f"{row['pck@50']:>6.3f} | "
            f"{row['delta_vs_all_zero']['mpjpe_mm']:>+7.3f}"
        )
    print("=" * 104)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Leave-one-out CSI perturbation ablation on protocol1/random_split."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--selected_file", default=DEFAULT_SELECTION)
    parser.add_argument("--teacher_selection", default=DEFAULT_TEACHER_SELECTION)
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--attempt_no_output_timeout", type=int, default=600)
    parser.add_argument("--progress_log_every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait_for_pid", type=int, default=0)
    parser.add_argument("--wait_poll_seconds", type=int, default=60)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.worker_fallbacks = []
    args.min_free_gpu_mib = 0
    args.gpu_poll_seconds = 60

    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.wait_poll_seconds <= 0:
        raise ValueError("--wait_poll_seconds must be positive.")

    require_file(args.selected_file, "selection manifest")
    require_file(args.teacher_selection, "S1P1 teacher selection")
    selection = read_json(args.selected_file)
    teacher = read_json(args.teacher_selection)
    teacher_checkpoint = teacher.get("checkpoint")
    require_file(teacher_checkpoint, "selected S1P1 teacher checkpoint")

    fixed = dict(selection["fixed_student_config"])
    fixed["epochs"] = args.epochs
    fixed["seeds"] = [SEED]

    for key in ("aug_noise", "aug_freq_mask", "aug_time_mask"):
        if float(fixed[key]) != float(FULL_AUGMENTATION[key]):
            raise ValueError(
                f"Selected config has {key}={fixed[key]}, "
                f"expected {FULL_AUGMENTATION[key]}."
            )

    wait_for_process(args.wait_for_pid, args.wait_poll_seconds)

    os.makedirs(args.results_root, exist_ok=True)
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": PROTOCOL,
        "split": SPLIT,
        "seed": SEED,
        "selection_manifest": args.selected_file,
        "teacher_selection": args.teacher_selection,
        "teacher_fold": teacher.get("selected_fold"),
        "teacher_checkpoint": teacher_checkpoint,
        "num_workers": args.num_workers,
        "fixed_student_config": fixed,
        "variants": list(VARIANTS),
    }
    write_json(os.path.join(args.results_root, "ablation_plan.json"), manifest)

    rows = []

    for variant in VARIANTS:
        variant_fixed = dict(fixed)
        for key in ("aug_noise", "aug_freq_mask", "aug_time_mask"):
            variant_fixed[key] = variant[key]

        run_dir = os.path.join(args.results_root, variant["name"], "seed_00")
        env = base_env(PROTOCOL, SEED)
        metric = run_with_retries(
            args,
            f"S1P1/{variant['name']}/seed_00",
            run_dir,
            SEED,
            env,
            lambda workers, directory=run_dir, config=variant_fixed: student_command(
                args,
                config,
                SPLIT,
                teacher_checkpoint,
                directory,
                workers,
            ),
            lambda directory=run_dir: completed_metric(directory),
        )
        if args.dry_run:
            continue
        rows.append(result_row(variant, metric, run_dir, "independent_ablation_run"))

    if args.dry_run:
        print("Dry run complete; no training process was started.")
        return

    add_deltas(rows)
    summary = {
        "status": "complete",
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_metric": "minimum validation MPJPE within each run",
        "results": rows,
        "ranking_by_mpjpe": sorted(rows, key=lambda row: row["mpjpe_mm"]),
    }
    write_json(os.path.join(args.results_root, "ablation_result.json"), summary)
    print_summary(rows)


if __name__ == "__main__":
    main()
