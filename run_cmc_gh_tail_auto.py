import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


RESULTS_ROOT = Path("experiment_results/cmc_gh_tail")
DEFAULT_TEACHER = "strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth"


def fmt_float(x):
    return str(x).replace(".", "p")


def expected_tag(args):
    tag = (
        "cmc_lr7.0e-05_bs64_a0p16_do0p15_ly2_ff1024_head512"
        "_an0p02_af0p05_at0p05_rel0p0_root0p0_sub0p0"
        "_bone0p005_phgraph_root"
        f"_lc{fmt_float(args.lambda_cmc)}"
        f"_cs{args.cmc_start_epoch}"
        f"_cr{args.cmc_ramp_epochs}"
        f"_cg{fmt_float(args.cmc_encoder_grad_scale)}"
    )
    if args.tail_ckpts > 0:
        tag += f"_tail{args.tail_ckpts}"
    if args.tail_soup:
        tag += "_tailsoup"
    return tag


def run_command(cmd, env=None):
    print("\n" + "=" * 80, flush=True)
    print("Running:", flush=True)
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd), flush=True)
    print("=" * 80, flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def find_run_dir(results_root, seed, tag):
    candidates = []
    seed_dir = f"seed_{seed:02d}"
    for job_info_path in results_root.glob(f"*/{seed_dir}/job_info.json"):
        try:
            with open(job_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if info.get("tag") != tag:
            continue
        run_dir = job_info_path.parent
        candidates.append((job_info_path.stat().st_mtime, run_dir))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def list_checkpoints(run_dir):
    return sorted(path.name for path in run_dir.glob("*.pth"))


def summarize_checkpoint_results(run_dirs, output_path):
    by_checkpoint = defaultdict(list)
    for run_dir in run_dirs:
        result_path = run_dir / "final_test_checkpoints.json"
        if not result_path.is_file():
            print(f"Skip summary for missing file: {result_path}", flush=True)
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload.get("results", []):
            by_checkpoint[item["checkpoint"]].append(item)

    summary = []
    for checkpoint, items in sorted(by_checkpoint.items()):
        mpjpes = [float(item["mpjpe_mm"]) for item in items]
        pas = [float(item["pa_mpjpe_mm"]) for item in items]
        pck20 = [float(item["pck@20"]) for item in items]
        pck50 = [float(item["pck@50"]) for item in items]
        n = len(items)
        mean_mpjpe = sum(mpjpes) / n
        mean_pa = sum(pas) / n
        mean_pck20 = sum(pck20) / n
        mean_pck50 = sum(pck50) / n
        var_mpjpe = sum((x - mean_mpjpe) ** 2 for x in mpjpes) / n
        var_pa = sum((x - mean_pa) ** 2 for x in pas) / n
        summary.append({
            "checkpoint": checkpoint,
            "n": n,
            "mpjpe_mean": mean_mpjpe,
            "mpjpe_std": var_mpjpe ** 0.5,
            "pa_mpjpe_mean": mean_pa,
            "pa_mpjpe_std": var_pa ** 0.5,
            "pck@20_mean": mean_pck20,
            "pck@50_mean": mean_pck50,
            "items": items,
        })

    summary.sort(key=lambda item: item["mpjpe_mean"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nCheckpoint summary:", flush=True)
    for item in summary:
        print(
            f"{item['checkpoint']}: "
            f"MPJPE={item['mpjpe_mean']:.2f}+/-{item['mpjpe_std']:.2f}mm | "
            f"PA={item['pa_mpjpe_mean']:.2f}+/-{item['pa_mpjpe_std']:.2f}mm | "
            f"n={item['n']}",
            flush=True,
        )
    print(f"Saved summary: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the focused CMC graph-head tail-soup experiment."
    )
    parser.add_argument("--dataset_root", default="data_base")
    parser.add_argument("--config_file", default="config.yaml")
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--lambda_cmc", type=float, default=0.02)
    parser.add_argument("--cmc_start_epoch", type=int, default=5)
    parser.add_argument("--cmc_ramp_epochs", type=int, default=10)
    parser.add_argument("--cmc_encoder_grad_scale", type=float, default=0.25)
    parser.add_argument("--tail_ckpts", type=int, default=5)
    parser.add_argument("--tail_soup", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_compare", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    tag = expected_tag(args)
    print(f"Expected CMC tag: {tag}", flush=True)
    print(f"Teacher checkpoint: {args.teacher_ckpt}", flush=True)

    run_command([sys.executable, "check_project_integrity.py"], env=env)

    grid_cmd = [
        sys.executable,
        "run_strict_mpjpe_grid_search.py",
        args.dataset_root,
        args.config_file,
        "--mode",
        "cmc",
        "--teacher_ckpt",
        args.teacher_ckpt,
        "--results_root",
        str(RESULTS_ROOT),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--lrs",
        "7e-5",
        "--mixup_alphas",
        "0.16",
        "--dropouts",
        "0.15",
        "--transformer_layers",
        "2",
        "--dim_feedforwards",
        "1024",
        "--pose_head_hiddens",
        "512",
        "--batch_sizes",
        "64",
        "--aug_noises",
        "0.02",
        "--aug_freq_masks",
        "0.05",
        "--aug_time_masks",
        "0.05",
        "--lambda_rel_poses",
        "0.0",
        "--lambda_roots",
        "0.0",
        "--lambda_bones",
        "0.005",
        "--pose_head_types",
        "graph_root",
        "--subject_robust_weights",
        "0.0",
        "--lambda_cmcs",
        str(args.lambda_cmc),
        "--cmc_starts",
        str(args.cmc_start_epoch),
        "--cmc_ramps",
        str(args.cmc_ramp_epochs),
        "--cmc_grad_scales",
        str(args.cmc_encoder_grad_scale),
        "--swa_start",
        "0",
        "--tail_ckpts",
        str(args.tail_ckpts),
        "--tail_soup",
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.dry_run:
        grid_cmd.append("--dry_run")

    run_command(grid_cmd, env=env)
    if args.dry_run or args.skip_compare:
        return

    run_dirs = []
    for seed in args.seeds:
        run_dir = find_run_dir(RESULTS_ROOT, seed, tag)
        if run_dir is None:
            raise SystemExit(
                f"Could not find seed_{seed:02d} run dir for expected tag under {RESULTS_ROOT}."
            )

        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            raise SystemExit(
                f"No .pth checkpoints found in {run_dir}. Training likely did not finish."
            )

        print("\nLocated CMC run dir:", flush=True)
        print(run_dir, flush=True)
        print("Checkpoints:", flush=True)
        for name in checkpoints:
            print(f"  {name}", flush=True)

        eval_cmd = [
            sys.executable,
            "evaluate_final_test_checkpoints_mpjpe.py",
            args.dataset_root,
            args.config_file,
            "--run_dir",
            str(run_dir),
            "--model_type",
            "lupi",
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
        run_command(eval_cmd, env=env)
        run_dirs.append(run_dir)

    summarize_checkpoint_results(
        run_dirs,
        RESULTS_ROOT / "cmc_tail_checkpoint_summary.json",
    )


if __name__ == "__main__":
    main()
