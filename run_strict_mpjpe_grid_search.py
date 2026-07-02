"""
Small strict four-way MPJPE grid search.

Goal: find practical candidates that can move final-test MPJPE toward 230 mm
without launching a large Bayesian search. The script writes each run to an
isolated directory and relies on the training script to emit final_test.json.
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime


RESULTS_ROOT = "experiment_results/strict_mpjpe_grid_search"
DEFAULT_TEACHER = r"strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(obj, path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def fmt_float(x):
    return str(x).replace(".", "p")


def short_job_id(index: int, tag: str) -> str:
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    return f"r{index:03d}_{digest}"


def job_run_parent(args, index: int, tag: str) -> str:
    if args.short_run_dirs:
        return os.path.join(args.results_root, short_job_id(index, tag))
    return os.path.join(args.results_root, tag)


def read_final_test(run_dir: str):
    path = os.path.join(run_dir, "final_test.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "mpjpe": float(data["mpjpe"]) * 1000.0,
        "pa_mpjpe": float(data.get("pa_mpjpe", 0.0)) * 1000.0,
        "pck@20": float(data.get("pck@20", 0.0)),
        "pck@50": float(data.get("pck@50", 0.0)),
        "checkpoint": data.get("checkpoint", "best.pth"),
        "checkpoint_source": data.get("checkpoint_source", "unknown"),
    }


def run_command(cmd, seed=None):
    env = os.environ.copy()
    if seed is not None:
        env["TRAIN_SEED"] = str(seed)
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode


def run_eval_command(dataset_root, config_file, run_dir, model_type, device,
                     num_workers, val_batch_size):
    cmd = [
        sys.executable,
        "-u",
        "evaluate_final_test_mpjpe.py",
        dataset_root,
        config_file,
        "--run_dir",
        run_dir,
        "--model_type",
        model_type,
        "--device",
        device,
        "--num_workers",
        str(num_workers),
        "--val_batch_size",
        str(val_batch_size),
    ]
    return run_command(cmd, seed=None)


def build_base_args(args):
    return [
        "--split", "four_way_split",
        "--window", str(args.window),
        "--stride", str(args.stride),
        "--val_batch_size", str(args.val_batch_size),
        "--no_cbam",
        "--epochs", str(args.epochs),
        "--warmup_epochs", str(args.warmup_epochs),
        "--weight_decay", str(args.weight_decay),
        "--lr_patience", str(args.lr_patience),
        "--patience", str(args.patience),
        "--swa_start", str(args.swa_start),
        "--no_root_relative",
        "--device", args.device,
        "--num_workers", str(args.num_workers),
    ]


def stability_suffix(aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type):
    suffix = (
        f"_an{fmt_float(aug_noise)}_af{fmt_float(aug_freq)}_at{fmt_float(aug_time)}"
        f"_rel{fmt_float(rel)}_root{fmt_float(root)}_sub{fmt_float(robust)}"
    )
    if bone != 0.0:
        suffix += f"_bone{fmt_float(bone)}"
    if head_type != "mlp":
        suffix += f"_ph{head_type}"
    return suffix


def stability_args(aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type):
    return [
        "--aug_noise", str(aug_noise),
        "--aug_freq_mask", str(aug_freq),
        "--aug_time_mask", str(aug_time),
        "--lambda_rel_pose", str(rel),
        "--lambda_root", str(root),
        "--lambda_bone", str(bone),
        "--subject_robust_weight", str(robust),
        "--pose_head_type", str(head_type),
    ]


def make_baseline_jobs(args):
    for lr, bs, alpha, dropout, layers, ff, head, aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type in itertools.product(
            args.lrs, args.batch_sizes, args.mixup_alphas, args.dropouts,
            args.transformer_layers, args.dim_feedforwards, args.pose_head_hiddens,
            args.aug_noises, args.aug_freq_masks, args.aug_time_masks,
            args.lambda_rel_poses, args.lambda_roots, args.lambda_bones,
            args.subject_robust_weights, args.pose_head_types):
        tag = (
            f"baseline_lr{lr:.1e}_bs{bs}_a{fmt_float(alpha)}_do{fmt_float(dropout)}"
            f"_ly{layers}_ff{ff}_head{head}"
            f"{stability_suffix(aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type)}"
        )
        if args.topk_ckpts > 0:
            tag += f"_topk{args.topk_ckpts}"
        if args.topk_soup:
            tag += "_soup"
        if args.tail_ckpts > 0:
            tag += f"_tail{args.tail_ckpts}"
        if args.tail_soup:
            tag += "_tailsoup"
        extra = [
            "--lr", str(lr),
            "--batch_size", str(bs),
            "--mixup_alpha", str(alpha),
            "--dropout", str(dropout),
            "--transformer_layers", str(layers),
            "--dim_feedforward", str(ff),
            "--pose_head_hidden", str(head),
        ] + stability_args(aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type)
        if args.topk_ckpts > 0:
            extra += ["--topk_ckpts", str(args.topk_ckpts)]
        if args.topk_soup:
            extra += ["--topk_soup"]
        if args.tail_ckpts > 0:
            extra += ["--tail_ckpts", str(args.tail_ckpts)]
        if args.tail_soup:
            extra += ["--tail_soup"]
        yield tag, "train_baseline_mpjpe.py", "baseline", extra


def make_cmc_jobs(args):
    for lr, bs, alpha, dropout, layers, ff, head, aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type, lam, start, ramp, grad in itertools.product(
            args.lrs, args.batch_sizes, args.mixup_alphas, args.dropouts,
            args.transformer_layers, args.dim_feedforwards, args.pose_head_hiddens,
            args.aug_noises, args.aug_freq_masks, args.aug_time_masks,
            args.lambda_rel_poses, args.lambda_roots, args.lambda_bones,
            args.subject_robust_weights, args.pose_head_types,
            args.lambda_cmcs, args.cmc_starts, args.cmc_ramps,
            args.cmc_grad_scales):
        tag = (
            f"cmc_lr{lr:.1e}_bs{bs}_a{fmt_float(alpha)}_do{fmt_float(dropout)}"
            f"_ly{layers}_ff{ff}_head{head}"
            f"{stability_suffix(aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type)}"
            f"_lc{fmt_float(lam)}_cs{start}_cr{ramp}_cg{fmt_float(grad)}"
        )
        extra = [
            "--lr", str(lr),
            "--batch_size", str(bs),
            "--mixup_alpha", str(alpha),
            "--dropout", str(dropout),
            "--transformer_layers", str(layers),
            "--dim_feedforward", str(ff),
            "--pose_head_hidden", str(head),
            "--teacher_ckpt", args.teacher_ckpt,
            "--lambda_cmc", str(lam),
            "--lambda_distill", "0.0",
            "--cmc_stopgrad_rgb",
            "--swa_start", "0",
            "--cmc_start_epoch", str(start),
            "--cmc_ramp_epochs", str(ramp),
            "--cmc_encoder_grad_scale", str(grad),
        ] + stability_args(aug_noise, aug_freq, aug_time, rel, root, bone, robust, head_type)
        yield tag, "train_lupi_rgb_teacher_mpjpe.py", "lupi", extra


def main():
    parser = argparse.ArgumentParser(description="Strict MPJPE small grid search.")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--mode", choices=["baseline", "cmc", "both"], default="both")
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--swa_start", type=int, default=20)
    parser.add_argument("--topk_ckpts", type=int, default=0,
                        help="Baseline only: keep K validation checkpoints")
    parser.add_argument("--topk_soup", action="store_true",
                        help="Baseline only: average saved top-k checkpoints")
    parser.add_argument("--tail_ckpts", type=int, default=0,
                        help="Baseline only: keep K final epoch checkpoints")
    parser.add_argument("--tail_soup", action="store_true",
                        help="Baseline only: average saved tail checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lrs", nargs="+", type=float, default=[7e-5, 1.03e-4, 1.5e-4])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--mixup_alphas", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.15, 0.2])
    parser.add_argument("--transformer_layers", nargs="+", type=int, default=[2])
    parser.add_argument("--dim_feedforwards", nargs="+", type=int, default=[1024])
    parser.add_argument("--pose_head_hiddens", nargs="+", type=int, default=[512])
    parser.add_argument("--aug_noises", nargs="+", type=float, default=[0.0])
    parser.add_argument("--aug_freq_masks", nargs="+", type=float, default=[0.0])
    parser.add_argument("--aug_time_masks", nargs="+", type=float, default=[0.0])
    parser.add_argument("--lambda_rel_poses", nargs="+", type=float, default=[0.0])
    parser.add_argument("--lambda_roots", nargs="+", type=float, default=[0.0])
    parser.add_argument("--lambda_bones", nargs="+", type=float, default=[0.0])
    parser.add_argument("--subject_robust_weights", nargs="+", type=float, default=[0.0])
    parser.add_argument("--pose_head_types", nargs="+", type=str, default=["mlp"],
                        choices=["mlp", "root", "root_relative", "graph", "graph_root", "topology"])
    parser.add_argument("--lambda_cmcs", nargs="+", type=float, default=[0.0, 0.02, 0.05])
    parser.add_argument("--cmc_starts", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--cmc_ramps", nargs="+", type=int, default=[10])
    parser.add_argument("--cmc_grad_scales", nargs="+", type=float, default=[0.0, 0.25])
    parser.add_argument("--results_root", default=RESULTS_ROOT)
    parser.add_argument("--short_run_dirs", action="store_true", default=True,
                        help="Use short hashed run directories to avoid Windows path length limits")
    parser.add_argument("--long_run_dirs", action="store_false", dest="short_run_dirs",
                        help="Use full tag names as run directories")
    parser.add_argument("--inline_final_test", action="store_true",
                        help="Run final-test evaluation inside the training process")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the planned jobs without launching training")
    args = parser.parse_args()

    ensure_dir(args.results_root)
    if args.mode in ("cmc", "both") and not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(args.teacher_ckpt)

    jobs = []
    if args.mode in ("baseline", "both"):
        jobs.extend(make_baseline_jobs(args))
    if args.mode in ("cmc", "both"):
        jobs.extend(make_cmc_jobs(args))

    print("=" * 80)
    print(f"Strict MPJPE grid search started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Jobs: {len(jobs)} x seeds {args.seeds}")
    print(f"Results root: {args.results_root}")
    print("=" * 80)

    if args.dry_run:
        print("Dry run only. Planned jobs:")
        for index, (tag, entry, model_type, extra) in enumerate(jobs):
            parent = job_run_parent(args, index, tag)
            print(f"  {entry}: {tag}")
            if args.short_run_dirs:
                print(f"    run_parent: {parent}")
        return

    base_args = build_base_args(args)

    summaries = []
    for job_index, (tag, entry, model_type, extra) in enumerate(jobs):
        seed_metrics = []
        for seed in args.seeds:
            run_parent = job_run_parent(args, job_index, tag)
            run_dir = os.path.join(run_parent, f"seed_{seed:02d}")
            ensure_dir(run_dir)
            write_json({
                "tag": tag,
                "entry": entry,
                "model_type": model_type,
                "job_index": job_index,
                "seed": seed,
                "extra_args": extra,
                "short_run_dirs": bool(args.short_run_dirs),
            }, os.path.join(run_dir, "job_info.json"))
            final_test = read_final_test(run_dir)
            if final_test is not None:
                print(f"Skip existing: {run_dir} MPJPE={final_test['mpjpe']:.2f}mm")
                seed_metrics.append(final_test)
                continue

            best_path = os.path.join(run_dir, "best.pth")
            complete_path = os.path.join(run_dir, "train_complete.json")
            if os.path.isfile(best_path) and os.path.isfile(complete_path):
                print(f"Found existing checkpoint without final_test.json: {run_dir}")
                print("Running final-test evaluation only...")
                eval_returncode = run_eval_command(
                    args.dataset_root, args.config_file, run_dir, model_type,
                    args.device, args.num_workers, args.val_batch_size)
                if eval_returncode == 0:
                    final_test = read_final_test(run_dir)
                    if final_test is not None:
                        seed_metrics.append(final_test)
                        continue
                print(f"EVAL FAILED: {tag} seed={seed} returncode={eval_returncode}")
                continue
            if os.path.isfile(best_path) and not os.path.isfile(complete_path):
                print(f"Existing incomplete run found, retraining: {run_dir}")

            cmd = [sys.executable, "-u", entry, args.dataset_root, args.config_file]
            cmd += base_args + extra + ["--run_dir", run_dir]
            if not args.inline_final_test:
                cmd += ["--skip_final_test"]

            print("\n" + "=" * 80)
            print(f"Run: {tag} | seed={seed}")
            print(f"Run dir: {run_dir}")
            print("=" * 80)
            returncode = run_command(cmd, seed)
            if returncode != 0:
                print(f"FAILED: {tag} seed={seed} returncode={returncode}")
                continue
            if not args.inline_final_test:
                eval_returncode = run_eval_command(
                    args.dataset_root, args.config_file, run_dir, model_type,
                    args.device, args.num_workers, args.val_batch_size)
                if eval_returncode != 0:
                    print(f"EVAL FAILED: {tag} seed={seed} returncode={eval_returncode}")
                    continue
            final_test = read_final_test(run_dir)
            if final_test is not None:
                seed_metrics.append(final_test)

        if seed_metrics:
            import numpy as np

            mpjpes = [m["mpjpe"] for m in seed_metrics]
            pas = [m["pa_mpjpe"] for m in seed_metrics]
            summary = {
                "tag": tag,
                "entry": entry,
                "n": len(seed_metrics),
                "mpjpe_mean": float(np.mean(mpjpes)),
                "mpjpe_std": float(np.std(mpjpes)),
                "pa_mpjpe_mean": float(np.mean(pas)),
                "pa_mpjpe_std": float(np.std(pas)),
                "metrics": seed_metrics,
            }
            summaries.append(summary)
            write_json(
                sorted(summaries, key=lambda x: x["mpjpe_mean"]),
                os.path.join(args.results_root, "summary.json"))

            print(
                f"SUMMARY {tag}: MPJPE={summary['mpjpe_mean']:.2f}±{summary['mpjpe_std']:.2f} "
                f"PA={summary['pa_mpjpe_mean']:.2f}±{summary['pa_mpjpe_std']:.2f}"
            )

    print("\nTop candidates:")
    for item in sorted(summaries, key=lambda x: x["mpjpe_mean"])[:10]:
        sources = sorted({
            metric.get("checkpoint_source", "unknown")
            for metric in item.get("metrics", [])
        })
        print(
            f"{item['mpjpe_mean']:.2f}±{item['mpjpe_std']:.2f} mm | "
            f"PA {item['pa_mpjpe_mean']:.2f}±{item['pa_mpjpe_std']:.2f} | "
            f"source={','.join(sources)} | {item['tag']}"
        )


if __name__ == "__main__":
    main()
