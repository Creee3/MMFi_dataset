"""
Sequential S2P2 WiFi + CMC + Mixup exploration.

This runner now keeps only the clean CMC main line:
  WiFi temporal pose loss + RGB feature CMC + Mixup.

No graph/root/bone heads, pose distillation, subject-robust loss, SWA, or
checkpoint soup variants are included here.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


RESULTS_ROOT = "strict_offline_runs"
DEFAULT_TEACHER = r"strict_offline_runs\T1_rgb_teacher_single_s2p2_mix02\best.pth"


VALUE_FLAGS = {
    "--epochs",
    "--batch_size",
    "--val_batch_size",
    "--lr",
    "--weight_decay",
    "--warmup_epochs",
    "--log_every",
    "--device",
    "--lr_patience",
    "--min_lr",
    "--num_workers",
    "--eval_num_workers",
    "--d_model",
    "--window",
    "--stride",
    "--dropout",
    "--transformer_layers",
    "--dim_feedforward",
    "--pose_head_hidden",
    "--aug_noise",
    "--aug_freq_mask",
    "--aug_time_mask",
    "--mixup_alpha",
    "--patience",
    "--split",
    "--teacher_ckpt",
    "--lambda_cmc",
    "--cmc_tau",
    "--cmc_proj_dim",
    "--cmc_start_epoch",
    "--cmc_ramp_epochs",
    "--cmc_encoder_grad_scale",
    "--adaptive_cmc_factor",
    "--adaptive_cmc_min_scale",
    "--run_dir",
}


BOOLEAN_FLAGS = {
    "--use_cbam",
    "--no_cbam",
    "--adaptive_cmc",
    "--skip_final_test",
}


FALLBACK_SUPPORTED_FLAGS = VALUE_FLAGS | BOOLEAN_FLAGS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(obj, path):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_best_val(run_dir):
    hist_path = os.path.join(run_dir, "history.json")
    if not os.path.isfile(hist_path):
        return None
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return None
    best = min(history, key=lambda row: float(row.get("mpjpe", 1e9)))
    return {
        "epoch": int(best.get("epoch", -1)),
        "mpjpe": float(best["mpjpe"]) * 1000.0,
        "pa_mpjpe": float(best.get("pa_mpjpe", 0.0)) * 1000.0,
        "pck@20": float(best.get("pck@20", 0.0)),
        "pck@50": float(best.get("pck@50", 0.0)),
    }


def run_command(cmd, seed):
    env = os.environ.copy()
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


def get_supported_flags():
    proc = subprocess.run(
        [sys.executable, "train_lupi_rgb_teacher_mpjpe.py", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stdout
    if proc.returncode != 0:
        return set(FALLBACK_SUPPORTED_FLAGS), text
    flags = set()
    for token in text.replace(",", " ").split():
        if token.startswith("--"):
            flags.add(token.split()[0].strip())
    if not flags:
        flags = set(FALLBACK_SUPPORTED_FLAGS)
    return flags, text


def filter_supported_args(arg_list, supported_flags):
    filtered = []
    skipped = []
    i = 0
    while i < len(arg_list):
        item = arg_list[i]
        if item.startswith("--") and item not in supported_flags:
            if item in VALUE_FLAGS and i + 1 < len(arg_list) and not arg_list[i + 1].startswith("--"):
                skipped.append(f"{item} {arg_list[i + 1]}")
                i += 2
            else:
                skipped.append(item)
                i += 1
            continue
        filtered.append(item)
        i += 1
    return filtered, skipped


def jobs():
    return [
        {
            "tag": "01_cmc010_mix02_ramp",
            "desc": "Stable CMC ramp, mixup 0.2.",
            "args": ["--mixup_alpha", "0.2", "--lambda_cmc", "0.1",
                     "--cmc_start_epoch", "5", "--cmc_ramp_epochs", "10",
                     "--cmc_encoder_grad_scale", "0.25"],
        },
        {
            "tag": "02_cmc005_mix01_ramp",
            "desc": "Gentler mixup and weaker CMC.",
            "args": ["--mixup_alpha", "0.1", "--lambda_cmc", "0.05",
                     "--cmc_start_epoch", "5", "--cmc_ramp_epochs", "10",
                     "--cmc_encoder_grad_scale", "0.25"],
        },
        {
            "tag": "03_cmc005_mix03_ramp",
            "desc": "More mixup with weak CMC.",
            "args": ["--mixup_alpha", "0.3", "--lambda_cmc", "0.05",
                     "--cmc_start_epoch", "5", "--cmc_ramp_epochs", "10",
                     "--cmc_encoder_grad_scale", "0.25"],
        },
        {
            "tag": "04_latecmc_mix02",
            "desc": "Later CMC start after WiFi pose warm-up.",
            "args": ["--mixup_alpha", "0.2", "--lambda_cmc", "0.1",
                     "--cmc_start_epoch", "10", "--cmc_ramp_epochs", "10",
                     "--cmc_encoder_grad_scale", "0.25"],
        },
        {
            "tag": "05_tau007_mix02",
            "desc": "Sharper contrastive temperature.",
            "args": ["--mixup_alpha", "0.2", "--lambda_cmc", "0.1",
                     "--cmc_tau", "0.07", "--cmc_start_epoch", "5",
                     "--cmc_ramp_epochs", "10", "--cmc_encoder_grad_scale", "0.25"],
        },
        {
            "tag": "06_tau020_mix02",
            "desc": "Smoother contrastive temperature.",
            "args": ["--mixup_alpha", "0.2", "--lambda_cmc", "0.1",
                     "--cmc_tau", "0.20", "--cmc_start_epoch", "5",
                     "--cmc_ramp_epochs", "10", "--cmc_encoder_grad_scale", "0.25"],
        },
        {
            "tag": "07_proj256_mix02",
            "desc": "Larger CMC projection space.",
            "args": ["--mixup_alpha", "0.2", "--lambda_cmc", "0.1",
                     "--cmc_proj_dim", "256", "--cmc_start_epoch", "5",
                     "--cmc_ramp_epochs", "10", "--cmc_encoder_grad_scale", "0.25"],
        },
        {
            "tag": "08_adaptive_cmc_mix02",
            "desc": "Adaptive CMC scale after non-improving validation epochs.",
            "args": ["--mixup_alpha", "0.2", "--lambda_cmc", "0.1",
                     "--cmc_start_epoch", "5", "--cmc_ramp_epochs", "10",
                     "--cmc_encoder_grad_scale", "0.25", "--adaptive_cmc",
                     "--adaptive_cmc_factor", "0.85",
                     "--adaptive_cmc_min_scale", "0.35"],
        },
    ]


def build_base_cmd(args, run_dir, supported_flags):
    cmd = [
        sys.executable,
        "-u",
        "train_lupi_rgb_teacher_mpjpe.py",
        args.dataset_root,
        args.config_file,
        "--split", "cross_subject_split",
        "--teacher_ckpt", args.teacher_ckpt,
        "--epochs", str(args.epochs),
        "--window", str(args.window),
        "--stride", str(args.stride),
        "--batch_size", str(args.batch_size),
        "--val_batch_size", str(args.val_batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--no_cbam",
        "--patience", str(args.patience),
        "--log_every", str(args.log_every),
        "--run_dir", run_dir,
    ]
    optional_pairs = [
        ("--eval_num_workers", args.eval_num_workers),
        ("--min_lr", args.min_lr),
    ]
    for flag, value in optional_pairs:
        if value is not None and flag in supported_flags:
            cmd.extend([flag, str(value)])
    if "--skip_final_test" in supported_flags:
        cmd.append("--skip_final_test")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run clean S2P2 WiFi+CMC+Mixup variants.")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    parser.add_argument("--results_root", default=os.path.join(RESULTS_ROOT, "S2P2_wifi_cmc_clean"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=1, help="1-based first job index")
    parser.add_argument("--end", type=int, default=8, help="1-based last job index")
    parser.add_argument("--force", action="store_true", help="Run even if history.json already exists")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.results_root)
    all_jobs = jobs()
    selected = [
        (idx, job) for idx, job in enumerate(all_jobs, 1)
        if args.start <= idx <= args.end
    ]

    print("=" * 80)
    print("S2P2 clean WiFi + CMC + Mixup sequential exploration")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Jobs: {len(selected)} / {len(all_jobs)}")
    print(f"Teacher: {args.teacher_ckpt}")
    print(f"Results root: {args.results_root}")
    print(f"Loader: batch={args.batch_size}, val_batch={args.val_batch_size}, workers={args.num_workers}")
    print("=" * 80)

    supported_flags, _ = get_supported_flags()
    summaries = []
    for idx, job in selected:
        run_dir = os.path.join(args.results_root, job["tag"])
        ensure_dir(run_dir)
        write_json(
            {"index": idx, "tag": job["tag"], "desc": job["desc"], "args": job["args"]},
            os.path.join(run_dir, "job_info.json"),
        )

        metric = read_best_val(run_dir)
        if metric is not None and not args.force:
            print(f"[{idx:02d}] skip existing {job['tag']} MPJPE={metric['mpjpe']:.2f}mm")
            summaries.append({"index": idx, "tag": job["tag"], "run_dir": run_dir, **metric})
            continue

        job_args, skipped = filter_supported_args(job["args"], supported_flags)
        cmd = build_base_cmd(args, run_dir, supported_flags) + job_args
        print("\n" + "=" * 80)
        print(f"[{idx:02d}/{len(all_jobs)}] {job['tag']}")
        print(job["desc"])
        if skipped:
            print(f"Skipped unsupported args: {', '.join(skipped)}")
        print(" ".join(cmd))
        print("=" * 80)

        if args.dry_run:
            continue

        returncode = run_command(cmd, seed=args.seed)
        metric = read_best_val(run_dir)
        if metric is None:
            print(f"[{idx:02d}] no validation metric, returncode={returncode}")
            summaries.append({
                "index": idx,
                "tag": job["tag"],
                "run_dir": run_dir,
                "returncode": returncode,
                "status": "failed_or_incomplete",
            })
        else:
            item = {"index": idx, "tag": job["tag"], "run_dir": run_dir, "returncode": returncode, **metric}
            summaries.append(item)
            print(
                f"[{idx:02d}] best val MPJPE={metric['mpjpe']:.2f}mm "
                f"PA={metric['pa_mpjpe']:.2f}mm epoch={metric['epoch']}"
            )

        ranked = sorted(
            [s for s in summaries if "mpjpe" in s],
            key=lambda row: row["mpjpe"],
        )
        write_json(summaries, os.path.join(args.results_root, "summary_all.json"))
        write_json(ranked, os.path.join(args.results_root, "summary_ranked.json"))

    ranked = sorted(
        [s for s in summaries if "mpjpe" in s],
        key=lambda row: row["mpjpe"],
    )
    print("\nTop validation candidates:")
    for item in ranked[:10]:
        print(
            f"{item['mpjpe']:.2f}mm | PA {item['pa_mpjpe']:.2f}mm | "
            f"epoch {item['epoch']} | {item['tag']} | {item['run_dir']}"
        )
    print(f"\nSaved summaries under: {args.results_root}")


if __name__ == "__main__":
    main()
