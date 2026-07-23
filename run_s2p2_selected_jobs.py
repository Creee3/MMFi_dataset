"""
Run selected S2P2 WiFi + CMC + Mixup jobs sequentially.

Default selected jobs:
  01_cmc010_mix02_ramp
  04_latecmc_mix02
  05_tau007_mix02
  07_proj256_mix02
  08_adaptive_cmc_mix02

This script reuses run_s2p2_wifi_cmc_mix20.py so the job definitions stay in
one place. It skips completed runs unless --force is given.
"""

import argparse
import os

from run_s2p2_wifi_cmc_mix20 import (
    DEFAULT_TEACHER,
    RESULTS_ROOT,
    build_base_cmd,
    ensure_dir,
    filter_supported_args,
    get_supported_flags,
    jobs,
    read_best_val,
    run_command,
    write_json,
)


DEFAULT_SELECTED = "1,4,5,7,8"


def parse_job_ids(text):
    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            ids.extend(range(int(left), int(right) + 1))
        else:
            ids.append(int(part))
    return sorted(dict.fromkeys(ids))


def main():
    parser = argparse.ArgumentParser(
        description="Run selected S2P2 WiFi+CMC+Mixup variants.")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    parser.add_argument(
        "--results_root",
        default=os.path.join(RESULTS_ROOT, "S2P2_wifi_cmc_mix20"),
    )
    parser.add_argument("--jobs", default=DEFAULT_SELECTED,
                        help="Comma/range list, e.g. 1,4,5,7-8")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--scheduler", default="plateau",
                        choices=["plateau", "cosine"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.results_root)
    selected_ids = set(parse_job_ids(args.jobs))
    all_jobs = jobs()
    selected = [
        (idx, job)
        for idx, job in enumerate(all_jobs, 1)
        if idx in selected_ids
    ]
    if not selected:
        raise SystemExit(f"No valid jobs selected from: {args.jobs}")

    print("=" * 80)
    print("S2P2 selected WiFi + CMC + Mixup exploration")
    print(f"Jobs: {', '.join(str(idx) for idx, _ in selected)}")
    print(f"Teacher: {args.teacher_ckpt}")
    print(f"Results root: {args.results_root}")
    print(
        f"Loader: batch={args.batch_size}, val_batch={args.val_batch_size}, "
        f"workers={args.num_workers}, eval_workers={args.eval_num_workers}"
    )
    print("=" * 80)

    supported_flags, _ = get_supported_flags()
    summaries = []

    for idx, job in selected:
        run_dir = os.path.join(args.results_root, job["tag"])
        ensure_dir(run_dir)
        write_json(
            {
                "index": idx,
                "tag": job["tag"],
                "desc": job["desc"],
                "args": job["args"],
                "selected_runner": True,
            },
            os.path.join(run_dir, "job_info.json"),
        )

        metric = read_best_val(run_dir)
        if metric is not None and not args.force:
            print(f"[{idx:02d}] skip existing {job['tag']} "
                  f"MPJPE={metric['mpjpe']:.2f}mm")
            summaries.append({
                "index": idx,
                "tag": job["tag"],
                "run_dir": run_dir,
                **metric,
            })
            continue

        job_args, skipped = filter_supported_args(job["args"], supported_flags)
        cmd = build_base_cmd(args, run_dir, supported_flags) + job_args

        print("\n" + "=" * 80)
        print(f"[{idx:02d}] {job['tag']}")
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
            summaries.append({
                "index": idx,
                "tag": job["tag"],
                "run_dir": run_dir,
                "returncode": returncode,
                "status": "failed_or_incomplete",
            })
            print(f"[{idx:02d}] no validation metric, returncode={returncode}")
        else:
            summaries.append({
                "index": idx,
                "tag": job["tag"],
                "run_dir": run_dir,
                "returncode": returncode,
                **metric,
            })
            print(f"[{idx:02d}] best val MPJPE={metric['mpjpe']:.2f}mm "
                  f"PA={metric['pa_mpjpe']:.2f}mm epoch={metric['epoch']}")

        ranked = sorted(
            [s for s in summaries if "mpjpe" in s],
            key=lambda row: row["mpjpe"],
        )
        write_json(
            summaries,
            os.path.join(args.results_root, "summary_selected_all.json"),
        )
        write_json(
            ranked,
            os.path.join(args.results_root, "summary_selected_ranked.json"),
        )

    ranked = sorted(
        [s for s in summaries if "mpjpe" in s],
        key=lambda row: row["mpjpe"],
    )
    print("\nSelected validation candidates:")
    for item in ranked:
        print(
            f"{item['mpjpe']:.2f}mm | PA {item['pa_mpjpe']:.2f}mm | "
            f"epoch {item['epoch']} | {item['tag']} | {item['run_dir']}"
        )
    print(f"\nSaved selected summaries under: {args.results_root}")


if __name__ == "__main__":
    main()
