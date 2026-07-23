"""Rebuild the S2P2 Optuna study from completed seed-level results.

This utility never changes model checkpoints or dataset files. It is intended
for a run where individual seeds were repaired outside the normal Optuna
runner, leaving the result directories ahead of optuna_study.db.
"""

import argparse
import gc
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_TRIALS = list(range(8))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def mean(values):
    return sum(values) / len(values)


def population_std(values):
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def require_complete_seed(seed_dir, expected_epochs):
    complete_path = seed_dir / "train_complete.json"
    history_path = seed_dir / "history.json"
    checkpoint_path = seed_dir / "best.pth"
    if not (complete_path.is_file() and history_path.is_file() and checkpoint_path.is_file()):
        raise ValueError(f"Incomplete seed artifacts: {seed_dir}")

    complete = read_json(complete_path)
    history = read_json(history_path)
    if complete.get("status") != "complete":
        raise ValueError(f"Seed has no complete marker: {seed_dir}")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Seed history is empty: {seed_dir}")

    metric_rows = [row for row in history if isinstance(row, dict) and "mpjpe" in row]
    if not metric_rows:
        raise ValueError(f"Seed history has no MPJPE rows: {seed_dir}")
    last_epoch = max(int(row.get("epoch", 0)) for row in metric_rows)
    if expected_epochs > 0 and last_epoch < expected_epochs:
        raise ValueError(
            f"Seed history ends at epoch {last_epoch}, expected {expected_epochs}: {seed_dir}"
        )

    best = min(metric_rows, key=lambda row: float(row["mpjpe"]))
    pck = {
        "pck@20": float(best.get("pck@20", 0.0)),
        "pck@50": float(best.get("pck@50", 0.0)),
    }
    return {
        "epoch": int(best.get("epoch", -1)),
        "mpjpe": float(best["mpjpe"]) * 1000.0,
        "pa_mpjpe": float(best.get("pa_mpjpe", 0.0)) * 1000.0,
        **pck,
        "last_epoch": last_epoch,
        "run_dir": str(seed_dir),
    }


def collect_trial(trial_dir, expected_epochs):
    info_path = trial_dir / "trial_info.json"
    if not info_path.is_file():
        raise ValueError(f"Missing trial_info.json: {trial_dir}")
    info = read_json(info_path)
    trial_number = int(info["trial"])
    params = info.get("tuned_params", {})
    if not isinstance(params, dict) or "lr" not in params or "batch_size" not in params:
        raise ValueError(f"Invalid tuned params: {trial_dir}")

    seeds = info.get("seeds", [0, 1, 2])
    if sorted(seeds) != [0, 1, 2]:
        raise ValueError(f"Expected seeds [0, 1, 2]: {trial_dir}")

    metrics = []
    for seed in seeds:
        seed_dir = trial_dir / f"seed_{int(seed):02d}"
        metric = require_complete_seed(seed_dir, expected_epochs)
        metric["seed"] = int(seed)
        metrics.append(metric)

    mpjpes = [metric["mpjpe"] for metric in metrics]
    pa_mpjpes = [metric["pa_mpjpe"] for metric in metrics]
    result = {
        "status": "complete",
        "n_completed": len(metrics),
        "n_expected": len(seeds),
        "mpjpe_mean": mean(mpjpes),
        "mpjpe_std": population_std(mpjpes),
        "pa_mpjpe_mean": mean(pa_mpjpes),
        "pa_mpjpe_std": population_std(pa_mpjpes),
        "seed_metrics": metrics,
    }
    return {
        "trial": trial_number,
        "trial_dir": trial_dir,
        "params": {"lr": float(params["lr"]), "batch_size": int(params["batch_size"])},
        "result": result,
    }


def find_trial_dirs(results_root):
    found = {}
    for trial_dir in sorted(results_root.glob("trial_*")):
        if not trial_dir.is_dir() or not (trial_dir / "trial_info.json").is_file():
            continue
        number = int(read_json(trial_dir / "trial_info.json")["trial"])
        if number in found:
            raise ValueError(f"Multiple directories declare trial {number}: {found[number]}, {trial_dir}")
        found[number] = trial_dir
    return found


def print_records(records):
    print("trial  lr           batch  mean_mpjpe  std    seed_best_epochs")
    print("-" * 74)
    for record in records:
        result = record["result"]
        epochs = ",".join(str(metric["epoch"]) for metric in result["seed_metrics"])
        print(
            f"{record['trial']:03d}    {record['params']['lr']:.3e}  "
            f"{record['params']['batch_size']:5d}  "
            f"{result['mpjpe_mean']:10.2f}  {result['mpjpe_std']:5.2f}  {epochs}"
        )


def write_result_artifacts(results_root, records):
    rows = []
    for record in records:
        write_json(record["trial_dir"] / "trial_result.json", record["result"])
        rows.append(
            {
                "trial": record["trial"],
                "run_dir": str(record["trial_dir"]),
                "tuned_params": record["params"],
                **record["result"],
            }
        )

    ranked = sorted(rows, key=lambda row: row["mpjpe_mean"])
    write_json(results_root / "summary_ranked.json", ranked)
    write_json(results_root / "summary_all.json", sorted(rows, key=lambda row: row["trial"]))
    if ranked:
        write_json(results_root / "best_trial.json", ranked[0])


def sqlite_url(path):
    return "sqlite:///" + str(path.resolve()).replace("\\", "/")


def remove_sqlite_sidecars(path):
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def backup_existing_database(database_path):
    if not database_path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = database_path.with_name(
        f"{database_path.stem}.before_rebuild_{timestamp}{database_path.suffix}"
    )
    shutil.copy2(database_path, backup)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(database_path) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, Path(str(backup) + suffix))
    return backup


def rebuild_study(results_root, records, args):
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError("Optuna is required to rebuild the study.") from exc

    database_path = results_root / "optuna_study.db"
    temporary_path = results_root / "optuna_study.rebuild_tmp.db"
    if temporary_path.exists():
        temporary_path.unlink()
    remove_sqlite_sidecars(temporary_path)

    distributions = {
        "lr": optuna.distributions.FloatDistribution(
            args.lr_low, args.lr_high, log=True
        ),
        "batch_size": optuna.distributions.CategoricalDistribution(args.batch_sizes),
    }
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=args.search_seed),
        storage=sqlite_url(temporary_path),
    )
    for record in records:
        params = record["params"]
        if not args.lr_low <= params["lr"] <= args.lr_high:
            raise ValueError(f"Trial {record['trial']} lr falls outside the configured search range.")
        if params["batch_size"] not in args.batch_sizes:
            raise ValueError(f"Trial {record['trial']} batch size is outside the configured choices.")
        study.add_trial(
            optuna.trial.create_trial(
                state=TrialState.COMPLETE,
                value=record["result"]["mpjpe_mean"],
                params=params,
                distributions=distributions,
            )
        )

    expected_numbers = list(range(len(records)))
    actual_numbers = [trial.number for trial in study.trials]
    if actual_numbers != expected_numbers:
        raise RuntimeError(f"Unexpected rebuilt Optuna trial numbers: {actual_numbers}")

    # Close SQLite connections before replacing the live study database.
    try:
        study._storage.remove_session()
    except AttributeError:
        pass
    del study
    gc.collect()

    backup = backup_existing_database(database_path)
    remove_sqlite_sidecars(database_path)
    os.replace(temporary_path, database_path)
    remove_sqlite_sidecars(temporary_path)
    return backup, database_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Rebuild S2P2 trial summaries and Optuna bookkeeping from complete seeds."
    )
    parser.add_argument("results_root")
    parser.add_argument("--apply", action="store_true", help="Write results and replace the backed-up Optuna database.")
    parser.add_argument("--study_name", default="s2p2_cmc_lr_bs_tuning")
    parser.add_argument("--search_seed", type=int, default=41)
    parser.add_argument("--lr_low", type=float, default=1e-4)
    parser.add_argument("--lr_high", type=float, default=5e-3)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--expected_epochs", type=int, default=30)
    parser.add_argument("--expected_trials", nargs="+", type=int, default=DEFAULT_TRIALS)
    return parser


def main():
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    trial_dirs = find_trial_dirs(results_root)
    expected = sorted(set(args.expected_trials))
    missing = [number for number in expected if number not in trial_dirs]
    if missing:
        raise ValueError("Missing expected trial directories: " + ", ".join(map(str, missing)))

    records = [collect_trial(trial_dirs[number], args.expected_epochs) for number in expected]
    records.sort(key=lambda record: record["trial"])
    if [record["trial"] for record in records] != expected:
        raise RuntimeError("Collected trial numbers do not match the requested sequence.")

    print_records(records)
    print(f"\nComplete trials to register: {', '.join(str(record['trial']) for record in records)}")
    print(f"Next Optuna trial after rebuild: {len(records):03d}")
    if not args.apply:
        print("\nDry run only. No files were changed. Re-run with --apply after all rows look correct.")
        return

    backup, database_path = rebuild_study(results_root, records, args)
    write_result_artifacts(results_root, records)
    print("\nBookkeeping rebuild complete.")
    if backup:
        print(f"Previous study database backup: {backup}")
    print(f"Active rebuilt study database: {database_path}")
    print("trial_result.json and summary files now include the registered trials.")


if __name__ == "__main__":
    main()
