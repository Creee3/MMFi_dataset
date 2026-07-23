"""Fresh S2P2 CMC learning-rate and batch-size tuning with Optuna."""

import argparse
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime


DEFAULT_RESULTS_ROOT = os.path.join(
    "strict_offline_runs", "S2P2_cmc_lr_bs_tuning_fresh"
)


def remove_user_site_from_sys_path():
    """Keep the Optuna parent in the same Conda-only runtime as its children."""
    os.environ["PYTHONNOUSERSITE"] = "1"
    try:
        import site

        user_sites = site.getusersitepackages()
    except (ImportError, AttributeError):
        return []

    if isinstance(user_sites, str):
        user_sites = [user_sites]
    normalized_user_sites = {
        os.path.normcase(os.path.abspath(path)) for path in user_sites
    }
    removed = []
    retained = []
    for path in sys.path:
        normalized = os.path.normcase(os.path.abspath(path or os.curdir))
        if normalized in normalized_user_sites:
            removed.append(path)
        else:
            retained.append(path)
    sys.path[:] = retained
    return removed


# ``PYTHONNOUSERSITE`` only takes effect while Python starts. The tuner itself
# may be launched from an interactive shell where AppData is already on
# sys.path, so remove it before Optuna (and any of its dependencies) imports.
REMOVED_USER_SITE_PATHS = remove_user_site_from_sys_path()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, value):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def completed_metric(seed_dir):
    complete_path = os.path.join(seed_dir, "train_complete.json")
    history_path = os.path.join(seed_dir, "history.json")
    best_path = os.path.join(seed_dir, "best.pth")
    if not (os.path.isfile(complete_path) and os.path.isfile(history_path) and os.path.isfile(best_path)):
        return None

    try:
        complete = read_json(complete_path)
        history = read_json(history_path)
    except (OSError, json.JSONDecodeError):
        return None
    if complete.get("status") != "complete" or not history:
        return None

    best = min(history, key=lambda row: float(row.get("mpjpe", float("inf"))))
    return {
        "epoch": int(best.get("epoch", -1)),
        "mpjpe": float(best["mpjpe"]) * 1000.0,
        "pa_mpjpe": float(best.get("pa_mpjpe", 0.0)) * 1000.0,
        "pck@20": float(best.get("pck@20", 0.0)),
        "pck@50": float(best.get("pck@50", 0.0)),
    }


def trial_dir(results_root, trial_number, params):
    digest = hashlib.sha1(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return os.path.join(results_root, f"trial_{trial_number:03d}_{digest}")


def training_env(seed=None):
    env = os.environ.copy()
    if seed is not None:
        env["TRAIN_SEED"] = str(seed)
    env["MMFI_PROTOCOL"] = "protocol2"
    env["HELDOUT_SPLIT_ENABLED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    # Prevent the Windows user-site NumPy from mixing with the Conda runtime.
    env["PYTHONNOUSERSITE"] = "1"
    # This training script does not use torch.compile; disabling Dynamo avoids
    # rare Windows/PyTorch optimizer-startup crashes during retry subprocesses.
    env["TORCHDYNAMO_DISABLE"] = "1"
    return env


def verify_numpy_runtime():
    probe = [
        sys.executable,
        "-c",
        "import numpy, sys; print(sys.executable); print(numpy.__version__); print(numpy.__file__)",
    ]
    result = subprocess.run(
        probe,
        env=training_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        raise RuntimeError(
            "NumPy cannot be imported with user site-packages disabled.\n" + output
        )
    if "appdata/roaming/python" in output.replace("\\", "/").lower():
        raise RuntimeError(
            "NumPy still resolves to the Windows user site instead of WiMANS2.\n"
            + output
        )
    print("NumPy runtime with user site disabled:")
    print(output)


def build_train_command(args, params, seed_dir, num_workers):
    command = [
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
        "--batch_size", str(params["batch_size"]),
        "--val_batch_size", str(args.val_batch_size),
        "--num_workers", str(num_workers),
        "--eval_num_workers", str(args.eval_num_workers),
        "--device", args.device,
        "--no_cbam",
        "--lr", str(params["lr"]),
        "--min_lr", str(args.min_lr),
        "--lr_patience", str(args.lr_patience),
        "--warmup_epochs", str(args.warmup_epochs),
        "--weight_decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--aug_noise", str(args.aug_noise),
        "--aug_freq_mask", str(args.aug_freq_mask),
        "--aug_time_mask", str(args.aug_time_mask),
        "--mixup_alpha", str(args.mixup_alpha),
        "--lambda_cmc", str(args.lambda_cmc),
        "--cmc_tau", str(args.cmc_tau),
        "--cmc_proj_dim", str(args.cmc_proj_dim),
        "--cmc_start_epoch", str(args.cmc_start_epoch),
        "--cmc_ramp_epochs", str(args.cmc_ramp_epochs),
        "--cmc_encoder_grad_scale", str(args.cmc_encoder_grad_scale),
        "--adaptive_cmc_factor", str(args.adaptive_cmc_factor),
        "--adaptive_cmc_min_scale", str(args.adaptive_cmc_min_scale),
        "--patience", str(args.patience),
        "--log_every", str(args.log_every),
        "--skip_final_test",
        "--resume_last",
        "--run_dir", seed_dir,
    ]
    if args.adaptive_cmc:
        command.append("--adaptive_cmc")
    return command


def worker_attempts(args):
    if args.worker_fallbacks is not None:
        return [args.num_workers] + args.worker_fallbacks
    return [args.num_workers] + [0] * args.seed_retries


def _write_runner_output(text, log_files):
    sys.stdout.write(text)
    sys.stdout.flush()
    for log_file in log_files:
        log_file.write(text)
        log_file.flush()


def _read_process_output(stream, output_queue):
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line)
    finally:
        output_queue.put(None)


def _terminate_process_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_command(
    command,
    seed,
    process_log_path,
    attempt_log_path=None,
    no_output_timeout=0,
):
    log_paths = [process_log_path]
    if attempt_log_path is not None:
        log_paths.append(attempt_log_path)
    log_files = [open(path, "a", encoding="utf-8") for path in log_paths]
    try:
        proc = subprocess.Popen(
            command,
            env=training_env(seed),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        output_queue = queue.Queue()
        reader = threading.Thread(
            target=_read_process_output,
            args=(proc.stdout, output_queue),
            daemon=True,
        )
        reader.start()

        last_output_at = time.monotonic()
        timed_out = False
        while True:
            try:
                line = output_queue.get(timeout=1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                if (
                    no_output_timeout > 0
                    and time.monotonic() - last_output_at >= no_output_timeout
                ):
                    timed_out = True
                    _write_runner_output(
                        "\n[runner] no output for "
                        f"{no_output_timeout}s; terminating hung child process tree.\n",
                        log_files,
                    )
                    _terminate_process_tree(proc)
                    break
                continue

            if line is None:
                if proc.poll() is not None:
                    break
                continue
            last_output_at = time.monotonic()
            _write_runner_output(line, log_files)

        if proc.poll() is None:
            _terminate_process_tree(proc)
        reader.join(timeout=2)
        returncode = 124 if timed_out else proc.returncode
        message = f"\n[runner] returncode={returncode}\n"
        _write_runner_output(message, log_files)
        return returncode
    finally:
        for log_file in log_files:
            log_file.close()


def has_nonretryable_error(process_log_path):
    if not os.path.isfile(process_log_path):
        return False
    try:
        with open(process_log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    # These indicate deterministic data, checkpoint, or CLI failures.
    # Windows access violations, worker-spawn faults, and CPython/NumPy
    # internal exceptions are deliberately retryable: the dataset checks are
    # clean, so they must use the configured worker fallback chain.
    patterns = [
        "Failed to load valid CSIamp",
        "CSIamp not found",
        "Weights only load failed",
        "Unsupported global: GLOBAL numpy.core.multiarray.scalar",
        "Teacher checkpoint not found",
        "unrecognized arguments:",
    ]
    return any(pattern in text for pattern in patterns)


def run_seed(args, trial_number, params, run_dir, seed, worker_counts=None):
    seed_dir = os.path.join(run_dir, f"seed_{seed:02d}")
    ensure_dir(seed_dir)

    existing = completed_metric(seed_dir)
    if existing is not None:
        print(f"[trial {trial_number:03d} seed {seed}] existing complete result reused.")
        existing["seed"] = seed
        existing["run_dir"] = seed_dir
        return existing

    process_log_path = os.path.join(seed_dir, "process_output.log")
    attempts = worker_counts or worker_attempts(args)
    for attempt, workers in enumerate(attempts):
        command = build_train_command(args, params, seed_dir, workers)
        command_name = (
            "command.txt"
            if attempt == 0
            else f"command_retry_{attempt:02d}_workers{workers}.txt"
        )
        write_command(os.path.join(seed_dir, command_name), command)
        print_trial_seed_header(trial_number, seed, params, command)
        if attempt > 0:
            print(
                f"[trial {trial_number:03d} seed {seed}] "
                f"restart {attempt}/{len(attempts) - 1} with num_workers={workers}"
            )

        attempt_log_path = os.path.join(
            seed_dir, f"process_output_attempt_{attempt:02d}_workers{workers}.log")
        returncode = run_command(
            command,
            seed,
            process_log_path,
            attempt_log_path,
            no_output_timeout=args.attempt_no_output_timeout,
        )
        metric = completed_metric(seed_dir)
        if returncode == 0 and metric is not None:
            break
        print(
            f"[trial {trial_number:03d} seed {seed}] "
            f"attempt {attempt + 1}/{len(attempts)} did not complete "
            f"(returncode={returncode}, complete_marker={metric is not None})."
        )
        if has_nonretryable_error(attempt_log_path):
            print(
                f"[trial {trial_number:03d} seed {seed}] "
                "non-retryable data/runtime error detected; stop instead of "
                "restarting the seed from epoch 1."
            )
            return None
        if attempt == len(attempts) - 1:
            print(
                f"[trial {trial_number:03d} seed {seed}] "
                f"failed after {len(attempts)} full attempts."
            )
            return None

    metric["seed"] = seed
    metric["run_dir"] = seed_dir
    print(
        f"[trial {trial_number:03d} seed {seed}] "
        f"MPJPE={metric['mpjpe']:.2f}mm PA={metric['pa_mpjpe']:.2f}mm "
        f"epoch={metric['epoch']}"
    )
    return metric


def write_command(path, command):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(command) + "\n")


def print_trial_seed_header(trial_number, seed, params, command):
    print("\n" + "=" * 80)
    print(f"Trial {trial_number:03d} | seed {seed}")
    print(json.dumps(params, indent=2))
    print(" ".join(command))
    print("=" * 80)


def mean(values):
    return sum(values) / len(values)


def std(values):
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def aggregate(metrics, expected_seeds):
    if len(metrics) != expected_seeds:
        return {
            "status": "incomplete",
            "n_completed": len(metrics),
            "n_expected": expected_seeds,
            "seed_metrics": metrics,
        }
    mpjpes = [metric["mpjpe"] for metric in metrics]
    pa_mpjpes = [metric["pa_mpjpe"] for metric in metrics]
    return {
        "status": "complete",
        "n_completed": expected_seeds,
        "n_expected": expected_seeds,
        "mpjpe_mean": mean(mpjpes),
        "mpjpe_std": std(mpjpes),
        "pa_mpjpe_mean": mean(pa_mpjpes),
        "pa_mpjpe_std": std(pa_mpjpes),
        "seed_metrics": metrics,
    }


def run_trial(args, trial_number, params, worker_counts=None):
    run_dir = trial_dir(args.results_root, trial_number, params)
    ensure_dir(run_dir)
    write_json(
        os.path.join(run_dir, "trial_info.json"),
        {
            "trial": trial_number,
            "tuned_params": params,
            "seeds": list(range(args.seeds)),
            "objective": "mean_val_mpjpe",
            "protocol": "protocol2",
            "split": "cross_subject_split",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    metrics = []
    for seed in range(args.seeds):
        metric = run_seed(
            args,
            trial_number,
            params,
            run_dir,
            seed,
            worker_counts=worker_counts,
        )
        if metric is None:
            result = aggregate(metrics, args.seeds)
            write_json(os.path.join(run_dir, "trial_result.json"), result)
            raise RuntimeError(
                f"Trial {trial_number:03d} stopped: seed {seed} failed repeatedly. "
                "No new hyperparameter trial will be started."
            )
        metrics.append(metric)

    result = aggregate(metrics, args.seeds)
    write_json(os.path.join(run_dir, "trial_result.json"), result)
    if result["status"] == "complete":
        print(
            f"[trial {trial_number:03d}] mean MPJPE="
            f"{result['mpjpe_mean']:.2f}+/-{result['mpjpe_std']:.2f}mm"
        )
    else:
        print(
            f"[trial {trial_number:03d}] incomplete "
            f"({result['n_completed']}/{result['n_expected']} seeds)."
        )
    return result


def completed_trials(results_root):
    rows = []
    if not os.path.isdir(results_root):
        return rows
    for name in sorted(os.listdir(results_root)):
        if not name.startswith("trial_"):
            continue
        run_dir = os.path.join(results_root, name)
        info_path = os.path.join(run_dir, "trial_info.json")
        result_path = os.path.join(run_dir, "trial_result.json")
        if not (os.path.isfile(info_path) and os.path.isfile(result_path)):
            continue
        try:
            info = read_json(info_path)
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") != "complete":
            continue
        rows.append(
            {
                "trial": int(info["trial"]),
                "run_dir": run_dir,
                "tuned_params": info["tuned_params"],
                **result,
            }
        )
    return sorted(rows, key=lambda row: row["mpjpe_mean"])


def mark_abandoned_trials(args):
    """Exclude explicitly selected infrastructure-failed trials from recovery."""
    requested = set(args.abandon_trials)
    if not requested:
        return

    found = set()
    for name in sorted(os.listdir(args.results_root)):
        if not name.startswith("trial_"):
            continue
        run_dir = os.path.join(args.results_root, name)
        info_path = os.path.join(run_dir, "trial_info.json")
        if not os.path.isfile(info_path):
            continue
        try:
            info = read_json(info_path)
            trial_number = int(info["trial"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if trial_number not in requested:
            continue

        result_path = os.path.join(run_dir, "trial_result.json")
        try:
            result = read_json(result_path) if os.path.isfile(result_path) else {}
        except (OSError, json.JSONDecodeError):
            result = {}
        result.update(
            {
                "status": "abandoned",
                "reason": "Explicitly excluded after repeated infrastructure failures.",
                "abandoned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        write_json(result_path, result)
        found.add(trial_number)
        print(
            f"Trial {trial_number:03d} marked abandoned: it will not be "
            "recovered or counted as a valid candidate."
        )

    missing = sorted(requested - found)
    if missing:
        raise ValueError(
            "Could not find requested trial directories: "
            + ", ".join(str(number) for number in missing)
        )


def incomplete_trials(results_root):
    """Return trial directories whose three-seed result is not complete yet."""
    rows = []
    if not os.path.isdir(results_root):
        return rows
    for name in sorted(os.listdir(results_root)):
        if not name.startswith("trial_"):
            continue
        run_dir = os.path.join(results_root, name)
        info_path = os.path.join(run_dir, "trial_info.json")
        if not os.path.isfile(info_path):
            continue
        try:
            info = read_json(info_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        result_path = os.path.join(run_dir, "trial_result.json")
        result = None
        if os.path.isfile(result_path):
            try:
                result = read_json(result_path)
            except (OSError, json.JSONDecodeError):
                pass
        if result and result.get("status") in {"complete", "abandoned"}:
            continue
        params = info.get("tuned_params")
        trial_number = info.get("trial")
        if not isinstance(trial_number, int) or not isinstance(params, dict):
            continue
        if "lr" not in params or "batch_size" not in params:
            continue
        rows.append({
            "trial": trial_number,
            "params": params,
            "run_dir": run_dir,
        })
    return sorted(rows, key=lambda row: row["trial"])


def recovered_result_registered(study, source_trial):
    return any(
        trial.user_attrs.get("recovered_from_trial") == source_trial
        for trial in study.trials
    )


def register_recovered_result(optuna, study, args, result):
    """Feed a repaired failed Optuna trial back into later TPE suggestions."""
    source_trial = int(result["trial"])
    if recovered_result_registered(study, source_trial):
        return
    distributions = {
        "lr": optuna.distributions.FloatDistribution(
            args.lr_low, args.lr_high, log=True
        ),
        "batch_size": optuna.distributions.CategoricalDistribution(args.batch_sizes),
    }
    recovered = optuna.trial.create_trial(
        params=result["tuned_params"],
        distributions=distributions,
        value=float(result["mpjpe_mean"]),
        user_attrs={"recovered_from_trial": source_trial},
    )
    study.add_trial(recovered)
    print(
        f"Recovered trial {source_trial:03d} registered for subsequent "
        "Optuna suggestions."
    )


def write_summaries(args):
    ranked = completed_trials(args.results_root)
    write_json(os.path.join(args.results_root, "summary_ranked.json"), ranked)
    write_json(
        os.path.join(args.results_root, "summary_all.json"),
        sorted(ranked, key=lambda row: row["trial"]),
    )
    if ranked:
        write_json(os.path.join(args.results_root, "best_trial.json"), ranked[0])
    return ranked


def run_optuna(args):
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError("Optuna is required for this tuner.") from exc

    storage_path = os.path.abspath(os.path.join(args.results_root, "optuna_study.db"))
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=args.search_seed),
        storage="sqlite:///" + storage_path.replace("\\", "/"),
        load_if_exists=True,
    )

    stale = [trial for trial in study.trials if trial.state == TrialState.RUNNING]
    for trial in stale:
        study.tell(trial.number, state=TrialState.FAIL)
    if study.trials:
        study.sampler.reseed_rng()

    pending = incomplete_trials(args.results_root)
    if pending:
        print(f"Recovering {len(pending)} incomplete trial(s) before new suggestions.")
    for pending_trial in pending:
        trial_number = pending_trial["trial"]
        params = pending_trial["params"]
        print(
            f"Recovering trial {trial_number:03d}: "
            f"lr={params['lr']:.3e}, batch_size={params['batch_size']}"
        )
        recovery_workers = None
        if args.recovery_num_workers is not None:
            recovery_workers = [args.recovery_num_workers, args.recovery_num_workers]
            print(
                f"Recovery worker override: num_workers={args.recovery_num_workers}"
            )
        result = run_trial(
            args,
            trial_number,
            params,
            worker_counts=recovery_workers,
        )
        if result.get("status") != "complete":
            raise RuntimeError(
                f"Recovery of trial {trial_number:03d} did not finish. "
                "No new hyperparameter trial will be started."
            )
        recovered_row = {
            "trial": trial_number,
            "tuned_params": params,
            **result,
        }
        register_recovered_result(optuna, study, args, recovered_row)
        write_summaries(args)

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", args.lr_low, args.lr_high, log=True),
            "batch_size": trial.suggest_categorical("batch_size", args.batch_sizes),
        }
        result = run_trial(args, trial.number, params)
        write_summaries(args)
        return result["mpjpe_mean"]

    ranked = write_summaries(args)
    remaining = max(0, args.n_trials - len(ranked))
    print(
        f"Valid candidates: {len(ranked)} | "
        f"target: {args.n_trials} | remaining: {remaining}"
    )
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return write_summaries(args)


def print_ranking(rows):
    if not rows:
        print("No complete three-seed trial yet.")
        return
    print("\nTop validation candidates:")
    for row in rows[:10]:
        params = row["tuned_params"]
        print(
            f"{row['mpjpe_mean']:.2f}+/-{row['mpjpe_std']:.2f}mm | "
            f"PA {row['pa_mpjpe_mean']:.2f}+/-{row['pa_mpjpe_std']:.2f}mm | "
            f"trial {row['trial']:03d} | "
            f"lr={params['lr']:.2e} bs={params['batch_size']}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fresh S2P2 CMC tuning: Optuna over lr and batch size."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--study_name", default="s2p2_cmc_lr_bs_tuning")
    parser.add_argument("--n_trials", type=int, default=16)
    parser.add_argument(
        "--abandon_trials",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Explicitly exclude repeated infrastructure-failed trial numbers. "
            "Excluded trials do not count toward --n_trials."
        ),
    )
    parser.add_argument("--search_seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--seed_retries",
        type=int,
        default=2,
        help="Extra full restarts with num_workers=0 when --worker_fallbacks is omitted.",
    )
    parser.add_argument(
        "--worker_fallbacks",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Retry worker counts after the initial --num_workers attempt. "
            "Example: --num_workers 8 --worker_fallbacks 4 2 0 0"
        ),
    )
    parser.add_argument(
        "--recovery_num_workers",
        type=int,
        default=None,
        help=(
            "Worker count used only when completing an existing incomplete "
            "trial; new Optuna trials still use --num_workers."
        ),
    )
    parser.add_argument(
        "--attempt_no_output_timeout",
        type=int,
        default=600,
        help=(
            "Kill a hung child training process after this many seconds with "
            "no output, then continue its worker fallback. Set 0 to disable."
        ),
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=0.0)

    parser.add_argument("--lr_low", type=float, default=1e-4)
    parser.add_argument("--lr_high", type=float, default=5e-3)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[16, 32, 64, 128, 256])

    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--aug_noise", type=float, default=0.08)
    parser.add_argument("--aug_freq_mask", type=float, default=0.15)
    parser.add_argument("--aug_time_mask", type=float, default=0.15)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--lambda_cmc", type=float, default=0.1)
    parser.add_argument("--cmc_tau", type=float, default=0.1)
    parser.add_argument("--cmc_proj_dim", type=int, default=128)
    parser.add_argument("--cmc_start_epoch", type=int, default=4)
    parser.add_argument("--cmc_ramp_epochs", type=int, default=10)
    parser.add_argument("--cmc_encoder_grad_scale", type=float, default=0.25)
    parser.add_argument("--adaptive_cmc", action="store_true")
    parser.add_argument("--adaptive_cmc_factor", type=float, default=0.85)
    parser.add_argument("--adaptive_cmc_min_scale", type=float, default=0.35)
    return parser


def main():
    args = build_parser().parse_args()
    if REMOVED_USER_SITE_PATHS:
        print(
            "[runner] removed user-site package path(s) from the Optuna parent: "
            + "; ".join(REMOVED_USER_SITE_PATHS)
        )
    else:
        print("[runner] Optuna parent and training children use Conda packages only.")
    if args.recovery_num_workers is not None and args.recovery_num_workers < 0:
        raise ValueError("--recovery_num_workers must be non-negative")
    if args.attempt_no_output_timeout < 0:
        raise ValueError("--attempt_no_output_timeout must be non-negative")
    ensure_dir(args.results_root)
    if not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_ckpt}")
    mark_abandoned_trials(args)

    write_json(
        os.path.join(args.results_root, "search_space.json"),
        {
            "tuned": {
                "lr": [args.lr_low, args.lr_high],
                "batch_sizes": args.batch_sizes,
            },
            "fixed": {
                "epochs": args.epochs,
                "seeds": args.seeds,
                "abandon_trials": args.abandon_trials,
                "seed_retries": args.seed_retries,
                "worker_attempts": worker_attempts(args),
                "recovery_num_workers": args.recovery_num_workers,
                "attempt_no_output_timeout": args.attempt_no_output_timeout,
                "patience": args.patience,
                "mixup_alpha": args.mixup_alpha,
                "lambda_cmc": args.lambda_cmc,
                "cmc_start_epoch": args.cmc_start_epoch,
                "cmc_ramp_epochs": args.cmc_ramp_epochs,
            },
        },
    )

    print("=" * 80)
    print("Fresh S2P2 WiFi + RGB-privileged CMC tuning")
    print("Protocol: protocol2 | Split: cross_subject_split | heldout split: disabled")
    print(f"Objective: mean validation MPJPE over {args.seeds} seeds")
    print(f"Trials: {args.n_trials} | epochs: {args.epochs} | early stopping: {args.patience}")
    print(f"Current-seed worker attempts: {' -> '.join(str(v) for v in worker_attempts(args))}")
    print(f"CMC: start epoch {args.cmc_start_epoch}, ramp {args.cmc_ramp_epochs}")
    print(f"Results: {args.results_root}")
    print("=" * 80)

    verify_numpy_runtime()
    ranking = run_optuna(args)
    print_ranking(ranking)
    print(f"\nSaved summaries under: {args.results_root}")


if __name__ == "__main__":
    main()
