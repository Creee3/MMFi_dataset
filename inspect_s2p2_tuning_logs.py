"""Summarize completed, interrupted, and failed S2P2 tuning seed logs."""

import argparse
import json
import os
import re
from collections import Counter


ERROR_RULES = [
    (
        re.compile(r"TypeError: unsupported operand type\(s\) for -: 'int' and 'Untokenizer'"),
        "NumPy .npy header/tokenize failure",
    ),
    (
        re.compile(r"DataLoader worker .* exited unexpectedly"),
        "PyTorch DataLoader worker exited",
    ),
    (
        re.compile(r"Couldn't open shared file mapping"),
        "Windows shared-memory mapping failure",
    ),
    (
        re.compile(r"returncode=3221225477"),
        "Windows process access-violation exit",
    ),
    (
        re.compile(r"Weights only load failed|Unsupported global: GLOBAL numpy\.core\.multiarray\.scalar"),
        "PyTorch 2.6 checkpoint loading default",
    ),
    (
        re.compile(r"TypeError: '_EnumDict' object is not callable"),
        "Windows Python worker-spawn environment failure",
    ),
    (
        re.compile(
            r"'slice' object is not subscriptable|"
            r"'range_iterator' object is not subscriptable|"
            r"bad argument to internal function"
        ),
        "Python/NumPy runtime integrity failure",
    ),
    (
        re.compile(r"Traceback \(most recent call last\):"),
        "Unhandled Python exception",
    ),
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def last_epoch(log_text):
    matches = re.findall(r"Epoch (\d+):", log_text)
    return int(matches[-1]) if matches else None


def history_progress(path):
    """Return the number of saved validation rows and their latest epoch."""
    if not os.path.isfile(path):
        return 0, None
    try:
        history = read_json(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return 0, None
    if not isinstance(history, list):
        return 0, None

    epochs = []
    for row in history:
        if not isinstance(row, dict):
            continue
        epoch = row.get("epoch")
        if isinstance(epoch, int):
            epochs.append(epoch)
    return len(history), max(epochs) if epochs else None


def classify_log(log_text):
    matches = []
    for pattern, label in ERROR_RULES:
        if pattern.search(log_text):
            matches.append(label)
    return matches


def relevant_error_lines(log_text, limit=4):
    lines = []
    for line in log_text.splitlines():
        if (
            "Error:" in line
            or "Exception" in line
            or "Traceback" in line
            or "returncode=" in line
        ):
            lines.append(line.strip())
    return lines[-limit:]


def seed_record(trial_name, seed_name, seed_dir, params):
    complete_path = os.path.join(seed_dir, "train_complete.json")
    log_path = os.path.join(seed_dir, "log.txt")
    process_log_path = os.path.join(seed_dir, "process_output.log")
    history_path = os.path.join(seed_dir, "history.json")

    complete = None
    if os.path.isfile(complete_path):
        try:
            complete = read_json(complete_path)
        except (OSError, json.JSONDecodeError):
            complete = None

    log_parts = []
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_parts.append(f.read())
    if os.path.isfile(process_log_path):
        with open(process_log_path, "r", encoding="utf-8", errors="replace") as f:
            log_parts.append(f.read())
    log_text = "\n".join(log_parts)

    history_epochs, history_last_epoch = history_progress(history_path)
    last_log_epoch = last_epoch(log_text)

    error_types = classify_log(log_text)
    if complete and complete.get("status") == "complete":
        status = "complete"
    elif error_types:
        status = "error"
    elif log_text:
        status = "interrupted"
    else:
        status = "no_log"

    return {
        "trial": trial_name,
        "seed": seed_name,
        "status": status,
        # history.json is the authoritative progress record. process_output.log
        # accumulates failed attempts, so its final epoch can be stale.
        "last_epoch": history_last_epoch or last_log_epoch,
        "history_epochs": history_epochs,
        "history_last_epoch": history_last_epoch,
        "last_log_epoch": last_log_epoch,
        "errors": error_types,
        "error_lines": relevant_error_lines(log_text),
        "params": params,
        "seed_dir": seed_dir,
    }


def collect_records(results_root):
    records = []
    for trial_name in sorted(os.listdir(results_root)):
        if not trial_name.startswith("trial_"):
            continue
        trial_dir = os.path.join(results_root, trial_name)
        if not os.path.isdir(trial_dir):
            continue

        params = {}
        expected_seeds = []
        info_path = os.path.join(trial_dir, "trial_info.json")
        result_path = os.path.join(trial_dir, "trial_result.json")
        trial_abandoned = False
        if os.path.isfile(result_path):
            try:
                trial_abandoned = read_json(result_path).get("status") == "abandoned"
            except (OSError, json.JSONDecodeError):
                pass
        if os.path.isfile(info_path):
            try:
                info = read_json(info_path)
                params = info.get("tuned_params", {})
                expected_seeds = [int(seed) for seed in info.get("seeds", [])]
            except (OSError, json.JSONDecodeError):
                pass

        if not expected_seeds:
            expected_seeds = sorted(
                int(name.split("_", 1)[1])
                for name in os.listdir(trial_dir)
                if name.startswith("seed_") and name.split("_", 1)[1].isdigit()
            )
        for seed in expected_seeds:
            seed_name = f"seed_{seed:02d}"
            seed_dir = os.path.join(trial_dir, seed_name)
            record = seed_record(trial_name, seed_name, seed_dir, params)
            if trial_abandoned and record["status"] != "complete":
                record["status"] = "abandoned"
            records.append(record)
    return records


def trial_decisions(records):
    by_trial = {}
    for record in records:
        by_trial.setdefault(record["trial"], []).append(record)

    completed_signatures = set()
    for trial_name, trial_records in by_trial.items():
        if trial_records and all(r["status"] == "complete" for r in trial_records):
            completed_signatures.add(
                json.dumps(trial_records[0]["params"], sort_keys=True)
            )

    decisions = []
    environment_errors = {
        "NumPy .npy header/tokenize failure",
        "PyTorch DataLoader worker exited",
        "Windows shared-memory mapping failure",
        "Windows process access-violation exit",
        "Windows Python worker-spawn environment failure",
        "Python/NumPy runtime integrity failure",
    }
    for trial_name, trial_records in sorted(by_trial.items()):
        params = trial_records[0]["params"] if trial_records else {}
        signature = json.dumps(params, sort_keys=True)
        statuses = [record["status"] for record in trial_records]
        errors = sorted(
            {error for record in trial_records for error in record["errors"]}
        )

        if trial_records and all(status == "complete" for status in statuses):
            action = "keep_valid"
            reason = "All seeds completed."
        elif signature in completed_signatures:
            action = "skip_duplicate"
            reason = "Same lr/batch-size already has a complete three-seed trial."
        elif set(errors) & environment_errors:
            action = "rerun_same_params"
            reason = "Failure is infrastructure-related, not a model score."
        elif any(status in {"interrupted", "no_log"} for status in statuses):
            action = "rerun_same_params"
            reason = "Training was interrupted before all seeds completed."
        else:
            action = "needs_review"
            reason = "No recognized infrastructure failure; inspect the seed log."

        decisions.append(
            {
                "trial": trial_name,
                "params": params,
                "seed_statuses": statuses,
                "errors": errors,
                "action": action,
                "reason": reason,
            }
        )
    return decisions


def main():
    parser = argparse.ArgumentParser(
        description="Find and summarize errors in S2P2 tuning trial logs."
    )
    parser.add_argument("results_root")
    parser.add_argument("--only_errors", action="store_true")
    parser.add_argument(
        "--report_name",
        default="log_diagnostics.json",
        help="JSON report filename written under results_root.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_root):
        raise FileNotFoundError(f"Results directory not found: {args.results_root}")

    records = collect_records(args.results_root)
    report_path = os.path.join(args.results_root, args.report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    decisions = trial_decisions(records)
    plan_path = os.path.join(args.results_root, "recovery_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    shown = [r for r in records if r["status"] != "complete"] if args.only_errors else records
    print(f"Scanned {len(records)} seed directories under: {args.results_root}")
    print("trial  seed     status       history log    lr          batch  issue")
    print("-" * 100)
    for record in shown:
        params = record["params"]
        lr = params.get("lr")
        lr_text = f"{lr:.3e}" if isinstance(lr, (float, int)) else "-"
        issue = "; ".join(record["errors"]) or "-"
        if record["status"] == "complete" and record["errors"]:
            issue = "historical retry: " + issue
        history_epoch = (
            record["history_last_epoch"]
            if record["history_last_epoch"] is not None
            else "-"
        )
        log_epoch = (
            record["last_log_epoch"]
            if record["last_log_epoch"] is not None
            else "-"
        )
        print(
            f"{record['trial'][:15]:15} {record['seed']:8} "
            f"{record['status']:12} {str(history_epoch):7} {str(log_epoch):6} "
            f"{lr_text:11} "
            f"{str(params.get('batch_size', '-')):6} {issue}"
        )
        if record["status"] == "complete" and record["errors"]:
            print("  Completion marker exists; listed errors are from earlier retry attempts.")
        else:
            for line in record["error_lines"]:
                print(f"  {line}")

    status_counts = Counter(record["status"] for record in records)
    error_counts = Counter(
        error for record in records for error in record["errors"]
    )
    print("\nSeed status:", dict(sorted(status_counts.items())))
    if error_counts:
        print("Historical error types (includes completed retries):", dict(error_counts.most_common()))
    print(f"JSON report: {report_path}")
    print(f"Recovery plan: {plan_path}")
    print("\nTrial-level decision:")
    for decision in decisions:
        params = decision["params"]
        print(
            f"{decision['trial']}: {decision['action']} | "
            f"lr={params.get('lr', '-')} bs={params.get('batch_size', '-')} | "
            f"{decision['reason']}"
        )


if __name__ == "__main__":
    main()
