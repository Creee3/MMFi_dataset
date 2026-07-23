"""Check MMFi S2P2 data files without starting model training."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml


PROTOCOL2_ACTIONS = [
    "A01", "A06", "A07", "A08", "A09", "A10", "A11",
    "A12", "A15", "A16", "A24", "A25", "A26",
]


def scene_for_subject(subject):
    number = int(subject[1:])
    if 1 <= number <= 10:
        return "E01"
    if 11 <= number <= 20:
        return "E02"
    if 21 <= number <= 30:
        return "E03"
    if 31 <= number <= 40:
        return "E04"
    raise ValueError(f"Unexpected subject: {subject}")


def read_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def short_issue(path, issue):
    return {"issue": issue, "path": str(path)}


def valid_csiamp(path):
    import scipy.io as scio

    mat = scio.loadmat(path)
    data = mat.get("CSIamp")
    if not isinstance(data, np.ndarray):
        return False, f"CSIamp is {type(data).__name__}"
    if data.ndim != 3 or data.shape[2] < 10:
        return False, f"CSIamp shape is {getattr(data, 'shape', None)}"
    if not np.issubdtype(data.dtype, np.number):
        return False, f"CSIamp dtype is {data.dtype}"
    return True, ""


def maybe_check_mat(path, mode, counter):
    if mode == "none":
        return None
    counter["seen"] += 1
    if mode == "sample" and counter["seen"] % counter["sample_every"] != 0:
        return None
    try:
        ok, detail = valid_csiamp(path)
    except Exception as exc:
        return short_issue(path, f"mat_read_error: {type(exc).__name__}: {exc}")
    if not ok:
        return short_issue(path, detail)
    counter["checked"] += 1
    return None


def valid_npy(path):
    # mmap_mode validates the .npy header and layout without copying every RGB
    # array into RAM. This is enough to catch the tokenizer/header failures
    # that can otherwise surface only part-way through a training epoch.
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(data, np.ndarray):
        return False, f"loaded {type(data).__name__}, expected ndarray"
    if not np.issubdtype(data.dtype, np.number):
        return False, f"dtype is {data.dtype}, expected numeric"
    if data.size == 0:
        return False, "array has zero elements"
    return True, ""


def maybe_check_npy(path, label, mode, counter):
    if mode == "none":
        return None
    counter["seen"] += 1
    if (
        counter["progress_every"] > 0
        and counter["seen"] % counter["progress_every"] == 0
    ):
        print(
            f"[npy-check] {label}: parsed {counter['seen']:,} files "
            f"({counter['checked']:,} validated)"
        )
    if mode == "sample" and counter["seen"] % counter["sample_every"] != 0:
        return None
    try:
        ok, detail = valid_npy(path)
    except Exception as exc:
        return short_issue(path, f"{label}_npy_read_error: {type(exc).__name__}: {exc}")
    if not ok:
        return short_issue(path, f"{label}_npy_invalid: {detail}")
    counter["checked"] += 1
    counter[f"{label}_checked"] += 1
    return None


def check_files(args):
    cfg = read_config(args.config_file)
    split = cfg["cross_subject_split"]
    subjects = (
        list(split["train_dataset"]["subjects"])
        + list(split["val_dataset"]["subjects"])
    )
    actions = PROTOCOL2_ACTIONS
    root = Path(args.dataset_root)

    counts = {
        "subjects": len(subjects),
        "actions": len(actions),
        "frames_per_action": args.frames,
        "expected_wifi": 0,
        "expected_rgb": 0,
        "missing": 0,
        "empty": 0,
        "mat_checked": 0,
        "mat_seen": 0,
        "npy_checked": 0,
        "npy_seen": 0,
        "ground_truth_checked": 0,
        "rgb_checked": 0,
    }
    issues = []
    mat_counter = {"seen": 0, "checked": 0, "sample_every": args.sample_every}
    npy_counter = {
        "seen": 0,
        "checked": 0,
        "sample_every": args.npy_sample_every,
        "progress_every": args.progress_every,
        "ground_truth_checked": 0,
        "rgb_checked": 0,
    }

    for subject in subjects:
        scene = scene_for_subject(subject)
        for action in actions:
            action_dir = root / scene / subject / action
            gt_path = action_dir / "ground_truth.npy"
            if not gt_path.is_file():
                counts["missing"] += 1
                issues.append(short_issue(gt_path, "missing_ground_truth"))
            elif gt_path.stat().st_size == 0:
                counts["empty"] += 1
                issues.append(short_issue(gt_path, "empty_ground_truth"))
            elif args.npy_scope in {"all", "ground_truth"}:
                issue = maybe_check_npy(
                    gt_path, "ground_truth", args.npy_check, npy_counter
                )
                if issue:
                    issues.append(issue)

            for frame_idx in range(1, args.frames + 1):
                wifi_path = action_dir / "wifi-csi" / f"frame{frame_idx:03d}.mat"
                rgb_path = action_dir / "rgb" / f"frame{frame_idx:03d}.npy"
                counts["expected_wifi"] += 1
                counts["expected_rgb"] += 1

                for path, label in ((wifi_path, "wifi"), (rgb_path, "rgb")):
                    if not path.is_file():
                        counts["missing"] += 1
                        issues.append(short_issue(path, f"missing_{label}"))
                        continue
                    if path.stat().st_size == 0:
                        counts["empty"] += 1
                        issues.append(short_issue(path, f"empty_{label}"))

                if wifi_path.is_file() and wifi_path.stat().st_size > 0:
                    issue = maybe_check_mat(wifi_path, args.mat_check, mat_counter)
                    if issue:
                        issues.append(issue)

                if (
                    args.npy_scope in {"all", "rgb"}
                    and rgb_path.is_file()
                    and rgb_path.stat().st_size > 0
                ):
                    issue = maybe_check_npy(
                        rgb_path, "rgb", args.npy_check, npy_counter
                    )
                    if issue:
                        issues.append(issue)

                if len(issues) >= args.max_issues:
                    break
            if len(issues) >= args.max_issues:
                break
        if len(issues) >= args.max_issues:
            break

    counts["mat_seen"] = mat_counter["seen"]
    counts["mat_checked"] = mat_counter["checked"]
    counts["npy_seen"] = npy_counter["seen"]
    counts["npy_checked"] = npy_counter["checked"]
    counts["ground_truth_checked"] = npy_counter["ground_truth_checked"]
    counts["rgb_checked"] = npy_counter["rgb_checked"]
    return counts, issues


def main():
    parser = argparse.ArgumentParser(
        description="Check S2P2 MMFi data paths and optional CSIamp contents."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--frames", type=int, default=297)
    parser.add_argument(
        "--mat_check",
        choices=["none", "sample", "full"],
        default="sample",
        help="none: only paths/sizes; sample: read every Nth mat; full: read all mats.",
    )
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument(
        "--npy_check",
        choices=["none", "sample", "full"],
        default="sample",
        help="Check ground-truth/RGB .npy files by actually parsing their headers.",
    )
    parser.add_argument(
        "--npy_scope",
        choices=["all", "ground_truth", "rgb"],
        default="all",
        help="Which .npy files to parse when --npy_check is enabled.",
    )
    parser.add_argument("--npy_sample_every", type=int, default=1000)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5000,
        help="Print NPY scan progress every N parsed files; 0 disables progress.",
    )
    parser.add_argument("--max_issues", type=int, default=30)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    counts, issues = check_files(args)
    report = {"counts": counts, "issues": issues}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
