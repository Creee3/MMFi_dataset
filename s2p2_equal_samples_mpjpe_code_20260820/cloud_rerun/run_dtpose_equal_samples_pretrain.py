#!/usr/bin/env python3
"""Prepare and run the nine equal-sample DT-Pose pretraining jobs.

The original DT-Pose pretrainer is kept in ``pretrain.py``.  This launcher
only writes per-cell configurations, checks the data range, and invokes that
unchanged script through ``pretrain_equal_samples_entry.py``.

One pretraining checkpoint is produced for each protocol/split cell and is
then shared by the three pose-training seeds for that cell.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


TARGET_FRAME_START = 62
PRETRAIN_CURRENT_FRAME_END = 295
POSE_TARGET_FRAME_END = 296
PRETRAIN_FRAME_COUNT = PRETRAIN_CURRENT_FRAME_END - TARGET_FRAME_START + 1

PROTOCOL_TO_LABEL = {
    "protocol1": "P1",
    "protocol2": "P2",
    "protocol3": "P3",
}
LABEL_TO_PROTOCOL = {value: key for key, value in PROTOCOL_TO_LABEL.items()}

SPLIT_TO_LABEL = {
    "random_split": "S1",
    "cross_subject_split": "S2",
    "cross_scene_split": "S3",
}
LABEL_TO_SPLIT = {value: key for key, value in SPLIT_TO_LABEL.items()}

ALL_CELLS = [
    "P1-S1", "P2-S1", "P3-S1",
    "P1-S2", "P2-S2", "P3-S2",
    "P1-S3", "P2-S3", "P3-S3",
]

# Pose-stage counts are used to check that pretraining has exactly one fewer
# current-frame sample per sequence, while still touching frame 296 as the
# final next-frame target.
POSE_COUNTS = {
    "P1-S1": {"train": 105280, "validation": 26320},
    "P2-S1": {"train": 97760, "validation": 24440},
    "P3-S1": {"train": 203040, "validation": 50760},
    "P1-S2": {"train": 105280, "validation": 26320},
    "P2-S2": {"train": 97760, "validation": 24440},
    "P3-S2": {"train": 203040, "validation": 50760},
    "P1-S3": {"train": 98700, "validation": 32900},
    "P2-S3": {"train": 91650, "validation": 30550},
    "P3-S3": {"train": 190350, "validation": 63450},
}

WEIGHT_RELATIVE = {
    "P1-S1": "pretrain_weights/mmfi-csi/protocol1-s1/pretrain_our_ratio80.pt",
    "P2-S1": "pretrain_weights/mmfi-csi/protocol2-s1/pretrain_our_ratio80.pt",
    "P3-S1": "pretrain_weights/mmfi-csi/protocol3-s1/pretrain_our_ratio80.pt",
    "P1-S2": "pretrain_weights/mmfi-csi/protocol1-s2/pretrain_our_ratio80.pt",
    "P2-S2": "pretrain_weights/mmfi-csi/protocol2-s2/pretrain_our_ratio80.pt",
    "P3-S2": "pretrain_weights/mmfi-csi/protocol3-s2/pretrain_our_ratio80.pt",
    "P1-S3": "pretrain_weights/mmfi-csi/protocol1-s3/pretrain_our_ratio80.pt",
    "P2-S3": "pretrain_weights/mmfi-csi/protocol2-s3/pretrain_our_ratio80.pt",
    "P3-S3": "pretrain_weights/mmfi-csi/protocol3-s3/pretrain_our_ratio80.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run equal-sample DT-Pose MAE pretraining for all 3x3 cells."
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--template", default=None)
    cell_group = parser.add_mutually_exclusive_group()
    cell_group.add_argument("--cells", nargs="+", default=None)
    cell_group.add_argument("--all_cells", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_device_batch_size", type=int, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument(
        "--base_learning_rate", "--lr",
        dest="base_learning_rate",
        type=float,
        default=None,
        help="Base LR before DT-Pose's batch_size / 256 scaling.",
    )
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epoch", type=int, default=None)
    parser.add_argument(
        "--total_epoch", "--epochs",
        dest="total_epoch",
        type=int,
        default=None,
    )
    parser.add_argument("--skip_file_preflight", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


def normalize_cell(value: str) -> str:
    compact = "".join(ch for ch in value.upper() if ch.isalnum())
    if compact.startswith("P") and len(compact) == 4:
        protocol_label = compact[:2]
        split_label = "S" + compact[3]
    elif compact.startswith("S") and len(compact) == 4:
        split_label = compact[:2]
        protocol_label = "P" + compact[3]
    else:
        raise ValueError(f"Invalid cell {value!r}; use P2-S2, P2S2, or S2P2.")
    if protocol_label not in LABEL_TO_PROTOCOL or split_label not in LABEL_TO_SPLIT:
        raise ValueError(f"Unsupported cell {value!r}.")
    return f"{protocol_label}-{split_label}"


def requested_cells(args: argparse.Namespace) -> List[str]:
    if args.all_cells:
        cells = list(ALL_CELLS)
    elif args.cells is None:
        cells = list(ALL_CELLS)
    else:
        cells = [normalize_cell(value) for value in args.cells]
    if len(set(cells)) != len(cells):
        raise ValueError(f"Duplicate cells were requested: {cells}")
    return cells


def effective_hyperparameters(args: argparse.Namespace) -> dict:
    values = {
        "batch_size": 4096,
        "max_device_batch_size": 256,
        "mask_ratio": 0.80,
        "base_learning_rate": 1.5e-4,
        "weight_decay": 0.05,
        "warmup_epoch": 40,
        "total_epoch": 400,
    }
    for argument, key in (
        (args.batch_size, "batch_size"),
        (args.max_device_batch_size, "max_device_batch_size"),
        (args.mask_ratio, "mask_ratio"),
        (args.base_learning_rate, "base_learning_rate"),
        (args.weight_decay, "weight_decay"),
        (args.warmup_epoch, "warmup_epoch"),
        (args.total_epoch, "total_epoch"),
    ):
        if argument is not None:
            values[key] = argument

    if values["batch_size"] <= 0 or values["max_device_batch_size"] <= 0:
        raise ValueError("batch_size and max_device_batch_size must be positive")
    load_batch_size = min(
        values["batch_size"], values["max_device_batch_size"]
    )
    if values["batch_size"] % load_batch_size != 0:
        raise ValueError(
            "batch_size must be divisible by the effective micro-batch size "
            f"{load_batch_size}"
        )
    if values["base_learning_rate"] <= 0 or values["total_epoch"] <= 0:
        raise ValueError("base_learning_rate and total_epoch must be positive")
    if not 0.0 < values["mask_ratio"] < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if values["weight_decay"] < 0 or values["warmup_epoch"] < 0:
        raise ValueError("weight_decay and warmup_epoch must be non-negative")
    values["scaled_optimizer_learning_rate"] = (
        values["base_learning_rate"] * values["batch_size"] / 256
    )
    return values


def split_cell(cell_id: str) -> Tuple[str, str]:
    protocol_label, split_label = cell_id.split("-")
    return LABEL_TO_PROTOCOL[protocol_label], LABEL_TO_SPLIT[split_label]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_launcher_sources(project_root: Path) -> dict:
    manifest_path = project_root / "dtpose_pretrain_source_hashes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {}
    mismatches = []
    for relative, expected in manifest.get("files", {}).items():
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        actual[relative] = {
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
        }
        if actual_hash != expected.get("sha256") or path.stat().st_size != expected.get("bytes"):
            mismatches.append({
                "file": relative,
                "expected": expected,
                "actual": actual[relative],
            })
    if mismatches:
        raise RuntimeError(
            "DT-Pose pretraining source verification failed:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    print(f"DT-Pose pretraining source verification passed: {len(actual)} files")
    return {"manifest": str(manifest_path), "files": actual}


def resolve_template(project_root: Path, requested: Optional[str]) -> Path:
    candidates: List[Path] = []
    if requested:
        requested_path = Path(requested)
        candidates.extend([requested_path, project_root / requested_path])
    candidates.extend([
        project_root / "config/mmfi/pretrain_config.yaml",
    ])
    for path in candidates:
        path = path.resolve()
        if path.is_file():
            return path
    raise FileNotFoundError("No pretraining config template was found")


def make_cell_config(
    template: dict,
    cell_id: str,
    dataset_root: Path,
    output_root: Path,
    run_dir: Path,
    seed: int,
    hyperparameters: dict,
) -> dict:
    protocol, split = split_cell(cell_id)
    config = copy.deepcopy(template)
    config["protocol"] = protocol
    config["split_to_use"] = split
    config["data_unit"] = "frame"
    config["random_split"]["random_seed"] = 0
    config["random_split"]["ratio"] = 0.8
    config.update({
        "dataset_root": str(dataset_root),
        "seed": seed,
        "init_rand_seed": 0,
        "batch_size": hyperparameters["batch_size"],
        "max_device_batch_size": hyperparameters["max_device_batch_size"],
        "mask_ratio": hyperparameters["mask_ratio"],
        "base_learning_rate": hyperparameters["base_learning_rate"],
        "weight_decay": hyperparameters["weight_decay"],
        "warmup_epoch": hyperparameters["warmup_epoch"],
        "total_epoch": hyperparameters["total_epoch"],
        "dataset_name": "mmfi-csi",
        "experiment_name": f"{protocol}-{SPLIT_TO_LABEL[split].lower()}",
        "training_semi": False,
        "model_name": "our_ratio80",
        "save_path": str((output_root / "pretrain_weights").resolve()),
        "run_root": str(run_dir.resolve()),
        # A next-frame pair uses current 62..295 and next 63..296.  Thus the
        # union of CSI frames seen by pretraining is exactly 62..296.
        "target_frame_start": TARGET_FRAME_START,
        "target_frame_end": PRETRAIN_CURRENT_FRAME_END,
        "include_next_frame": True,
        "equal_sample_pretrain": True,
        "pose_target_frame_end": POSE_TARGET_FRAME_END,
        "pretrain_num_workers": "controlled by launcher",
        "cell_id": cell_id,
    })
    return config


def data_form_from_feeder(project_root: Path, config: dict) -> dict:
    project_text = str(project_root)
    if project_text not in sys.path:
        sys.path.insert(0, project_text)
    feeder = importlib.import_module("feeder.mmfi")
    return feeder.decode_config(config)


def count_form(data_form: dict) -> Tuple[int, int, set[Tuple[str, str]]]:
    sequence_count = 0
    pairs: set[Tuple[str, str]] = set()
    for subject, actions in data_form.items():
        sequence_count += len(actions)
        pairs.update((subject, action) for action in actions)
    return sequence_count * PRETRAIN_FRAME_COUNT, sequence_count, pairs


def scene_for_subject(subject: str) -> str:
    number = int(subject[1:])
    return f"E{(number - 1) // 10 + 1:02d}"


def check_required_files(
    dataset_root: Path,
    pairs: Iterable[Tuple[str, str]],
    file_cache: Dict[str, str],
) -> dict:
    missing: List[str] = []
    zero_size: List[str] = []

    def check(path: Path) -> None:
        key = os.fspath(path)
        status = file_cache.get(key)
        if status is None:
            if not path.is_file():
                status = "missing"
            elif path.stat().st_size == 0:
                status = "zero_size"
            else:
                status = "ok"
            file_cache[key] = status
        if status == "missing" and len(missing) < 20:
            missing.append(key)
        elif status == "zero_size" and len(zero_size) < 20:
            zero_size.append(key)

    for subject, action in sorted(set(pairs)):
        base = dataset_root / scene_for_subject(subject) / subject / action
        check(base / "ground_truth.npy")
        for index in range(TARGET_FRAME_START, POSE_TARGET_FRAME_END + 1):
            check(base / "wifi-csi" / f"frame{index + 1:03d}.mat")
        if len(missing) >= 20 or len(zero_size) >= 20:
            break
    if missing or zero_size:
        raise RuntimeError(
            "Dataset file preflight failed. "
            f"missing(first 20)={missing}, zero_size(first 20)={zero_size}"
        )
    return {
        "status": "passed",
        "unique_subject_action_pairs": len(set(pairs)),
        "cached_file_checks": len(file_cache),
    }


def preflight_cells(
    project_root: Path,
    dataset_root: Path,
    template: dict,
    cells: Sequence[str],
    check_files: bool,
) -> dict:
    file_cache: Dict[str, str] = {}
    records = {}
    for cell_id in cells:
        config = copy.deepcopy(template)
        protocol, split = split_cell(cell_id)
        config["protocol"] = protocol
        config["split_to_use"] = split
        config["target_frame_start"] = TARGET_FRAME_START
        config["target_frame_end"] = PRETRAIN_CURRENT_FRAME_END
        config["include_next_frame"] = True
        decoded = data_form_from_feeder(project_root, config)
        train_count, train_sequences, train_pairs = count_form(
            decoded["train_dataset"]["data_form"]
        )
        val_count, val_sequences, val_pairs = count_form(
            decoded["val_dataset"]["data_form"]
        )
        expected = POSE_COUNTS[cell_id]
        expected_train = expected["train"] - train_sequences
        expected_val = expected["validation"] - val_sequences
        if train_count != expected_train or val_count != expected_val:
            raise RuntimeError(
                f"{cell_id} pretrain count mismatch: "
                f"derived train={train_count}, validation={val_count}; "
                f"expected {expected_train}, {expected_val}"
            )
        file_record = {"status": "skipped"}
        if check_files:
            file_record = check_required_files(
                dataset_root, train_pairs | val_pairs, file_cache
            )
        records[cell_id] = {
            "cell_id": cell_id,
            "protocol": protocol,
            "split": split,
            "train_sequence_count": train_sequences,
            "validation_sequence_count": val_sequences,
            "train_samples": train_count,
            "validation_samples": val_count,
            "current_frame_range": [TARGET_FRAME_START, PRETRAIN_CURRENT_FRAME_END],
            "next_frame_range": [TARGET_FRAME_START + 1, POSE_TARGET_FRAME_END],
            "unique_frame_range": [TARGET_FRAME_START, POSE_TARGET_FRAME_END],
            "file_preflight": file_record,
        }
        print(
            f"{cell_id} pretrain preflight passed: train={train_count}, "
            f"validation={val_count}, sequences={train_sequences}/{val_sequences}"
        )
    return {
        "cells": records,
        "file_checks_cached": len(file_cache),
        "file_check": "performed" if check_files else "skipped",
    }


def require_cuda(project_root: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import torch; print('CUDA available:', torch.cuda.is_available()); "
        "raise SystemExit(0 if torch.cuda.is_available() else 2)",
    ]
    result = subprocess.run(command, cwd=str(project_root), check=False)
    if result.returncode != 0:
        raise RuntimeError("CUDA is not available in this Python environment")


def stream_process(command: List[str], cwd: Path, project_root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project_root)] + ([existing] if existing else [])
    )
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def validate_weight(path: Path, project_root: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"Missing or incomplete pretraining weight: {path}")
    # Loading the serialized object catches truncated transfers immediately.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import torch

    model = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(model, "encoder"):
        raise RuntimeError(f"Pretraining weight has no encoder: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_weight_manifest(
    output_root: Path,
    records: dict,
    source_integrity: dict,
    seed: int,
    project_root: Path,
) -> Path:
    files = {}
    for cell_id, record in records.items():
        path = output_root / WEIGHT_RELATIVE[cell_id]
        checked = validate_weight(path, project_root)
        files[cell_id] = {
            "path": WEIGHT_RELATIVE[cell_id],
            "sha256": checked["sha256"],
            "bytes": checked["bytes"],
            "pretrain_seed": seed,
            "source_run": record["run_dir"],
        }
    manifest = {
        "schema": 1,
        "reference": "equal-sample DT-Pose pretraining generated by this bundle",
        "scope": "one checkpoint per protocol/split cell",
        "current_frame_range": [TARGET_FRAME_START, PRETRAIN_CURRENT_FRAME_END],
        "next_frame_range": [TARGET_FRAME_START + 1, POSE_TARGET_FRAME_END],
        "unique_csi_frame_range": [TARGET_FRAME_START, POSE_TARGET_FRAME_END],
        "seed": seed,
        "files": files,
        "launcher_source_integrity": source_integrity,
    }
    path = output_root / "pretrain_source_hashes.json"
    write_json(path, manifest)
    return path


def manifest_signature(manifest: dict) -> Tuple:
    return (
        tuple(manifest.get("cells", [])),
        manifest.get("seed"),
        manifest.get("num_workers"),
        tuple(manifest.get("current_frame_range", [])),
        json.dumps(
            manifest.get("hyperparameters", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def main() -> None:
    args = parse_args()
    cells = requested_cells(args)
    hyperparameters = effective_hyperparameters(args)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.verify_only and (args.dry_run or args.resume):
        raise ValueError("--verify_only cannot be combined with --dry_run or --resume")
    if args.resume and not args.output_root:
        raise ValueError("--resume requires --output_root")

    project_root = Path(args.project_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    if not project_root.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("project_root or dataset_root does not exist")
    if not (project_root / "pretrain.py").is_file():
        raise FileNotFoundError(project_root / "pretrain.py")
    if not (project_root / "cloud_rerun/pretrain_equal_samples_entry.py").is_file():
        raise FileNotFoundError("pretrain_equal_samples_entry.py is missing")

    source_integrity = verify_launcher_sources(project_root)
    template_path = resolve_template(project_root, args.template)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if template.get("dataset_name") != "mmfi-csi":
        raise RuntimeError("The pretraining template must use dataset_name: mmfi-csi")
    dataset_manifest = preflight_cells(
        project_root,
        dataset_root,
        template,
        cells,
        check_files=not args.skip_file_preflight,
    )

    if args.verify_only:
        print(f"Verification only passed: {len(cells)} pretraining cells.")
        return

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = project_root / "strict_offline_runs" / f"dtpose_equal_samples_pretrain_{stamp}"
    manifest_path = output_root / "manifest.json"
    requested_manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "launcher": str(Path(__file__).resolve()),
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "cells": cells,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "requested_run_count": len(cells),
        "current_frame_range": [TARGET_FRAME_START, PRETRAIN_CURRENT_FRAME_END],
        "next_frame_range": [TARGET_FRAME_START + 1, POSE_TARGET_FRAME_END],
        "unique_csi_frame_range": [TARGET_FRAME_START, POSE_TARGET_FRAME_END],
        "model_name": "our_ratio80",
        "hyperparameter_overrides": {
            "batch_size": args.batch_size,
            "max_device_batch_size": args.max_device_batch_size,
            "mask_ratio": args.mask_ratio,
            "base_learning_rate": args.base_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_epoch": args.warmup_epoch,
            "total_epoch": args.total_epoch,
        },
        "hyperparameters": hyperparameters,
        "total_epoch": hyperparameters["total_epoch"],
        "batch_size": hyperparameters["batch_size"],
        "max_device_batch_size": hyperparameters["max_device_batch_size"],
        "mask_ratio": hyperparameters["mask_ratio"],
        "source_integrity": source_integrity,
        "dataset_preflight": dataset_manifest,
        "runs": [],
    }

    if args.resume:
        if not output_root.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"--resume requires {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_signature(manifest) != manifest_signature(requested_manifest):
            raise RuntimeError("Resume arguments do not match the existing manifest")
        old_records = {record["cell"]: record for record in manifest.get("runs", [])}
    else:
        if output_root.exists():
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Choose a new output_root or use --resume."
            )
        output_root.mkdir(parents=True, exist_ok=False)
        manifest = requested_manifest
        old_records = {}

    if not args.resume:
        for cell_id in cells:
            run_dir = output_root / cell_id
            run_dir.mkdir(parents=True, exist_ok=False)
            config = make_cell_config(
                template, cell_id, dataset_root, output_root, run_dir, args.seed,
                hyperparameters,
            )
            config_path = run_dir / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            manifest["runs"].append({
                "cell": cell_id,
                "run_dir": str(run_dir),
                "config": str(config_path),
                "weight": str(output_root / WEIGHT_RELATIVE[cell_id]),
                "hyperparameters": hyperparameters,
                "status": "prepared",
            })
        write_json(manifest_path, manifest)
    else:
        if set(old_records) != set(cells):
            raise RuntimeError("Existing manifest does not contain exactly the requested cells")

    if not args.dry_run and not args.allow_cpu:
        require_cuda(project_root)

    records = manifest["runs"]
    for record in records:
        cell_id = record["cell"]
        run_dir = Path(record["run_dir"])
        config_path = Path(record["config"])
        weight_path = output_root / WEIGHT_RELATIVE[cell_id]

        if args.resume and record.get("status") == "completed":
            record["result"] = validate_weight(weight_path, project_root)
            print(f"SKIP complete: {cell_id}")
            continue
        if args.dry_run:
            record["status"] = "dry_run"
            print(f"DRY RUN: {cell_id} -> {config_path}")
            write_json(manifest_path, manifest)
            continue

        print(f"===== DT-Pose equal-sample pretraining {cell_id} =====", flush=True)
        record["status"] = "running"
        write_json(manifest_path, manifest)
        command = [
            sys.executable,
            "-u",
            str(project_root / "cloud_rerun/pretrain_equal_samples_entry.py"),
            "--config_file",
            str(config_path),
            "--num_workers",
            str(args.num_workers),
        ]
        try:
            stream_process(command, run_dir, project_root, run_dir / "launcher.log")
            record["result"] = validate_weight(weight_path, project_root)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = repr(error)
            write_json(manifest_path, manifest)
            raise
        record["status"] = "completed"
        write_json(manifest_path, manifest)

    write_json(manifest_path, manifest)
    completed = sum(record.get("status") == "completed" for record in records)
    prepared = sum(record.get("status") in {"prepared", "dry_run"} for record in records)
    if completed == len(records):
        weight_manifest = write_weight_manifest(
            output_root,
            {record["cell"]: record for record in records},
            source_integrity,
            args.seed,
            project_root,
        )
        print(f"All pretraining cells completed; weight manifest={weight_manifest}")
    print(
        f"Finished request: total={len(records)}, completed={completed}, "
        f"not_started_or_dry_run={prepared}, output={output_root}"
    )


if __name__ == "__main__":
    main()
