#!/usr/bin/env python3
"""Run the complete MM-Fi P1/P2/P3 x S1/S2/S3 baseline grid.

This launcher deliberately lives beside, rather than inside, the locked S2-P2
launcher.  The model files and the historical trainer are treated as source
artifacts.  The launcher only selects a protocol/split cell, writes an
isolated configuration, and invokes ``train_pose.py``.

The default request is still one cell (P2-S2).  Use ``--all_cells`` for the
complete 3 x 3 grid.  With the default three models and three seeds that is
81 independent training runs.
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
TARGET_FRAME_END = 296
TARGET_FRAME_COUNT = TARGET_FRAME_END - TARGET_FRAME_START + 1

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

PROTOCOL_ACTIONS = {
    "protocol1": [
        "A02", "A03", "A04", "A05", "A13", "A14", "A17", "A18",
        "A19", "A20", "A21", "A22", "A23", "A27",
    ],
    "protocol2": [
        "A01", "A06", "A07", "A08", "A09", "A10", "A11", "A12",
        "A15", "A16", "A24", "A25", "A26",
    ],
    "protocol3": [
        f"A{index:02d}" for index in range(1, 28)
    ],
}

# These counts are the canonical target-frame counts for this bundle.  The
# runner also derives them from feeder.mmfi.decode_config() and rejects a
# disagreement, so a changed split cannot silently alter the experiment.
EXPECTED_COUNTS = {
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

MODEL_SPECS = {
    "metafi": {
        "label": "MetaFi++",
        "experiment_name": "metafi",
        "batch_size": 16,
        "max_device_batch_size": 16,
        "learning_rate": 1.85e-3,
        "pretrained": False,
        "weight_stage": "pose_scratch",
    },
    "hpeli": {
        "label": "HPE-Li",
        "experiment_name": "hpeli",
        "batch_size": 16,
        "max_device_batch_size": 16,
        "learning_rate": 1.0e-3,
        "pretrained": False,
        "weight_stage": "pose_scratch",
    },
    "dtpose": {
        "label": "DT-Pose",
        "experiment_name": "DT-Pose-ratio80",
        "batch_size": 32,
        "max_device_batch_size": 16,
        "learning_rate": 8.49e-4,
        "pretrained": True,
        "weight_stage": "pose_pretrain",
    },
}

# The filenames match the equal-sample pretraining runner's output exactly.
# Historical bundle weights, including the old P2-S2 ``_org`` file, are not
# accepted by this 3x3 runner.
DTPOSE_PRETRAIN_FILES = {
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

ALL_CELLS = [
    "P1-S1", "P2-S1", "P3-S1",
    "P1-S2", "P2-S2", "P3-S2",
    "P1-S3", "P2-S3", "P3-S3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete MM-Fi 3x3 baseline grid with MPJPE selection."
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--dtpose_pretrain_root",
        default=None,
        help=(
            "Root containing pretrain_source_hashes.json and the canonical "
            "pretrain_weights tree. Defaults to project_root."
        ),
    )
    parser.add_argument("--template", default=None)
    cell_group = parser.add_mutually_exclusive_group()
    cell_group.add_argument(
        "--cells", nargs="+", default=None,
        help="Cells such as P2-S2 S1P1. Default: P2-S2.",
    )
    cell_group.add_argument(
        "--all_cells", action="store_true",
        help="Run P1/P2/P3 x S1/S2/S3 (27 runs per model).",
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_num_workers", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the configured batch size for every selected model.",
    )
    parser.add_argument(
        "--max_device_batch_size",
        type=int,
        default=None,
        help="Override the single-device micro-batch size for every selected model.",
    )
    parser.add_argument(
        "--learning_rate", "--lr",
        dest="learning_rate",
        type=float,
        default=None,
        help="Override the learning rate for every selected model.",
    )
    parser.add_argument(
        "--total_epoch", "--epochs",
        dest="total_epoch",
        type=int,
        default=None,
        help="Override the training epoch count for every selected model.",
    )
    parser.add_argument(
        "--skip_file_preflight", action="store_true",
        help="Only derive and verify sample counts; skip per-file existence checks.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Create all cell/model/seed configs and manifest, but do not train.",
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Verify source hashes, weights, split counts and data files only.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an existing output_root and skip complete runs.",
    )
    parser.add_argument(
        "--allow_cpu", action="store_true",
        help="Allow train_pose.py to run without CUDA (GPU is required by default).",
    )
    return parser.parse_args()


def normalize_cell(value: str) -> str:
    compact = "".join(ch for ch in value.upper() if ch.isalnum())
    if compact.startswith("P") and len(compact) == 4:
        protocol_label = compact[:2]
        setting_label = "S" + compact[3]
    elif compact.startswith("S") and len(compact) == 4:
        setting_label = compact[:2]
        protocol_label = "P" + compact[3]
    else:
        raise ValueError(
            f"Invalid cell {value!r}; use P2-S2, P2S2, or S2P2."
        )
    if protocol_label not in LABEL_TO_PROTOCOL or setting_label not in LABEL_TO_SPLIT:
        raise ValueError(f"Unsupported cell {value!r}.")
    return f"{protocol_label}-{setting_label}"


def requested_cells(args: argparse.Namespace) -> List[str]:
    if args.all_cells:
        cells = list(ALL_CELLS)
    elif args.cells is None:
        cells = ["P2-S2"]
    else:
        cells = [normalize_cell(value) for value in args.cells]
    if len(set(cells)) != len(cells):
        raise ValueError(f"Duplicate cells were requested: {cells}")
    return cells


def effective_model_specs(args: argparse.Namespace) -> Dict[str, dict]:
    """Apply optional CLI overrides without mutating the formal defaults."""
    specs: Dict[str, dict] = {}
    for model_key in args.models:
        spec = copy.deepcopy(MODEL_SPECS[model_key])
        spec["total_epoch"] = 50
        for argument, key in (
            (args.batch_size, "batch_size"),
            (args.max_device_batch_size, "max_device_batch_size"),
            (args.learning_rate, "learning_rate"),
            (args.total_epoch, "total_epoch"),
        ):
            if argument is not None:
                spec[key] = argument

        if spec["batch_size"] <= 0 or spec["max_device_batch_size"] <= 0:
            raise ValueError("batch_size and max_device_batch_size must be positive")
        if spec["learning_rate"] <= 0 or spec["total_epoch"] <= 0:
            raise ValueError("learning_rate and total_epoch must be positive")
        load_batch_size = min(spec["batch_size"], spec["max_device_batch_size"])
        if spec["batch_size"] % load_batch_size != 0:
            raise ValueError(
                f"{model_key}: batch_size={spec['batch_size']} must be divisible by "
                f"the effective micro-batch size {load_batch_size}"
            )
        specs[model_key] = spec
    return specs


def serializable_hyperparameters(specs: Dict[str, dict]) -> dict:
    keys = (
        "batch_size", "max_device_batch_size", "learning_rate", "total_epoch"
    )
    return {
        model_key: {key: spec[key] for key in keys}
        for model_key, spec in specs.items()
    }


def split_cell(cell_id: str) -> Tuple[str, str]:
    protocol_label, setting_label = cell_id.split("-")
    return LABEL_TO_PROTOCOL[protocol_label], LABEL_TO_SPLIT[setting_label]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_template(project_root: Path, requested: Optional[str]) -> Path:
    candidates: List[Path] = []
    if requested:
        requested_path = Path(requested)
        candidates.extend([requested_path, project_root / requested_path])
    candidates.extend([
        project_root / "config/mmfi/pose_config-Copy1.yaml",
        project_root / "config/mmfi/pose_config.yaml",
    ])
    for path in candidates:
        path = path.resolve()
        if path.is_file():
            return path
    raise FileNotFoundError("No pose config template was found")


def verify_model_sources(project_root: Path) -> dict:
    manifest_path = project_root / "model_source_hashes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing model integrity manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {})
    if not expected:
        raise RuntimeError("The model integrity manifest has no files")

    model_root = project_root / "model"
    actual_files = sorted(
        str(path.relative_to(project_root)).replace(os.sep, "/")
        for path in model_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".ipynb_checkpoints" not in path.parts
    )
    expected_files = sorted(expected)
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        raise RuntimeError(
            "Model source file list changed. "
            f"missing={missing}, extra={extra}"
        )

    actual_hashes = {}
    mismatches = []
    for relative, expected_hash in expected.items():
        path = project_root / relative
        actual_hash = sha256_file(path)
        actual_hashes[relative] = actual_hash
        if actual_hash != expected_hash:
            mismatches.append({
                "file": relative,
                "expected": expected_hash,
                "actual": actual_hash,
            })
    if mismatches:
        raise RuntimeError(
            "Model source hash verification failed:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    print(f"Model source verification passed: {len(actual_hashes)} files")
    return {"manifest": str(manifest_path), "files": actual_hashes}


def verify_pretrained_sources(
    project_root: Path,
    pretrain_root: Optional[Path] = None,
) -> dict:
    weights_root = project_root if pretrain_root is None else pretrain_root.resolve()
    manifest_path = weights_root / "pretrain_source_hashes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing DT-Pose weight manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("reference") != (
        "equal-sample DT-Pose pretraining generated by this bundle"
    ):
        raise RuntimeError(
            "The supplied DT-Pose weights are not marked as equal-sample "
            "pretraining from this bundle. Do not use the historical "
            "project_root/pretrain_weights for the equal-sample 3x3 run."
        )
    expected = manifest.get("files", {})
    if sorted(expected) != sorted(DTPOSE_PRETRAIN_FILES):
        raise RuntimeError("DT-Pose weight manifest does not contain all nine cells")
    actual = {}
    mismatches = []
    for cell_id, relative in DTPOSE_PRETRAIN_FILES.items():
        record = expected[cell_id]
        if record.get("path") != relative:
            raise RuntimeError(
                f"DT-Pose weight mapping changed for {cell_id}: {record.get('path')}"
            )
        path = weights_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing DT-Pose weight for {cell_id}: {path}")
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"DT-Pose weight size mismatch for {cell_id}: "
                f"expected {expected_bytes}, got {path.stat().st_size}"
            )
        actual_hash = sha256_file(path)
        actual[cell_id] = {
            "path": relative,
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
        }
        if actual_hash != record.get("sha256"):
            mismatches.append({
                "cell": cell_id,
                "path": relative,
                "expected": record.get("sha256"),
                "actual": actual_hash,
            })
    if mismatches:
        raise RuntimeError(
            "DT-Pose pretraining weight verification failed:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    print(f"DT-Pose pretraining weight verification passed: {len(actual)} files")
    return {"root": str(weights_root), "manifest": str(manifest_path), "files": actual}


def scene_for_subject(subject: str) -> str:
    number = int(subject[1:])
    return f"E{(number - 1) // 10 + 1:02d}"


def make_cell_config(template: dict, cell_id: str) -> dict:
    protocol, split = split_cell(cell_id)
    config = copy.deepcopy(template)
    config["protocol"] = protocol
    config["split_to_use"] = split
    # S1 is the canonical action-wise random split.  The training seed must
    # not silently change the validation set, so keep its split seed at zero.
    if "random_split" not in config:
        raise RuntimeError("The template has no random_split section")
    config["random_split"]["random_seed"] = 0
    config["random_split"]["ratio"] = 0.8
    config["cell_id"] = cell_id
    return config


def data_form_from_feeder(project_root: Path, config: dict) -> dict:
    # Import the bundle's own decoder, rather than reimplementing S1's
    # action-wise random split in the launcher.
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
    return sequence_count * TARGET_FRAME_COUNT, sequence_count, pairs


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
        for index in range(TARGET_FRAME_START, TARGET_FRAME_END + 1):
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
    cell_records = {}
    for cell_id in cells:
        config = make_cell_config(template, cell_id)
        decoded = data_form_from_feeder(project_root, config)
        train_count, train_sequences, train_pairs = count_form(
            decoded["train_dataset"]["data_form"]
        )
        val_count, val_sequences, val_pairs = count_form(
            decoded["val_dataset"]["data_form"]
        )
        expected = EXPECTED_COUNTS[cell_id]
        if train_count != expected["train"] or val_count != expected["validation"]:
            raise RuntimeError(
                f"{cell_id} sample count mismatch: derived train={train_count}, "
                f"validation={val_count}; expected {expected}"
            )
        file_record = {"status": "skipped"}
        if check_files:
            file_record = check_required_files(
                dataset_root, train_pairs | val_pairs, file_cache
            )
        protocol, split = split_cell(cell_id)
        cell_records[cell_id] = {
            "cell_id": cell_id,
            "protocol": protocol,
            "protocol_label": PROTOCOL_TO_LABEL[protocol],
            "split": split,
            "setting_label": SPLIT_TO_LABEL[split],
            "protocol_actions": PROTOCOL_ACTIONS[protocol],
            "train_subject_count": len(decoded["train_dataset"]["data_form"]),
            "validation_subject_count": len(decoded["val_dataset"]["data_form"]),
            "train_sequence_count": train_sequences,
            "validation_sequence_count": val_sequences,
            "train_samples": train_count,
            "validation_samples": val_count,
            "target_frame_start": TARGET_FRAME_START,
            "target_frame_end": TARGET_FRAME_END,
            "target_frame_count_per_sequence": TARGET_FRAME_COUNT,
            "file_preflight": file_record,
        }
        print(
            f"{cell_id} preflight passed: train={train_count}, "
            f"validation={val_count}, sequences={train_sequences}/{val_sequences}"
        )
    return {
        "cells": cell_records,
        "file_checks_cached": len(file_cache),
        "file_check": "performed" if check_files else "skipped",
    }


def make_run_config(
    template: dict,
    cell_id: str,
    spec: dict,
    model_key: str,
    seed: int,
    project_root: Path,
    dataset_root: Path,
    run_dir: Path,
    dtpose_pretrain_root: Path,
    num_workers: int,
    eval_num_workers: int,
) -> dict:
    config = make_cell_config(template, cell_id)
    pretrained_relative = None
    if spec["pretrained"]:
        pretrained_relative = DTPOSE_PRETRAIN_FILES[cell_id]
        pretrained_path = (dtpose_pretrain_root / pretrained_relative).resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(pretrained_path)
        pretrained_value = str(pretrained_path)
    else:
        pretrained_value = None

    protocol, split = split_cell(cell_id)
    config.update({
        "protocol": protocol,
        "split_to_use": split,
        "dataset_root": str(dataset_root),
        "seed": seed,
        "init_rand_seed": seed,
        "batch_size": spec["batch_size"],
        "max_device_batch_size": spec["max_device_batch_size"],
        "base_learning_rate": spec["learning_rate"],
        "total_epoch": spec["total_epoch"],
        "dataset_name": "mmfi-csi",
        "num_person": 1,
        "setting": f"{protocol}-{SPLIT_TO_LABEL[split].lower()}",
        "experiment_name": spec["experiment_name"],
        "training_semi": False,
        "pretrained_model_path": pretrained_value,
        "save_path": str((run_dir / "weights").resolve()),
        "run_root": str(run_dir.resolve()),
        "num_workers": num_workers,
        "eval_num_workers": eval_num_workers,
        "target_frame_start": TARGET_FRAME_START,
        "target_frame_end": TARGET_FRAME_END,
        "include_next_frame": False,
        "checkpoint_selection": "mpjpe",
        "save_epoch_features": False,
        "rerun_model_key": model_key,
        "rerun_equal_target_samples": True,
        "checkpoint_selection_scope": "validation",
        "cell_id": cell_id,
        "pretrained_source_relative": pretrained_relative,
        "pretrained_source_root": str(dtpose_pretrain_root.resolve()),
    })
    return config


def expected_weights_dir(run_dir: Path, config: dict, spec: dict) -> Path:
    return (
        run_dir / "weights" / "mmfi-csi" / config["setting"]
        / spec["weight_stage"] / spec["experiment_name"]
    )


def validate_completed_run(run_dir: Path, config: dict, spec: dict) -> dict:
    weights_dir = expected_weights_dir(run_dir, config, spec)
    expected = [
        weights_dir / "pose_pampjpe.pt",
        weights_dir / "pose_mpjpe.pt",
        weights_dir / "selection.json",
        weights_dir / "pose_pck10.pt",
        weights_dir / "pose_pck20.pt",
        weights_dir / "pose_pck30.pt",
        weights_dir / "pose_pck40.pt",
        weights_dir / "pose_pck50.pt",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{spec['label']} {config['cell_id']} seed {config['seed']} "
            f"finished without files: {missing}"
        )
    selection = json.loads(
        (weights_dir / "selection.json").read_text(encoding="utf-8")
    )
    if selection.get("selection_metric") != "mpjpe":
        raise RuntimeError(
            f"{spec['label']} {config['cell_id']} seed {config['seed']} "
            f"did not use MPJPE selection: {selection.get('selection_metric')!r}"
        )
    if selection.get("selected_epoch") is None:
        raise RuntimeError("Completed run has no selected epoch")
    return {
        "weights_dir": str(weights_dir),
        "selection": selection,
        "checkpoints": [str(path) for path in expected],
    }


def stream_process(command: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    pythonpath_entries = [str(cwd)]
    existing_pythonpath = environment.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
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


def require_cuda(project_root: Path) -> None:
    command = [
        sys.executable, "-c",
        "import torch; "
        "print('CUDA available:', torch.cuda.is_available()); "
        "raise SystemExit(0 if torch.cuda.is_available() else 2)",
    ]
    result = subprocess.run(command, cwd=str(project_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Use the 5090 environment or explicitly pass --allow_cpu."
        )


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def manifest_signature(manifest: dict) -> Tuple:
    return (
        tuple(manifest.get("cells", [])),
        tuple(manifest.get("models", [])),
        tuple(manifest.get("seeds", [])),
        manifest.get("num_workers"),
        manifest.get("eval_num_workers"),
        manifest.get("checkpoint_selection"),
        manifest.get("dtpose_pretrain_root"),
        json.dumps(
            manifest.get("effective_hyperparameters", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def main() -> None:
    args = parse_args()
    cells = requested_cells(args)
    specs = effective_model_specs(args)
    if args.num_workers < 0 or (args.eval_num_workers is not None and args.eval_num_workers < 0):
        raise ValueError("num_workers and eval_num_workers must be non-negative")
    if not args.models or not args.seeds:
        raise ValueError("At least one model and one seed are required")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Duplicate models were requested")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    if args.verify_only and (args.dry_run or args.resume):
        raise ValueError("--verify_only cannot be combined with --dry_run or --resume")
    if args.resume and not args.output_root:
        raise ValueError("--resume requires --output_root")

    eval_num_workers = (
        args.num_workers if args.eval_num_workers is None
        else args.eval_num_workers
    )
    project_root = Path(args.project_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    if "dtpose" in args.models and args.dtpose_pretrain_root is None:
        raise ValueError(
            "DT-Pose was selected. Pass --dtpose_pretrain_root pointing to "
            "the completed equal-sample pretraining output directory."
        )
    dtpose_pretrain_root = (
        project_root
        if args.dtpose_pretrain_root is None
        else Path(args.dtpose_pretrain_root).resolve()
    )
    train_script = project_root / "train_pose.py"
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    source_manifest = verify_model_sources(project_root)
    if "dtpose" in args.models:
        pretrain_manifest = verify_pretrained_sources(
            project_root, dtpose_pretrain_root
        )
    else:
        pretrain_manifest = {
            "status": "not_required",
            "reason": "DT-Pose was not selected",
        }
    template_path = resolve_template(project_root, args.template)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if template.get("modality") != "wifi-csi":
        raise RuntimeError("The template must use modality: wifi-csi")
    dataset_manifest = preflight_cells(
        project_root,
        dataset_root,
        template,
        cells,
        check_files=not args.skip_file_preflight,
    )

    if args.verify_only:
        print(
            f"Verification only passed: {len(cells)} cells, "
            f"{len(cells) * len(args.models) * len(args.seeds)} requested runs."
        )
        return

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = project_root / "strict_offline_runs" / f"mmfi_3x3_mpjpe_{stamp}"

    requested_manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "bundle": str(project_root),
        "dataset_root": str(dataset_root),
        "cells": cells,
        "models": list(args.models),
        "seeds": list(args.seeds),
        "requested_run_count": len(cells) * len(args.models) * len(args.seeds),
        "protocol_grid": "P1/P2/P3 x S1/S2/S3",
        "target_frame_range": [TARGET_FRAME_START, TARGET_FRAME_END],
        "checkpoint_selection": "mpjpe",
        "num_workers": args.num_workers,
        "eval_num_workers": eval_num_workers,
        "dtpose_pretrain_root": str(dtpose_pretrain_root),
        "hyperparameter_overrides": {
            "batch_size": args.batch_size,
            "max_device_batch_size": args.max_device_batch_size,
            "learning_rate": args.learning_rate,
            "total_epoch": args.total_epoch,
        },
        "effective_hyperparameters": serializable_hyperparameters(specs),
        "source_integrity": source_manifest,
        "pretrain_integrity": pretrain_manifest,
        "dataset_preflight": dataset_manifest,
        "runs": [],
    }
    manifest_path = output_root / "manifest.json"

    if args.resume:
        if not output_root.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"--resume requires an existing manifest: {manifest_path}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_signature(existing) != manifest_signature(requested_manifest):
            raise RuntimeError(
                "Resume arguments do not match the existing manifest. "
                "Use the same cells, models, seeds and worker counts."
            )
        manifest = existing
        old_records = {
            (record["cell"], record["model"], record["seed"]): record
            for record in manifest.get("runs", [])
        }
    else:
        if output_root.exists():
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Choose a new output_root or use --resume."
            )
        output_root.mkdir(parents=True, exist_ok=False)
        manifest = requested_manifest
        old_records = {}

    # Prepare every config before the first training process.  This makes the
    # 81-run plan inspectable even if the machine is stopped midway.
    if not args.resume:
        for cell_id in cells:
            for model_key in args.models:
                spec = specs[model_key]
                for seed in args.seeds:
                    run_dir = output_root / cell_id / model_key / f"seed_{seed:02d}"
                    run_dir.mkdir(parents=True, exist_ok=False)
                    config = make_run_config(
                        template, cell_id, spec, model_key, seed,
                        project_root, dataset_root, run_dir,
                        dtpose_pretrain_root,
                        args.num_workers, eval_num_workers,
                    )
                    config_path = run_dir / "config.yaml"
                    config_path.write_text(
                        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                        encoding="utf-8",
                    )
                    manifest["runs"].append({
                        "cell": cell_id,
                        "model": model_key,
                        "label": spec["label"],
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "config": str(config_path),
                        "hyperparameters": serializable_hyperparameters(
                            {model_key: spec}
                        )[model_key],
                        "status": "prepared",
                    })
        write_json(manifest_path, manifest)
    else:
        # Reuse the existing configs and records; never regenerate a partial
        # run with different settings during resume.
        expected_keys = {
            (cell_id, model_key, seed)
            for cell_id in cells for model_key in args.models for seed in args.seeds
        }
        if set(old_records) != expected_keys:
            raise RuntimeError("Existing manifest does not contain exactly the requested runs")

    if not args.dry_run and not args.allow_cpu:
        require_cuda(project_root)

    records = manifest["runs"]
    for record in records:
        cell_id = record["cell"]
        model_key = record["model"]
        seed = record["seed"]
        spec = specs[model_key]
        run_dir = Path(record["run_dir"])
        config_path = Path(record["config"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        if args.resume and record.get("status") == "completed":
            # Validate the files on disk instead of trusting a stale status.
            result = validate_completed_run(run_dir, config, spec)
            record.update(result)
            print(f"SKIP complete: {cell_id} / {spec['label']} / seed {seed}")
            continue

        if args.dry_run:
            record["status"] = "dry_run"
            print(f"DRY RUN: {cell_id} / {spec['label']} / seed {seed} -> {config_path}")
            write_json(manifest_path, manifest)
            continue

        print(f"===== {cell_id} / {spec['label']} / seed {seed} =====", flush=True)
        record["status"] = "running"
        write_json(manifest_path, manifest)
        command = [
            sys.executable,
            "-u",
            str(train_script),
            "--config_file",
            str(config_path),
            "--num_workers",
            str(args.num_workers),
            "--eval_num_workers",
            str(eval_num_workers),
        ]
        try:
            stream_process(command, project_root, run_dir / "launcher.log")
            result = validate_completed_run(run_dir, config, spec)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = repr(error)
            write_json(manifest_path, manifest)
            raise
        record.update(result)
        record["status"] = "completed"
        write_json(manifest_path, manifest)

    write_json(manifest_path, manifest)
    completed = sum(record.get("status") == "completed" for record in records)
    prepared = sum(record.get("status") in {"prepared", "dry_run"} for record in records)
    print(
        f"Finished request: total={len(records)}, completed={completed}, "
        f"not_started_or_dry_run={prepared}, output={output_root}"
    )


if __name__ == "__main__":
    main()
