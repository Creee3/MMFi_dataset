#!/usr/bin/env python3
"""Organize the local MMFi workspace like the 5090 project root.

The operation is non-destructive: entries are renamed on the same volume,
verified after each move, and recorded in a JSON manifest. Model navigation
uses relative symlinks so each source tree still has one physical copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "_root_archive_20260821"
MANIFEST = ARCHIVE / "local_archive_manifest.json"
IN_PROGRESS = ARCHIVE / "local_archive_manifest.in_progress.json"


NAVIGATION_MOVES = [
    ("cloud_5090_launchers_20260821/00_PROJECT_ENTRY_CN.md", "00_PROJECT_ENTRY_CN.md"),
    ("cloud_5090_launchers_20260821/launchers", "launchers"),
    (
        "cloud_5090_launchers_20260821/models_README_models_CN.md",
        "models/README_models_CN.md",
    ),
    (
        "cloud_5090_launchers_20260821/baseline_equal_samples_README_CN.md",
        "models/baseline_equal_samples/README_CN.md",
    ),
    (
        "cloud_5090_launchers_20260821/model_docs/dtpose_README_CN.md",
        "models/dtpose/README_CN.md",
    ),
    (
        "cloud_5090_launchers_20260821/model_docs/hpeli_README_CN.md",
        "models/hpeli/README_CN.md",
    ),
    (
        "cloud_5090_launchers_20260821/model_docs/metafi_README_CN.md",
        "models/metafi_pp/README_CN.md",
    ),
    (
        "cloud_5090_launchers_20260821/model_docs/privpose_README_CN.md",
        "models/privpose/README_CN.md",
    ),
    (
        "p2s2_per_joint_baseline_rerun_20260816/source",
        "models/dtpose/official_source",
    ),
]


ARCHIVE_MOVES = [
    (".cloud_5090_audit_20260821", "audit_records/cloud_5090_audit_20260821"),
    (
        ".staging_s2p2_equal_samples_mpjpe_code_20260820",
        "transfer_and_incomplete/staging_s2p2_equal_samples_mpjpe_code_20260820",
    ),
    ("__pycache__", "cache_and_metadata/__pycache__"),
    ("tmp", "cache_and_metadata/tmp"),
    ("PrivPose_本地归档_20260815", "previous_archives/PrivPose_本地归档_20260815"),
    ("_archived_legacy_20260619", "previous_archives/_archived_legacy_20260619"),
    ("_cloud_sync_archive_20260704", "previous_archives/_cloud_sync_archive_20260704"),
    ("backup_exports", "transfer_and_incomplete/backup_exports"),
    ("canonical_eval_s2p2_20260818", "evaluation_projects/canonical_eval_s2p2_20260818"),
    ("p2_pck10_50_archive_20260816", "evaluation_projects/p2_pck10_50_archive_20260816"),
    (
        "p2s2_per_joint_baseline_rerun_20260816",
        "evaluation_projects/p2s2_per_joint_baseline_rerun_20260816",
    ),
    (
        "s2p2_equal_samples_mpjpe_cloud_bundle_20260818",
        "old_code_bundles/s2p2_equal_samples_mpjpe_cloud_bundle_20260818",
    ),
    ("experiment_results", "historical_results/experiment_results"),
    ("pose日志【整合】", "historical_results/pose日志【整合】"),
    ("s2p2_results_logs", "historical_results/s2p2_results_logs"),
    ("strict_offline_runs_4090_20260812", "historical_results/strict_offline_runs_4090_20260812"),
    (
        "cloud_5090_launchers_20260821",
        "transfer_and_incomplete/cloud_5090_launchers_20260821_container",
    ),
    ("check_s2p2_data_integrity.py", "dataset_cleanup_history/root_tools/check_s2p2_data_integrity.py"),
    (
        "cleanup_unused_mmfi_modalities.py",
        "dataset_cleanup_history/root_tools/cleanup_unused_mmfi_modalities.py",
    ),
    ("diagnose_s2p2_cloud_env.bat", "maintenance_tools/diagnose_s2p2_cloud_env.bat"),
    ("diagnose_s2p2_cloud_env.py", "maintenance_tools/diagnose_s2p2_cloud_env.py"),
    ("inspect_s2p2_tuning_logs.py", "maintenance_tools/inspect_s2p2_tuning_logs.py"),
    (
        "rebuild_s2p2_optuna_bookkeeping.py",
        "maintenance_tools/rebuild_s2p2_optuna_bookkeeping.py",
    ),
    ("quick_check.sh", "maintenance_tools/quick_check.sh"),
    ("optuna_study.db", "cache_and_metadata/optuna_study.db"),
    ("evaluate_final_test.sh", "evaluation_tools/evaluate_final_test.sh"),
    (
        "evaluate_final_test_breakdown_mpjpe.py",
        "evaluation_tools/evaluate_final_test_breakdown_mpjpe.py",
    ),
    (
        "evaluate_final_test_checkpoints_mpjpe.py",
        "evaluation_tools/evaluate_final_test_checkpoints_mpjpe.py",
    ),
    ("evaluate_final_test_mpjpe.py", "evaluation_tools/evaluate_final_test_mpjpe.py"),
    ("evaluate_pck_10_50.py", "evaluation_tools/evaluate_pck_10_50.py"),
    ("evaluate_per_joint_mpjpe.py", "evaluation_tools/evaluate_per_joint_mpjpe.py"),
    ("summarize_baseline_p2_pck.py", "evaluation_tools/summarize_baseline_p2_pck.py"),
    ("run_s1_cmc_from_epoch1.py", "old_privpose_scripts/run_s1_cmc_from_epoch1.py"),
    ("run_s2p2_3x3_all_zero.py", "old_privpose_scripts/run_s2p2_3x3_all_zero.py"),
    (
        "run_s2p2_3x3_all_zero_cloud.bat",
        "old_privpose_scripts/run_s2p2_3x3_all_zero_cloud.bat",
    ),
    (
        "run_s2p2_csi_perturbation_ablation.py",
        "old_privpose_scripts/run_s2p2_csi_perturbation_ablation.py",
    ),
    (
        "run_s2p2_csi_perturbation_ablation_queued.bat",
        "old_privpose_scripts/run_s2p2_csi_perturbation_ablation_queued.bat",
    ),
    (
        "run_s2p2_mmfi_original_jobs.py",
        "old_privpose_scripts/run_s2p2_mmfi_original_jobs.py",
    ),
    ("run_s2p2_selected_jobs.py", "old_privpose_scripts/run_s2p2_selected_jobs.py"),
    (
        "run_s2p2_strict_sequence_jobs.py",
        "old_privpose_scripts/run_s2p2_strict_sequence_jobs.py",
    ),
    (
        "run_s2p2_strict_subject_jobs.py",
        "old_privpose_scripts/run_s2p2_strict_subject_jobs.py",
    ),
    ("run_s2p2_wifi_cmc_mix20.py", "old_privpose_scripts/run_s2p2_wifi_cmc_mix20.py"),
    ("run_t1_rgb_teacher_s2p2_cv4.py", "old_privpose_scripts/run_t1_rgb_teacher_s2p2_cv4.py"),
    ("train_baseline_mpjpe.py", "old_privpose_scripts/train_baseline_mpjpe.py"),
    ("run_formal_3x3.sh", "old_automation/run_formal_3x3.sh"),
    ("train_rgb_teacher.sh", "old_automation/train_rgb_teacher.sh"),
    (
        "watch_metafi_then_per_joint_eval_20260817.ps1",
        "old_automation/watch_metafi_then_per_joint_eval_20260817.ps1",
    ),
    ("5090云端代码位置与接手说明_中文.md", "old_docs/5090云端代码位置与接手说明_中文.md"),
    ("CODE_GUIDE.md", "old_docs/CODE_GUIDE.md"),
    ("PRIVPOSE_QUICKSTART.md", "old_docs/PRIVPOSE_QUICKSTART.md"),
    ("PrivPose代码使用说明_中文.md", "old_docs/PrivPose代码使用说明_中文.md"),
    ("S2P2消融实验说明.md", "old_docs/S2P2消融实验说明.md"),
    ("云端同步清单_S2P2.md", "old_docs/云端同步清单_S2P2.md"),
    ("云端四模型目录说明_中文.md", "old_docs/云端四模型目录说明_中文.md"),
    ("四模型统一评估维护入口_中文.md", "old_docs/四模型统一评估维护入口_中文.md"),
    ("对比模型调优与正式训练参数记录_中文.md", "old_docs/对比模型调优与正式训练参数记录_中文.md"),
    ("本地主代码目录说明.md", "old_docs/本地主代码目录说明.md"),
    ("P2四模型联合对比_中文.md", "historical_reports/P2四模型联合对比_中文.md"),
    (
        "baseline_p2_pck_10_50_summary_中文.md",
        "historical_reports/baseline_p2_pck_10_50_summary_中文.md",
    ),
    ("paper_revision_checklist.md", "historical_reports/paper_revision_checklist.md"),
    (
        "近三日工作总账与严谨性核验_20260817.md",
        "historical_reports/近三日工作总账与严谨性核验_20260817.md",
    ),
    (
        "baseline_p2_pck_10_50_summary.json",
        "historical_reports/baseline_p2_pck_10_50_summary.json",
    ),
    (
        "p2_s2_per_joint_privpose_summary.json",
        "historical_reports/p2_s2_per_joint_privpose_summary.json",
    ),
]


ZIP_MOVES = [
    "p2_pck10_50_archive_20260816.zip",
    "p2s2_eval_and_metafi_logging_patch_20260817.zip",
    "p2s2_per_joint_baseline_code_v2.zip",
    "p2s2_per_joint_baseline_rerun_20260816.zip",
    "s2p2_results_logs.zip",
]


SYMLINKS = [
    ("models/baseline_equal_samples/code", "../../s2p2_equal_samples_mpjpe_code_20260820"),
    ("models/metafi_pp/code", "../../metafi_s2p2_cloud_bundle_20260817"),
    ("models/dtpose/code", "official_source"),
    ("models/hpeli/code", "../dtpose/official_source"),
    (
        "models/dtpose/checkpoints",
        "../../rerun_archives/dtpose_762_20260816/s2p2_best_checkpoints",
    ),
    (
        "models/hpeli/checkpoints",
        "../../rerun_archives/hpeli_792_20260811/s2p2_best_checkpoints",
    ),
]


ACTIVE_REQUIRED = [
    "00_PROJECT_ENTRY_CN.md",
    "README.md",
    "common.py",
    "config.yaml",
    "run_s2p2_3x3_rank2.py",
    "run_s2p2_cmc_hparam_tuning.py",
    "run_s2p2_temporal_cmc_mixup_ablation.py",
    "train_lupi_rgb_teacher_mpjpe.py",
    "train_t1_rgb_teacher_mpjpe.py",
    "mmfi_lib",
    "launchers",
    "models",
    "data_base",
    "strict_offline_runs",
    "rerun_archives",
    "Zipfiles",
    "s2p2_equal_samples_mpjpe_code_20260820",
    "metafi_s2p2_cloud_bundle_20260817",
    "IEEE-Transactions-LaTeX2e-templates-and-instructions",
    "资料收集",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(path: Path) -> dict[str, object]:
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path), "file_count": 0, "bytes": 0}
    if path.is_file():
        return {
            "kind": "file",
            "file_count": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    if not path.is_dir():
        raise RuntimeError(f"Unsupported path type: {path}")

    file_count = 0
    total_bytes = 0
    for current, dirnames, filenames in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in dirnames:
            candidate = current_path / name
            if candidate.is_symlink():
                file_count += 1
                total_bytes += candidate.lstat().st_size
        for name in filenames:
            candidate = current_path / name
            file_count += 1
            total_bytes += candidate.lstat().st_size
    return {"kind": "directory", "file_count": file_count, "bytes": total_bytes}


manifest: dict[str, object] = {
    "schema_version": 1,
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "root": str(ROOT),
    "policy": [
        "No files are deleted.",
        "Moves are same-volume renames and are verified after completion.",
        "Model navigation uses relative symlinks to avoid duplicate source trees.",
        "The local paper and reference workspaces retain their original paths.",
    ],
    "moves": [],
    "symlinks": [],
}


def flush(path: Path = IN_PROGRESS) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def move(source_rel: str, destination_rel: str, *, required: bool = False) -> None:
    source = ROOT / source_rel
    destination = ROOT / destination_rel
    if not source.exists() and not source.is_symlink():
        if destination.exists() or destination.is_symlink():
            manifest["moves"].append(
                {"source": source_rel, "destination": destination_rel, "status": "already_moved"}
            )
            flush()
            return
        if required:
            raise FileNotFoundError(f"Required source is missing: {source}")
        manifest["moves"].append(
            {"source": source_rel, "destination": destination_rel, "status": "source_absent"}
        )
        flush()
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite destination: {destination}")

    before = measure(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    after = measure(destination)
    for key in ("kind", "file_count", "bytes", "sha256"):
        if key in before and before.get(key) != after.get(key):
            raise RuntimeError(f"Verification failed for {source_rel}: {key}")
    manifest["moves"].append(
        {
            "source": source_rel,
            "destination": destination_rel,
            "status": "moved_verified",
            **before,
        }
    )
    flush()


def create_symlink(link_rel: str, target: str) -> None:
    link = ROOT / link_rel
    expected = (link.parent / target).resolve()
    if not expected.exists():
        raise FileNotFoundError(f"Symlink target does not exist: {expected}")
    if link.is_symlink():
        if os.readlink(link) != target:
            raise RuntimeError(f"Existing symlink has another target: {link}")
        status = "already_present"
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link}")
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        status = "created"
    if link.resolve() != expected:
        raise RuntimeError(f"Symlink verification failed: {link}")
    manifest["symlinks"].append(
        {"path": link_rel, "target": target, "resolved": str(expected), "status": status}
    )
    flush()


def preflight() -> None:
    problems: list[str] = []
    planned_moves = [
        *((source, destination, True) for source, destination in NAVIGATION_MOVES),
        *(
            (source, f"_root_archive_20260821/{destination}", False)
            for source, destination in ARCHIVE_MOVES
        ),
        *((source, f"Zipfiles/{source}", False) for source in ZIP_MOVES),
    ]
    for source_rel, destination_rel, required in planned_moves:
        source = ROOT / source_rel
        destination = ROOT / destination_rel
        source_exists = source.exists() or source.is_symlink()
        destination_exists = destination.exists() or destination.is_symlink()
        if source_exists and destination_exists:
            problems.append(f"Both source and destination exist: {source_rel} -> {destination_rel}")
        elif required and not source_exists and not destination_exists:
            problems.append(f"Required source is missing: {source_rel}")

    for link_rel, target in SYMLINKS:
        link = ROOT / link_rel
        if link.exists() and not link.is_symlink():
            problems.append(f"Symlink path is occupied: {link_rel}")
        elif link.is_symlink() and os.readlink(link) != target:
            problems.append(f"Symlink has another target: {link_rel}")

    if problems:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(problems))


def main() -> None:
    if MANIFEST.exists() and "--check" not in sys.argv:
        raise RuntimeError(
            f"Organization has already completed; inspect the existing manifest: {MANIFEST}"
        )
    preflight()
    if "--check" in sys.argv:
        print(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "planned_moves": len(NAVIGATION_MOVES) + len(ARCHIVE_MOVES) + len(ZIP_MOVES),
                    "planned_symlinks": len(SYMLINKS),
                }
            )
        )
        return

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    flush()

    for source, destination in NAVIGATION_MOVES:
        move(source, destination, required=True)
    for source, destination in ARCHIVE_MOVES:
        move(source, f"_root_archive_20260821/{destination}")
    for source in ZIP_MOVES:
        move(source, f"Zipfiles/{source}")
    for link, target in SYMLINKS:
        create_symlink(link, target)

    missing = [item for item in ACTIVE_REQUIRED if not (ROOT / item).exists()]
    if missing:
        raise RuntimeError(f"Active paths missing after organization: {missing}")
    manifest["active_required_verified"] = ACTIVE_REQUIRED
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to replace completed manifest: {MANIFEST}")
    flush(IN_PROGRESS)
    IN_PROGRESS.rename(MANIFEST)
    print(json.dumps({"status": "ok", "moves": len(manifest["moves"]), "symlinks": len(SYMLINKS)}))


if __name__ == "__main__":
    main()
