#!/usr/bin/env python3
"""Show a concise status summary without importing model code."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def marker(path: Path) -> str:
    return "complete" if path.is_file() else "not complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    results = root / "strict_offline_runs"

    print("=== PrivPose ===")
    formal = results / "S2P2_3x3_rank2"
    cell_results = list(formal.glob("protocol*/**/student_result.json"))
    print(f"formal 3x3: {marker(formal / 'run_complete.json')}; cells={len(cell_results)}/9")
    ablation = results / "S2P2_temporal_cmc_mixup_ablation"
    print(f"S2P2 ablation: {marker(ablation / 'ablation_result.json')}")

    print("\n=== Baseline manifests ===")
    manifests = sorted(results.glob("*/manifest.json"))
    if not manifests:
        print("no direct-child manifest found")
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"{manifest_path.parent.name}: unreadable ({error})")
            continue
        runs = manifest.get("runs", [])
        counts = Counter(record.get("status", "unknown") for record in runs)
        models = ",".join(manifest.get("models", [])) or manifest.get("model_name", "pretrain")
        print(
            f"{manifest_path.parent.name}: models={models}; "
            f"runs={len(runs)}; status={dict(sorted(counts.items()))}"
        )


if __name__ == "__main__":
    main()

