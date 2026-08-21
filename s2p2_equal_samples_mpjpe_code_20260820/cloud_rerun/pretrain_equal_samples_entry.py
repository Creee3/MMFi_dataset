#!/usr/bin/env python3
"""Run the archived DT-Pose pretrainer with an explicit DataLoader worker count.

The historical ``pretrain.py`` is kept verbatim.  This adapter only patches
the bundle's existing ``make_dataloader`` default before executing it, so the
pretraining loss and model code remain unchanged.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import feeder.mmfi as mmfi

    original_make_dataloader = mmfi.make_dataloader

    def make_dataloader_with_workers(*positional, **keyword):
        keyword["num_workers"] = args.num_workers
        return original_make_dataloader(*positional, **keyword)

    mmfi.make_dataloader = make_dataloader_with_workers
    pretrain_script = project_root / "pretrain.py"
    if not pretrain_script.is_file():
        raise FileNotFoundError(pretrain_script)

    sys.argv = [
        str(pretrain_script),
        "--config_file",
        str(Path(args.config_file).resolve()),
    ]
    runpy.run_path(str(pretrain_script), run_name="__main__")


if __name__ == "__main__":
    main()
