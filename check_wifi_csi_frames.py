import argparse
import os

import numpy as np
import scipy.io as scio

from common import load_config
from mmfi_lib.mmfi import (
    make_dataset,
    make_four_way_dataset,
    make_teacher_student_dataset,
)


def iter_base_datasets(dataset_root, cfg):
    split = cfg.get("split_to_use")
    if split == "teacher_student_split":
        train_ds, teacher_val_ds, student_val_ds = make_teacher_student_dataset(dataset_root, cfg)
        yield "train", train_ds
        yield "teacher_val", teacher_val_ds
        yield "student_val", student_val_ds
    elif split == "four_way_split":
        train_ds, teacher_val_ds, student_val_ds, test_ds = make_four_way_dataset(dataset_root, cfg)
        yield "train", train_ds
        yield "teacher_val", teacher_val_ds
        yield "student_val", student_val_ds
        yield "test", test_ds
    else:
        train_ds, val_ds = make_dataset(dataset_root, cfg)
        yield "train", train_ds
        yield "val", val_ds


def check_csi_file(path):
    try:
        mat = scio.loadmat(path)
        if "CSIamp" not in mat:
            return "missing CSIamp"
        data = mat["CSIamp"]
        if not isinstance(data, np.ndarray):
            return f"CSIamp is {type(data).__name__}, not ndarray"
        if data.ndim != 3:
            return f"CSIamp ndim={data.ndim}, expected 3"
        if data.shape[2] < 10:
            return f"CSIamp shape={data.shape}, expected third dim >= 10"
        if not np.issubdtype(data.dtype, np.number):
            return f"CSIamp dtype={data.dtype}, expected numeric"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Check MMFi WiFi CSI .mat frames.")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--split", default="teacher_student_split")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after checking this many unique files; 0 checks all")
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config_file, args.split)
    checked = set()
    total = 0
    errors = []

    for ds_name, ds in iter_base_datasets(args.dataset_root, cfg):
        print(f"Scanning {ds_name}: {len(ds.data_list)} frames")
        for item in ds.data_list:
            path = item.get("wifi-csi_path")
            if not path or path in checked:
                continue
            checked.add(path)
            total += 1
            if total % 5000 == 0:
                print(f"  checked {total} files...")
            if not os.path.isfile(path):
                err = "missing file"
            else:
                err = check_csi_file(path)
            if err:
                errors.append((ds_name, path, err))
                print(f"BAD [{ds_name}] {path}: {err}")
                if len(errors) >= args.max_errors:
                    print(f"Stopped after {len(errors)} errors.")
                    return 2
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break

    print("=" * 60)
    print(f"Checked files: {total}")
    print(f"Errors: {len(errors)}")
    if errors:
        return 2
    print("OK: all checked WiFi CSI frames look valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
