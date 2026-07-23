"""Preflight diagnostics for S2P2 cloud training runs.

This script is intentionally read-only by default.  It checks the Python
runtime, user-site package leakage, CUDA/Torch, required project files, optional
result logs, and can run a tiny DataLoader smoke test in a subprocess that
matches the training environment.
"""

import argparse
import json
import os
import platform
import re
import site
import subprocess
import sys
from pathlib import Path


ERROR_PATTERNS = [
    (re.compile(r"Traceback \(most recent call last\):"), "python_exception"),
    (re.compile(r"DataLoader worker .* exited unexpectedly"), "dataloader_worker_exit"),
    (re.compile(r"Couldn't open shared file mapping"), "windows_shared_mapping"),
    (re.compile(r"returncode=3221225477"), "windows_access_violation"),
    (
        re.compile(r"TypeError: unsupported operand type\(s\) for -: 'int' and 'Untokenizer'"),
        "numpy_header_tokenize_failure",
    ),
    (
        re.compile(r"TypeError: '_EnumDict' object is not callable"),
        "windows_python_worker_spawn_failure",
    ),
    (
        re.compile(r"'slice' object is not subscriptable|bad argument to internal function"),
        "python_numpy_runtime_integrity_failure",
    ),
]


class Reporter:
    def __init__(self):
        self.rows = []

    def section(self, title):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def add(self, status, label, detail=""):
        self.rows.append({"status": status, "label": label, "detail": detail})
        suffix = f" | {detail}" if detail else ""
        print(f"[{status}] {label}{suffix}")

    def ok(self, label, detail=""):
        self.add("OK", label, detail)

    def warn(self, label, detail=""):
        self.add("WARN", label, detail)

    def fail(self, label, detail=""):
        self.add("FAIL", label, detail)

    def info(self, label, detail=""):
        self.add("INFO", label, detail)

    def summary(self):
        counts = {}
        for row in self.rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(json.dumps(counts, indent=2, ensure_ascii=False))
        if counts.get("FAIL"):
            print("Result: fix FAIL items before starting a long run.")
        elif counts.get("WARN"):
            print("Result: runnable, but review WARN items first.")
        else:
            print("Result: environment looks ready.")


def training_env(extra=None):
    env = os.environ.copy()
    env["MMFI_PROTOCOL"] = "protocol2"
    env["HELDOUT_SPLIT_ENABLED"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def run_subprocess(command, env=None, timeout=60):
    try:
        result = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return result.returncode, result.stdout
    except FileNotFoundError as exc:
        return 127, str(exc)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        return 124, output + f"\n[TIMEOUT after {timeout}s]"


def check_current_python(rep):
    rep.section("Current Python")
    rep.info("executable", sys.executable)
    rep.info("version", sys.version.replace("\n", " "))
    rep.info("prefix", sys.prefix)
    rep.info("cwd", os.getcwd())
    rep.info("platform", platform.platform())
    rep.info("CONDA_PREFIX", os.environ.get("CONDA_PREFIX", "-"))
    rep.info("PYTHONNOUSERSITE", os.environ.get("PYTHONNOUSERSITE", "-"))
    rep.info("site.ENABLE_USER_SITE", str(site.ENABLE_USER_SITE))

    user_site = site.getusersitepackages()
    rep.info("user site", user_site)
    leaked_paths = [
        path for path in sys.path
        if "appdata" in path.lower() and "roaming" in path.lower()
    ]
    if leaked_paths:
        rep.warn("Windows user-site path is visible in sys.path", "; ".join(leaked_paths))
    else:
        rep.ok("no obvious Windows user-site path in sys.path")

    if platform.system().lower() == "windows":
        exe = sys.executable.replace("\\", "/").lower()
        if "wimans2" not in exe:
            rep.warn("Python executable name does not look like WiMANS2", sys.executable)
        else:
            rep.ok("Python executable looks like WiMANS2")


def check_training_runtime(rep):
    rep.section("Training Runtime Probe")
    code = r"""
import importlib
import os
import site
import sys

print("executable=" + sys.executable)
print("prefix=" + sys.prefix)
print("PYTHONNOUSERSITE=" + os.environ.get("PYTHONNOUSERSITE", ""))
print("ENABLE_USER_SITE=" + str(site.ENABLE_USER_SITE))
for name in ["numpy", "torch", "optuna", "yaml", "cv2"]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        path = getattr(module, "__file__", "built-in")
        print(f"{name}={version} | {path}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR | {type(exc).__name__}: {exc}")

try:
    import torch
    print("torch.cuda.available=" + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.version=" + str(torch.version.cuda))
        print("torch.cuda.device_count=" + str(torch.cuda.device_count()))
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"cuda:{index}={props.name} | total_gb={total_gb:.2f}")
except Exception as exc:
    print("torch_cuda_probe=ERROR | " + repr(exc))
"""
    returncode, output = run_subprocess(
        [sys.executable, "-c", code],
        env=training_env(),
        timeout=90,
    )
    print(output.strip())
    if returncode != 0:
        rep.fail("training runtime probe failed", f"returncode={returncode}")
        return

    normalized = output.replace("\\", "/").lower()
    if "appdata/roaming/python" in normalized:
        rep.fail("training runtime still imports from AppData/Roaming", "user-site leakage")
    else:
        rep.ok("training runtime avoids AppData/Roaming user packages")
    if "numpy=import_error" in output.lower():
        rep.fail("NumPy import failed under training env")
    else:
        rep.ok("NumPy imports under training env")
    if "torch=import_error" in output.lower():
        rep.fail("Torch import failed under training env")
    elif "torch.cuda.available=True" in output:
        rep.ok("Torch sees CUDA")
    else:
        rep.warn("Torch does not see CUDA", "training will fall back to CPU")
    if "optuna=IMPORT_ERROR" in output:
        rep.fail("Optuna is missing", "required by run_s2p2_cmc_hparam_tuning.py")
    else:
        rep.ok("Optuna imports")


def check_nvidia_smi(rep):
    rep.section("NVIDIA-SMI")
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    returncode, output = run_subprocess(command, timeout=20)
    if returncode != 0:
        rep.warn("nvidia-smi GPU query failed", output.strip())
        return
    print(output.strip())
    rep.ok("nvidia-smi GPU query succeeded")

    app_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    app_returncode, app_output = run_subprocess(app_command, timeout=20)
    if app_returncode == 0 and app_output.strip():
        print("\nCompute processes:")
        print(app_output.strip())
        rep.info("GPU compute processes found")
    elif app_returncode == 0:
        rep.info("GPU compute processes", "none reported")
    else:
        rep.warn("nvidia-smi compute-process query failed", app_output.strip())


def check_paths(rep, args):
    rep.section("Project Paths")
    required_files = [
        "train_lupi_rgb_teacher_mpjpe.py",
        "run_s2p2_cmc_hparam_tuning.py",
        "inspect_s2p2_tuning_logs.py",
        "common.py",
    ]
    for file_name in required_files:
        path = Path(file_name)
        if path.is_file():
            rep.ok(f"found {file_name}", str(path.resolve()))
        else:
            rep.fail(f"missing {file_name}")

    dataset_root = Path(args.dataset_root)
    config_file = Path(args.config_file)
    if dataset_root.is_dir():
        rep.ok("dataset_root exists", str(dataset_root.resolve()))
    else:
        rep.fail("dataset_root missing", str(dataset_root))
    if config_file.is_file():
        rep.ok("config_file exists", str(config_file.resolve()))
    else:
        rep.fail("config_file missing", str(config_file))

    if args.teacher_ckpt:
        teacher = Path(args.teacher_ckpt)
        if teacher.is_file():
            size_mb = teacher.stat().st_size / (1024 ** 2)
            rep.ok("teacher checkpoint exists", f"{teacher} ({size_mb:.1f} MB)")
        else:
            rep.fail("teacher checkpoint missing", str(teacher))

    if args.results_root:
        root = Path(args.results_root)
        if root.exists():
            rep.info("results_root exists", str(root.resolve()))
        else:
            rep.info("results_root not created yet", str(root))


def classify_log_text(text):
    return [label for pattern, label in ERROR_PATTERNS if pattern.search(text)]


def read_text_if_exists(path):
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def scan_results_root(rep, results_root):
    if not results_root:
        return
    root = Path(results_root)
    rep.section("Results Root Scan")
    if not root.is_dir():
        rep.info("results root does not exist yet", str(root))
        return

    trial_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("trial_"))
    rep.info("trial directories", str(len(trial_dirs)))
    if not trial_dirs:
        return

    status_counts = {}
    error_counts = {}
    rows = []
    for trial_dir in trial_dirs:
        seed_dirs = sorted(path for path in trial_dir.iterdir() if path.is_dir() and path.name.startswith("seed_"))
        if not seed_dirs:
            rows.append((trial_dir.name, "-", "no_seed_dir", "-", "-"))
            status_counts["no_seed_dir"] = status_counts.get("no_seed_dir", 0) + 1
            continue
        for seed_dir in seed_dirs:
            complete_path = seed_dir / "train_complete.json"
            history_path = seed_dir / "history.json"
            best_path = seed_dir / "best.pth"
            log_text = read_text_if_exists(seed_dir / "log.txt")
            process_text = read_text_if_exists(seed_dir / "process_output.log")
            combined = log_text + "\n" + process_text
            errors = classify_log_text(combined)

            if complete_path.is_file() and history_path.is_file() and best_path.is_file():
                status = "complete"
            elif errors:
                status = "error"
            elif combined.strip():
                status = "interrupted"
            else:
                status = "no_log"

            last_epoch_match = re.findall(r"Epoch (\d+):", combined)
            last_epoch = last_epoch_match[-1] if last_epoch_match else "-"
            issue = ",".join(errors) if errors else "-"
            rows.append((trial_dir.name, seed_dir.name, status, last_epoch, issue))
            status_counts[status] = status_counts.get(status, 0) + 1
            for error in errors:
                error_counts[error] = error_counts.get(error, 0) + 1

    print("trial               seed      status       last_epoch  issue")
    print("-" * 78)
    for trial, seed, status, last_epoch, issue in rows[:40]:
        print(f"{trial[:19]:19} {seed:9} {status:12} {str(last_epoch):10} {issue}")
    if len(rows) > 40:
        print(f"... {len(rows) - 40} more seed rows omitted")

    rep.info("seed status counts", json.dumps(status_counts, ensure_ascii=False))
    if error_counts:
        rep.warn("log error patterns found", json.dumps(error_counts, ensure_ascii=False))
    else:
        rep.ok("no recognized error pattern in scanned logs")

    ranked_path = root / "summary_ranked.json"
    if ranked_path.is_file():
        try:
            ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
            if ranked:
                best = ranked[0]
                params = best.get("tuned_params", {})
                rep.info(
                    "current best summary",
                    (
                        f"trial={best.get('trial')} "
                        f"mpjpe={best.get('mpjpe_mean')} "
                        f"lr={params.get('lr')} bs={params.get('batch_size')}"
                    ),
                )
            else:
                rep.info("summary_ranked.json exists but has no complete trial")
        except (OSError, json.JSONDecodeError) as exc:
            rep.warn("could not read summary_ranked.json", repr(exc))


def shape_summary(value, depth=0):
    if depth > 3:
        return type(value).__name__
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): shape_summary(item, depth + 1) for key, item in list(value.items())[:8]}
    if isinstance(value, (list, tuple)):
        return [shape_summary(item, depth + 1) for item in list(value)[:8]]
    return type(value).__name__


def loader_smoke_worker(args):
    os.environ["MMFI_PROTOCOL"] = "protocol2"
    os.environ["HELDOUT_SPLIT_ENABLED"] = "0"

    from common import get_loaders, load_config

    cfg = load_config(args.config_file, "cross_subject_split")
    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.dataset_root,
        cfg,
        window=args.window,
        stride=args.stride,
        batch_size=args.smoke_batch_size,
        val_batch_size=args.smoke_batch_size,
        role="student",
        num_workers=args.smoke_num_workers,
        eval_num_workers=0,
        include_rgb_window=False,
    )
    print(f"protocol={cfg.get('protocol')}")
    print(f"heldout_split={cfg.get('heldout_split')}")
    print(f"train_windows={len(train_ds)}")
    print(f"val_windows={len(val_ds)}")
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print("train_batch=" + json.dumps(shape_summary(train_batch), ensure_ascii=False))
    print("val_batch=" + json.dumps(shape_summary(val_batch), ensure_ascii=False))


def run_loader_smoke(rep, args):
    rep.section("DataLoader Smoke Test")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        args.dataset_root,
        args.config_file,
        "--_loader_smoke_worker",
        "--window",
        str(args.window),
        "--stride",
        str(args.stride),
        "--smoke_batch_size",
        str(args.smoke_batch_size),
        "--smoke_num_workers",
        str(args.smoke_num_workers),
    ]
    returncode, output = run_subprocess(
        command,
        env=training_env(),
        timeout=args.smoke_timeout,
    )
    print(output.strip())
    if returncode == 0:
        rep.ok("DataLoader smoke test passed")
    else:
        rep.fail("DataLoader smoke test failed", f"returncode={returncode}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Diagnose the cloud environment before S2P2 CMC tuning."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--teacher_ckpt", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--loader_smoke", action="store_true")
    parser.add_argument("--smoke_batch_size", type=int, default=2)
    parser.add_argument("--smoke_num_workers", type=int, default=0)
    parser.add_argument("--smoke_timeout", type=int, default=300)
    parser.add_argument("--_loader_smoke_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main():
    args = build_parser().parse_args()
    if args._loader_smoke_worker:
        loader_smoke_worker(args)
        return

    rep = Reporter()
    check_current_python(rep)
    check_training_runtime(rep)
    check_nvidia_smi(rep)
    check_paths(rep, args)
    scan_results_root(rep, args.results_root)
    if args.loader_smoke:
        run_loader_smoke(rep, args)
    else:
        rep.section("DataLoader Smoke Test")
        rep.info("skipped", "add --loader_smoke to fetch one tiny train/val batch")
    rep.summary()


if __name__ == "__main__":
    main()
