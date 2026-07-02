import os


def main():
    root = os.path.abspath(os.getcwd())
    common_path = os.path.abspath("common.py")
    baseline_path = os.path.abspath("train_baseline_mpjpe.py")
    grid_path = os.path.abspath("run_strict_mpjpe_grid_search.py")

    with open(common_path, "r", encoding="utf-8") as f:
        common_text = f.read()
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_text = f.read()
    with open(grid_path, "r", encoding="utf-8") as f:
        grid_text = f.read()

    common_lines = common_text.count("\n") + 1
    checks = {
        "cwd": root,
        "common_path": common_path,
        "common_lines": common_lines,
        "save_json_mkdir": "def save_json" in common_text and "os.makedirs(parent, exist_ok=True)" in common_text,
        "logger_mkdir": "class Logger" in common_text and "os.path.dirname(log_path)" in common_text,
        "has_graph_pose_head": "class GraphPoseHead" in common_text,
        "has_bone_length_loss": "def bone_length_loss" in common_text,
        "has_pose_head_arg": "--pose_head_type" in common_text,
        "baseline_path": baseline_path,
        "baseline_topk_soup": "def build_topk_soup" in baseline_text and "--topk_soup" in baseline_text,
        "baseline_topk_ckpts": "def maybe_save_topk_checkpoint" in baseline_text and "--topk_ckpts" in baseline_text,
        "baseline_tail_soup": "def build_tail_soup" in baseline_text and "--tail_soup" in baseline_text,
        "baseline_tail_ckpts": "def maybe_save_tail_checkpoint" in baseline_text and "--tail_ckpts" in baseline_text,
        "grid_path": grid_path,
        "grid_short_run_dirs": "short_run_dirs" in grid_text and "short_job_id" in grid_text,
        "grid_eval_seed_none": "run_command(cmd, seed=None)" in grid_text,
        "grid_topk_passthrough": "--topk_ckpts" in grid_text and "--topk_soup" in grid_text,
        "grid_tail_passthrough": "--tail_ckpts" in grid_text and "--tail_soup" in grid_text,
    }

    print("Project integrity check")
    print("=" * 60)
    ok = True
    for key, value in checks.items():
        print(f"{key}: {value}")
    required = [
        checks["save_json_mkdir"],
        checks["logger_mkdir"],
        checks["has_graph_pose_head"],
        checks["has_bone_length_loss"],
        checks["has_pose_head_arg"],
        checks["baseline_topk_soup"],
        checks["baseline_topk_ckpts"],
        checks["baseline_tail_soup"],
        checks["baseline_tail_ckpts"],
        checks["grid_short_run_dirs"],
        checks["grid_eval_seed_none"],
        checks["grid_topk_passthrough"],
        checks["grid_tail_passthrough"],
    ]
    ok = all(required)
    print("=" * 60)
    if ok:
        print("OK: codebase has the required stability fixes.")
        return 0
    print("FAILED: this folder is missing recent fixes. Do not launch training from it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
