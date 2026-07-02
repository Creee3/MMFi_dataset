import os


def main():
    root = os.path.abspath(os.getcwd())
    common_path = os.path.abspath("common.py")
    baseline_path = os.path.abspath("train_baseline_mpjpe.py")
    lupi_path = os.path.abspath("train_lupi_rgb_teacher_mpjpe.py")
    grid_path = os.path.abspath("run_strict_mpjpe_grid_search.py")
    cmc_auto_path = os.path.abspath("run_cmc_gh_tail_auto.py")
    student_val_auto_path = os.path.abspath("run_cmc_student_val_auto.py")
    csi_check_path = os.path.abspath("check_wifi_csi_frames.py")
    mmfi_path = os.path.abspath("mmfi_lib/mmfi.py")
    config_path = os.path.abspath("config.yaml")

    with open(common_path, "r", encoding="utf-8") as f:
        common_text = f.read()
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_text = f.read()
    with open(lupi_path, "r", encoding="utf-8") as f:
        lupi_text = f.read()
    with open(grid_path, "r", encoding="utf-8") as f:
        grid_text = f.read()
    with open(cmc_auto_path, "r", encoding="utf-8") as f:
        cmc_auto_text = f.read()
    with open(student_val_auto_path, "r", encoding="utf-8") as f:
        student_val_auto_text = f.read()
    with open(csi_check_path, "r", encoding="utf-8") as f:
        csi_check_text = f.read()
    with open(mmfi_path, "r", encoding="utf-8") as f:
        mmfi_text = f.read()
    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

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
        "lupi_path": lupi_path,
        "lupi_forward_alias": "def forward(self, wifi_window)" in common_text,
        "lupi_tail_soup": "build_tail_soup" in lupi_text and "--tail_soup" in lupi_text,
        "lupi_tail_ckpts": "maybe_save_tail_checkpoint" in lupi_text and "--tail_ckpts" in lupi_text,
        "lupi_last_ckpt": "last.pth" in lupi_text,
        "lupi_parser_lambda_bone": 'parser.add_argument("--lambda_bone"' in lupi_text,
        "lupi_parser_tail": 'parser.add_argument("--tail_ckpts"' in lupi_text and 'parser.add_argument("--tail_soup"' in lupi_text,
        "lupi_parser_skip_final": 'parser.add_argument("--skip_final_test"' in lupi_text,
        "grid_path": grid_path,
        "grid_short_run_dirs": "short_run_dirs" in grid_text and "short_job_id" in grid_text,
        "grid_eval_seed_none": "run_command(cmd, seed=None)" in grid_text,
        "grid_topk_passthrough": "--topk_ckpts" in grid_text and "--topk_soup" in grid_text,
        "grid_tail_passthrough": "--tail_ckpts" in grid_text and "--tail_soup" in grid_text,
        "grid_cmc_tail_passthrough": "make_cmc_jobs" in grid_text and "args.tail_ckpts" in grid_text,
        "cmc_auto_path": cmc_auto_path,
        "cmc_auto_mode": '"cmc"' in cmc_auto_text and '"--mode"' in cmc_auto_text,
        "cmc_auto_model_type": '"lupi"' in cmc_auto_text and '"--model_type"' in cmc_auto_text,
        "student_val_auto_path": student_val_auto_path,
        "student_val_auto_split": "teacher_student_split" in student_val_auto_text,
        "student_val_auto_cmc_immediate": "default=1" in student_val_auto_text and "default=0" in student_val_auto_text,
        "csi_check_path": csi_check_path,
        "csi_check_has_validation": "CSIamp" in csi_check_text and "expected third dim >= 10" in csi_check_text,
        "read_frame_reports_bad_csi": "expected numpy.ndarray" in mmfi_text,
        "config_teacher_student_split": "teacher_student_split:" in config_text,
        "grid_split_arg": "args.split" in grid_text and "teacher_student_split" in grid_text,
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
        checks["lupi_forward_alias"],
        checks["lupi_tail_soup"],
        checks["lupi_tail_ckpts"],
        checks["lupi_last_ckpt"],
        checks["lupi_parser_lambda_bone"],
        checks["lupi_parser_tail"],
        checks["lupi_parser_skip_final"],
        checks["grid_short_run_dirs"],
        checks["grid_eval_seed_none"],
        checks["grid_topk_passthrough"],
        checks["grid_tail_passthrough"],
        checks["grid_cmc_tail_passthrough"],
        checks["cmc_auto_mode"],
        checks["cmc_auto_model_type"],
        checks["student_val_auto_split"],
        checks["student_val_auto_cmc_immediate"],
        checks["csi_check_has_validation"],
        checks["read_frame_reports_bad_csi"],
        checks["config_teacher_student_split"],
        checks["grid_split_arg"],
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
