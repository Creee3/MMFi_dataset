@echo off
setlocal

set PYTHONIOENCODING=utf-8
set TRAIN_SEED=0
set DRY_RUN=

if "%1"=="dry" set DRY_RUN=--dry_run

python check_project_integrity.py
if errorlevel 1 (
  echo Project integrity check failed. Stop.
  exit /b 1
)

python run_strict_mpjpe_grid_search.py data_base config.yaml ^
  --mode baseline ^
  --results_root experiment_results/gh_focus ^
  --seeds 0 ^
  --lrs 7e-5 ^
  --mixup_alphas 0.16 0.18 ^
  --dropouts 0.15 ^
  --transformer_layers 2 ^
  --dim_feedforwards 1024 ^
  --pose_head_hiddens 512 ^
  --batch_sizes 64 ^
  --aug_noises 0.02 ^
  --aug_freq_masks 0.05 ^
  --aug_time_masks 0.05 ^
  --lambda_rel_poses 0.0 ^
  --lambda_roots 0.0 ^
  --lambda_bones 0.0025 0.005 ^
  --pose_head_types graph_root ^
  --subject_robust_weights 0.0 ^
  --swa_start 0 ^
  --num_workers 0 ^
  %DRY_RUN%

endlocal
