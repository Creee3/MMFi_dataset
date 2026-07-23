@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=C:\ProgramData\Anaconda3\envs\WiMANS2\python.exe"
set "TEACHER_CKPT=strict_offline_runs\T1_rgb_teacher_s2p2_cv4\fold04\best.pth"
set "RESULTS_ROOT=strict_offline_runs\S2P2_cmc_lr_bs_tuning_fresh30_cmcE4"

"%PYTHON_EXE%" diagnose_s2p2_cloud_env.py data_base config.yaml ^
  --teacher_ckpt "%TEACHER_CKPT%" ^
  --results_root "%RESULTS_ROOT%" ^
  --window 32 ^
  --stride 2 ^
  --loader_smoke ^
  --smoke_batch_size 2 ^
  --smoke_num_workers 0 ^
  --smoke_timeout 600

endlocal
