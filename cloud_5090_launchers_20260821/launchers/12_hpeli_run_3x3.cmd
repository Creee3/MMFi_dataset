@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
set "OUT=%ROOT%\strict_offline_runs\hpeli_equal_samples_3x3_mpjpe_20260821_01"
if not "%~1"=="" set "OUT=%~1"
set "RESUME_FLAG="
if /I "%~2"=="resume" set "RESUME_FLAG=--resume"

pushd "%BASELINES%"
set "PYTHONPATH=%BASELINES%;%DEPS%"
"%PY%" -B -u cloud_rerun\run_3x3_equal_samples_three_models.py --project_root "%BASELINES%" --dataset_root "%DATA%" --output_root "%OUT%" --all_cells --models hpeli --seeds 0 1 2 --num_workers 8 --eval_num_workers 8 %RESUME_FLAG%
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

