@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
set "OUT=%DTPRETRAIN_DEFAULT%"
if not "%~1"=="" set "OUT=%~1"
set "RESUME_FLAG="
if /I "%~2"=="resume" set "RESUME_FLAG=--resume"

pushd "%BASELINES%"
set "PYTHONPATH=%BASELINES%;%DEPS%"
"%PY%" -B -u cloud_rerun\run_dtpose_equal_samples_pretrain.py --project_root "%BASELINES%" --dataset_root "%DATA%" --output_root "%OUT%" --all_cells --seed 42 --num_workers 8 %RESUME_FLAG%
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

