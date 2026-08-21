@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
set "PRETRAIN=%DTPRETRAIN_DEFAULT%"
if not "%~1"=="" set "PRETRAIN=%~1"

pushd "%BASELINES%"
set "PYTHONPATH=%BASELINES%;%DEPS%"
"%PY%" -B -u cloud_rerun\run_3x3_equal_samples_three_models.py --project_root "%BASELINES%" --dataset_root "%DATA%" --dtpose_pretrain_root "%PRETRAIN%" --all_cells --models dtpose --seeds 0 1 2 --num_workers 8 --eval_num_workers 8 --skip_file_preflight --verify_only
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

