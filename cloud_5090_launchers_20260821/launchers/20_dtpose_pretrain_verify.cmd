@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
pushd "%BASELINES%"
set "PYTHONPATH=%BASELINES%;%DEPS%"
"%PY%" -B -u cloud_rerun\run_dtpose_equal_samples_pretrain.py --project_root "%BASELINES%" --dataset_root "%DATA%" --all_cells --seed 42 --num_workers 8 --skip_file_preflight --verify_only
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

