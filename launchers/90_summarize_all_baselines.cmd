@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
set "RESULTS=%ROOT%\strict_offline_runs\three_baselines_equal_samples_3x3_mpjpe_20260821_01"
if not "%~1"=="" set "RESULTS=%~1"
set "PARTIAL_FLAG="
if /I "%~2"=="partial" set "PARTIAL_FLAG=--allow_partial"

pushd "%BASELINES%"
set "PYTHONPATH=%BASELINES%;%DEPS%"
"%PY%" -B -u cloud_rerun\summarize_3x3_training_runs.py --results_root "%RESULTS%" %PARTIAL_FLAG%
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

