@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
set "OUT=%ROOT%\strict_offline_runs\_launcher_checks\privpose_3x3_dry_run"
if not "%~1"=="" set "OUT=%~1"

pushd "%PRIVPOSE%"
set "PYTHONPATH=%DEPS%"
set "PYTHONNOUSERSITE=1"
"%PY%" -B -u run_s2p2_3x3_rank2.py "%DATA%" "%ROOT%\config.yaml" --selected_file "%ROOT%\strict_offline_runs\S2P2_cmc_lr_bs_tuning_w84200_resume_clean2\selected_for_3x3.json" --results_root "%OUT%" --num_workers 8 --eval_num_workers 0 --worker_fallbacks 4 2 1 0 --min_free_gpu_mib 0 --dry_run
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

