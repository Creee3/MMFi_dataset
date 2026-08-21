@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
set "OUT=%ROOT%\strict_offline_runs\S2P2_temporal_cmc_mixup_ablation"
if not "%~1"=="" set "OUT=%~1"

pushd "%PRIVPOSE%"
set "PYTHONPATH=%DEPS%"
set "PYTHONNOUSERSITE=1"
"%PY%" -B -u run_s2p2_temporal_cmc_mixup_ablation.py "%DATA%" "%ROOT%\config.yaml" --formal_plan "%ROOT%\strict_offline_runs\S2P2_3x3_rank2\experiment_plan.json" --teacher_selection "%ROOT%\strict_offline_runs\S2P2_3x3_rank2\protocol2\cross_subject_split\selected_teacher.json" --full_result "%ROOT%\strict_offline_runs\S2P2_3x3_rank2\protocol2\cross_subject_split\student_result.json" --results_root "%OUT%" --num_workers 8 --eval_num_workers 0 --worker_fallbacks 4 2 1 0 --min_free_gpu_mib 0
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

