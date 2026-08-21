@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1

echo [1/4] Python and CUDA
"%PY%" -B -c "import torch, scipy, yaml, numpy, sklearn, pandas; print('python/dep OK'); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if errorlevel 1 exit /b 20

echo [2/4] PrivPose entry and required files
pushd "%PRIVPOSE%"
set "PYTHONPATH=%DEPS%"
set "PYTHONNOUSERSITE=1"
"%PY%" -B run_s2p2_3x3_rank2.py --help >nul
if errorlevel 1 (popd & exit /b 21)
"%PY%" -B -c "from pathlib import Path; r=Path('.'); fs=['common.py','config.yaml','run_s2p2_cmc_hparam_tuning.py','train_t1_rgb_teacher_mpjpe.py','train_lupi_rgb_teacher_mpjpe.py','mmfi_lib/mmfi.py','mmfi_lib/evaluate.py','strict_offline_runs/S2P2_cmc_lr_bs_tuning_w84200_resume_clean2/selected_for_3x3.json']; missing=[p for p in fs if not (r/p).is_file()]; print('PrivPose required files OK' if not missing else 'Missing: '+str(missing)); raise SystemExit(bool(missing))"
if errorlevel 1 (popd & exit /b 22)
popd

echo [3/4] MetaFi++ and HPE-Li source/sample verification
pushd "%BASELINES%"
set "PYTHONPATH=%BASELINES%;%DEPS%"
"%PY%" -B -u cloud_rerun\run_3x3_equal_samples_three_models.py --project_root "%BASELINES%" --dataset_root "%DATA%" --all_cells --models metafi hpeli --seeds 0 1 2 --num_workers 8 --eval_num_workers 8 --skip_file_preflight --verify_only
if errorlevel 1 (popd & exit /b 23)

echo [4/4] DT-Pose equal-sample pretraining verification
"%PY%" -B -u cloud_rerun\run_dtpose_equal_samples_pretrain.py --project_root "%BASELINES%" --dataset_root "%DATA%" --all_cells --seed 42 --num_workers 8 --skip_file_preflight --verify_only
if errorlevel 1 (popd & exit /b 24)
popd

echo ALL VERIFICATIONS PASSED. No training was started.
exit /b 0

