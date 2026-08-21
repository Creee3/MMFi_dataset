@echo off
setlocal

cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHON_EXE=D:\Anaconda_envs\envs\WiMANS2\python.exe"
set "MAIN_PID=%~1"
set "RESULT_DIR=strict_offline_runs\S2P2_csi_perturbation_ablation_s1p1_seed0"

if "%MAIN_PID%"=="" set "MAIN_PID=32692"
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

> "%RESULT_DIR%\queue_status.log" echo [%date% %time%] Waiting for main PID %MAIN_PID%.

:wait_main
tasklist /FI "PID eq %MAIN_PID%" /NH | findstr /C:"%MAIN_PID%" >nul
if errorlevel 1 goto run_ablation
timeout /t 60 /nobreak >nul
goto wait_main

:run_ablation
>> "%RESULT_DIR%\queue_status.log" echo [%date% %time%] Main PID exited. Starting ablation.
"%PYTHON_EXE%" -s run_s2p2_csi_perturbation_ablation.py data_base config.yaml ^
  --num_workers 1 ^
  --attempt_no_output_timeout 600 ^
  --progress_log_every 100 ^
  --device cuda ^
  >> "%RESULT_DIR%\queued_process_output.log" ^
  2>> "%RESULT_DIR%\queued_process_error.log"

set "EXIT_CODE=%ERRORLEVEL%"
>> "%RESULT_DIR%\queue_status.log" echo [%date% %time%] Ablation exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
