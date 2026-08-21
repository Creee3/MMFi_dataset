@echo off
chcp 65001 >nul

set "ROOT=D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset"
set "PY=C:\Users\CHEN\.conda\envs\bit\python.exe"
set "DATA=%ROOT%\data_base"
set "PRIVPOSE=%ROOT%"
set "BASELINES=%ROOT%\s2p2_equal_samples_mpjpe_code_20260820"
set "DEPS=%ROOT%\_python_deps"
set "DTPRETRAIN_DEFAULT=%ROOT%\strict_offline_runs\dtpose_equal_samples_pretrain_20260821_01"

if not exist "%PY%" (
  echo ERROR: Python not found: %PY%
  exit /b 10
)
if not exist "%DATA%" (
  echo ERROR: Dataset not found: %DATA%
  exit /b 11
)
if not exist "%PRIVPOSE%\run_s2p2_3x3_rank2.py" (
  echo ERROR: PrivPose entry missing
  exit /b 12
)
if not exist "%BASELINES%\cloud_rerun\run_3x3_equal_samples_three_models.py" (
  echo ERROR: Baseline entry missing
  exit /b 13
)
exit /b 0
