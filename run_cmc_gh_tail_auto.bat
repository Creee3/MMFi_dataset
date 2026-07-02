@echo off
setlocal

set PYTHONIOENCODING=utf-8

if "%1"=="dry" (
  python run_cmc_gh_tail_auto.py --dry_run
) else (
  python run_cmc_gh_tail_auto.py %*
)

endlocal
