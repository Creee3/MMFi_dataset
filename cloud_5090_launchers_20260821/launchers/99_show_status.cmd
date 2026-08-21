@echo off
setlocal
call "%~dp0_env.cmd"
if errorlevel 1 exit /b 1
"%PY%" -B "%~dp0status_four_models.py" --root "%ROOT%"
exit /b %ERRORLEVEL%

