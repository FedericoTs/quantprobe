@echo off
REM Unattended SERIAL re-run of the two measurements contaminated on 2026-07-31.
REM Launch from a NORMAL terminal with no coding agent running - the CPU gate cannot
REM open while an agent is making tool calls (measured: 29.9%% mean, ceiling is 20%%).
REM
REM   weights\RUN_UNATTENDED.bat                  both phases (ladder, then disk-tier)
REM   weights\RUN_UNATTENDED.bat --only=ladder    just the ladder
REM   weights\RUN_UNATTENDED.bat --only=disktier  just the disk-tier row
REM
REM Expect 2-3 hours. It waits up to ~60 min per phase for the box to go quiet, then
REM SKIPS that phase rather than forcing a number through a gate that never opened.

cd /d "%~dp0\.."
echo.
echo   Starting unattended serial run. Do not use this PC until it finishes.
echo   Progress: weights\data\unattended_*_run.log
echo.
python weights\unattended_serial.py %*
echo.
echo   Done. Exit code %ERRORLEVEL%.
echo   Result: weights\data\unattended_*_RESULT.json
echo.
pause
