@echo off
rem Run make_dataset.py and start it again by itself if it crashes.
rem
rem   scripts\run_dataset.bat                      the full set: IDs 0-1999, 2 shapes at once
rem   scripts\run_dataset.bat --ids 0-19           any make_dataset.py options instead
rem
rem Double-click it, run it from any window, or put a shortcut to it in the Startup folder
rem so the run continues by itself after a restart (docs/run-dataset.md). It finds the repo
rem from its own location and activates rift-teacher, so no Miniforge Prompt is needed.
rem
rem make_dataset.py's exit code decides what happens next:
rem   0   every shape is done            -> end
rem   10  stopped on request (STOP file, Ctrl+C) -> end
rem   11  five shapes failed in a row    -> end (fix the setup first, then start again)
rem   any other code (a crash)           -> wait one minute, start again (at most 20 times)
rem Finished shapes are always skipped, so a restart only redoes the shapes that were running.

setlocal
cd /d "%~dp0.."

rem Optional: set a backup folder here, for example: set "BACKUP=G:\My Drive\RIFT\backup_v2"
set "BACKUP="

if "%CONDA_ROOT%"=="" if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\miniforge3"
if "%CONDA_ROOT%"=="" if exist "C:\dev\miniforge3\Scripts\activate.bat" set "CONDA_ROOT=C:\dev\miniforge3"
if "%CONDA_ROOT%"=="" (
    echo Miniforge not found. Set CONDA_ROOT to your Miniforge folder, for example:
    echo     set CONDA_ROOT=C:\Users\you\miniforge3
    pause
    exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat" rift-teacher
if errorlevel 1 (
    echo Could not activate the rift-teacher environment.
    pause
    exit /b 1
)

rem Print in UTF-8, so the progress marks never crash Python when the output is not a
rem normal window (redirected to a file, or an older console).
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--ids 0-1999 --workers 2"
if not "%BACKUP%"=="" set ARGS=%ARGS% --backup-dir "%BACKUP%"

set /a TRIES=0
:again
set /a TRIES+=1
echo.
echo ==== run_dataset.bat: start %TRIES% at %date% %time%: make_dataset.py %ARGS%
python scripts\make_dataset.py %ARGS%
set CODE=%errorlevel%
if "%CODE%"=="0" (
    echo ==== Finished: every shape asked for is made, or failed for good: see the summary above.
    goto end
)
if "%CODE%"=="10" (
    echo ==== Stopped on request. Run this file again to continue where it stopped.
    goto end
)
if "%CODE%"=="11" (
    echo ==== Five shapes failed in a row. Read the errors above, fix the setup, then run this file again.
    goto end
)
if %TRIES% GEQ 20 (
    echo ==== Crashed %TRIES% times; not starting again. Read the errors above.
    goto end
)
echo ==== make_dataset.py ended with code %CODE%. Starting again in 60 s (close this window to cancel).
timeout /t 60 /nobreak >nul
goto again

:end
endlocal
pause
