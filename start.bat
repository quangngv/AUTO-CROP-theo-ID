@echo off
chcp 65001 >nul

REM Tim pythonw.exe (khong mo terminal)
where pythonw >nul 2>nul
if not errorlevel 1 (
    start /B pythonw main.py >nul 2>&1
) else (
    start "" /B python main.py >nul 2>&1
)

exit
