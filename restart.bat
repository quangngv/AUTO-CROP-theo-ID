@echo off
chcp 65001 >nul
echo ================================
echo  Dang khoi dong lai ung dung...
echo ================================

REM Tat app cu bang ten cua so (khong dong cac Python app khac)
taskkill /fi "WINDOWTITLE eq AUTO CROP*" /f >nul 2>&1

timeout /t 1 /nobreak >nul

echo Khoi dong lai...
start "" /B pythonw main.py >nul 2>&1
exit
