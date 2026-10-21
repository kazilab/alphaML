@echo off
rem Start alphaML 2.0 in Docker and open it in the web browser (double-click this file).
rem Usage: start-alphaml.cmd [port]
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-alphaml.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-alphaml.ps1" -Port %1
)
if errorlevel 1 pause
