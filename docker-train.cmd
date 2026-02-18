@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0docker-train.ps1" %*
exit /b %errorlevel%
