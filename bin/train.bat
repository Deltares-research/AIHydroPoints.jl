@echo off
setlocal enabledelayedexpansion
rem bin\train.bat — run the AIHydroPoints training pipeline
rem
rem Usage:
rem   bin\train.bat path\to\settings.toml

set "REPO=%~dp0.."
set "SCRIPT=%REPO%\scripts\train.jl"

where pixi >nul 2>&1
if not %errorlevel% == 0 goto try_julia
pixi run julia "--project=%REPO%" "%SCRIPT%" %*
exit /b !errorlevel!

:try_julia
where julia >nul 2>&1
if not %errorlevel% == 0 goto no_runner
julia "--project=%REPO%" "%SCRIPT%" %*
exit /b !errorlevel!

:no_runner
echo Error: neither 'pixi' nor 'julia' found on PATH. 1>&2
echo Install Julia (https://julialang.org/downloads) or pixi (https://pixi.sh). 1>&2
exit /b 1
