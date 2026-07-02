@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo [publish] Baking version from git tag...
python scripts\bake_version.py
if errorlevel 1 (
    echo [publish] FAILED: bake_version
    exit /b 1
)

echo [publish] Cleaning dist...
if exist dist rmdir /s /q dist

echo [publish] Building wheel + sdist...
python -m build
if errorlevel 1 (
    echo [publish] FAILED: build
    exit /b 1
)

echo [publish] Uploading to PyPI...
python -m twine upload --non-interactive dist\*
if errorlevel 1 (
    echo [publish] FAILED: upload
    exit /b 1
)

echo [publish] %DATE% %TIME% — Done.
