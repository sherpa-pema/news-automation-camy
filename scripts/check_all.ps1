<#
.SYNOPSIS
    Nepal News Bot CI & Golden Broadcast Regression Gate.

.DESCRIPTION
    Runs all quality and regression gates in one command:
      1. Bytecode compilation across src, web_office, tools, run_daily.py
      2. Schema & YAML validation across config/*.yaml and contracts/*.schema.json
      3. Full pytest suite (including Devanagari, render edge-cases, 16:9/9:16 E2E, and golden gate)

.EXAMPLE
    .\scripts\check_all.ps1
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\news-auto"
)

$ErrorActionPreference = "Stop"

if (Test-Path $ProjectRoot) {
    $RepoRoot = (Resolve-Path $ProjectRoot).Path
} else {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PytestExe = Join-Path $RepoRoot ".venv\Scripts\pytest.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error ".venv Python executable not found at: $PythonExe"
    exit 1
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Nepal News Bot — Quality & Regression Gate (check_all) " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Project Root: $RepoRoot"
Write-Host "Python:       $PythonExe"
Write-Host ""

# 1. Bytecode Compilation Gate
Write-Host "[1/3] Checking Python bytecode compilation (compileall)..." -ForegroundColor Yellow
& $PythonExe -m compileall -q src web_office tools run_daily.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Python syntax / bytecode compilation error detected." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "  --> [PASS] All source files compiled cleanly." -ForegroundColor Green
Write-Host ""

# 2. Schema and Config Validation Gate
Write-Host "[2/3] Validating YAML configs and JSON Schema contracts..." -ForegroundColor Yellow
& $PythonExe tools/validate_schemas_and_configs.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Config or JSON Schema validation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host ""

# 3. Full Pytest Suite Gate
Write-Host "[3/3] Running full automated test suite (pytest -q)..." -ForegroundColor Yellow
& $PytestExe -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Test suite regression detected." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host " [PASS] ALL GATES PASSED (Compile + Schemas + Pytest)   " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
exit 0
