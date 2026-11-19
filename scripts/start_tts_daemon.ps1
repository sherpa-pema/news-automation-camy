# Starts persistent Ming-3.0-TTS daemon for low-latency synthesis
param(
    [int]$Port = 8766,
    [string]$HostName = "127.0.0.1",
    [int]$IdleTtlSec = 1800,
    [string]$ModelPath = "models/Ming-3.0-tts-0.5b",
    [string]$Device = "cuda:0",
    [string]$Dtype = "bfloat16"
)

$ErrorActionPreference = "Stop"
$PythonPath = ".venv-tts\Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    Write-Error "Python environment not found at $PythonPath. Ensure .venv-tts is set up."
    exit 1
}

$HealthUrl = "http://${HostName}:${Port}/health"

# Check if already running
try {
    $existing = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($existing.status -eq "ok") {
        Write-Host "[start_tts_daemon] TTS Daemon is already running (PID=$($existing.pid)) on http://${HostName}:${Port}"
        exit 0
    }
} catch {}

Write-Host "[start_tts_daemon] Starting Ming-3.0-TTS daemon on http://${HostName}:${Port} (device=$Device, dtype=$Dtype, idle_ttl=${IdleTtlSec}s)..."

$proc = Start-Process -FilePath $PythonPath `
    -ArgumentList "tools/ming_worker.py --serve --host $HostName --port $Port --model-path $ModelPath --device $Device --dtype $Dtype --idle-ttl-sec $IdleTtlSec" `
    -RedirectStandardOutput "logs/tts_daemon.log" `
    -RedirectStandardError "logs/tts_daemon_err.log" `
    -PassThru -WindowStyle Hidden

Write-Host "[start_tts_daemon] Process launched (PID=$($proc.Id)). Waiting for model load and health check..."

$timeout = 60
$elapsed = 0
$ready = $false

while ($elapsed -lt $timeout) {
    Start-Sleep -Seconds 2
    $elapsed += 2
    try {
        $resp = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($resp.status -eq "ok") {
            Write-Host "[start_tts_daemon] TTS Daemon is READY! PID=$($resp.pid), Model=$($resp.model_path), Device=$($resp.device)"
            $ready = $true
            Set-Content -Path "logs/tts_daemon.pid" -Value $resp.pid -Force
            break
        }
    } catch {
        # Process might have died prematurely
        if ($proc.HasExited) {
            Write-Error "[start_tts_daemon] Daemon process exited unexpectedly with code $($proc.ExitCode)."
            exit 1
        }
    }
}

if (-not $ready) {
    Write-Error "[start_tts_daemon] Timed out waiting for TTS daemon to initialize after $timeout seconds."
    exit 1
}
