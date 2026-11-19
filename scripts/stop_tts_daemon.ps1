# Stops persistent Ming-3.0-TTS daemon
param(
    [int]$Port = 8766,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "SilentlyContinue"
$ShutdownUrl = "http://${HostName}:${Port}/shutdown"
$HealthUrl = "http://${HostName}:${Port}/health"

Write-Host "[stop_tts_daemon] Attempting graceful shutdown via $ShutdownUrl..."

try {
    $resp = Invoke-RestMethod -Uri $ShutdownUrl -Method Post -TimeoutSec 3
    if ($resp.status -eq "shutting_down") {
        Write-Host "[stop_tts_daemon] Shutdown signal accepted by PID $($resp.pid). Waiting for process exit..."
        Start-Sleep -Seconds 2
    }
} catch {
    Write-Host "[stop_tts_daemon] Daemon HTTP endpoint unreachable or already stopped."
}

# Verify port is released
$running = $true
try {
    $resp = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 1
    if ($resp.status -eq "ok") {
        Write-Warning "[stop_tts_daemon] Daemon still responding. Terminating PID $($resp.pid)..."
        Stop-Process -Id $resp.pid -Force -ErrorAction SilentlyContinue
    }
} catch {
    $running = $false
}

if (Test-Path "logs/tts_daemon.pid") {
    Remove-Item "logs/tts_daemon.pid" -Force
}

Write-Host "[stop_tts_daemon] TTS Daemon stopped successfully."
