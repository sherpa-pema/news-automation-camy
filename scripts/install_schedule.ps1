<#
.SYNOPSIS
    Installs (or previews with -DryRun) the Windows Scheduled Task "NepalNewsBot-Morning".

.DESCRIPTION
    Registers a daily scheduled task that runs at 07:00 local time:
      .venv\Scripts\python.exe run_daily.py --morning
    with WorkingDirectory set to the project root, stdout/stderr appended to:
      logs\morning_<YYYY-MM-DD>.log
    configured to run as soon as possible after a missed start (-StartWhenAvailable),
    and set to run only when the current user is logged on (LogonType Interactive).

    IMPORTANT REQUIREMENT:
      Your PC must be powered ON and you must be SIGNED IN to your Windows account
      for the task to run at 07:00 (or immediately after waking/signing in if missed).
      Windows will NOT wake a powered-off PC or run this task while signed out.

.PARAMETER DryRun
    Prints the exact task definition that would be registered without creating or
    modifying any Windows scheduled task.

.PARAMETER TaskName
    Name of the scheduled task (default: "NepalNewsBot-Morning").

.PARAMETER Time
    Local daily run time in HH:mm format (default: "07:00").
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$TaskName = "NepalNewsBot-Morning",
    [string]$Time = "07:00"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunDailyPy = Join-Path $RepoRoot "run_daily.py"
$LogsDir = Join-Path $RepoRoot "logs"
$LogPattern = "logs\morning_<YYYY-MM-DD>.log"

if (-not (Test-Path $RunDailyPy)) {
    throw "run_daily.py not found at expected path: $RunDailyPy"
}
if (-not (Test-Path $PythonExe)) {
    if ($DryRun) {
        Write-Warning ".venv Python executable not found at: $PythonExe"
    } else {
        throw ".venv Python executable not found at: $PythonExe"
    }
}

$CurrentUser = if ($env:USERDOMAIN) { "$env:USERDOMAIN\$env:USERNAME" } else { $env:USERNAME }

# Wrap .venv\Scripts\python.exe run_daily.py --morning so stdout+stderr append
# to logs\morning_<YYYY-MM-DD>.log using locale-independent ISO date formatting.
$EscapedPython = $PythonExe.Replace("'", "''")
$PsCommand = (
    "`$env:PYTHONIOENCODING='utf-8'; " +
    "New-Item -ItemType Directory -Force -Path 'logs' | Out-Null; " +
    "`$d = Get-Date -Format 'yyyy-MM-dd'; " +
    "`$log = Join-Path 'logs' ('morning_' + `$d + '.log'); " +
    "& '$EscapedPython' run_daily.py --morning 2>&1 | " +
    "ForEach-Object { if (`$_ -is [System.Management.Automation.ErrorRecord]) { `$_.Exception.Message } else { `$_ } } | " +
    "Out-File -FilePath `$log -Append -Encoding utf8; " +
    "exit `$LASTEXITCODE"
)
$ActionExecute = "powershell.exe"
$ActionArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$PsCommand`""
$Description = (
    "Daily Nepal News Bot morning prep (.venv\Scripts\python.exe run_daily.py --morning). " +
    "Requires PC powered on and user signed in (Interactive logon). Logs to logs\morning_<date>.log."
)

$Action = New-ScheduledTaskAction -Execute $ActionExecute -Argument $ActionArguments -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)
$Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited

if ($DryRun) {
    Write-Output "================================================================="
    Write-Output " DRY RUN - Windows Scheduled Task Preview (NOT REGISTERED)"
    Write-Output "================================================================="
    Write-Output "Task Name:         $TaskName"
    Write-Output "Schedule:          Daily at $Time (local PC time)"
    Write-Output "Missed Start:      StartWhenAvailable (runs as soon as possible)"
    Write-Output "Logon Condition:   Interactive (runs ONLY while '$CurrentUser' is logged on)"
    Write-Output "Executable:        $PythonExe"
    Write-Output "Python Arguments:  run_daily.py --morning"
    Write-Output "Working Directory: $RepoRoot"
    Write-Output "Log File Pattern:  $LogsDir\morning_<YYYY-MM-DD>.log ($LogPattern)"
    Write-Output ""
    Write-Output "Task Scheduler Wrapper Action:"
    Write-Output "  Execute:          $($Action.Execute)"
    Write-Output "  Arguments:        $($Action.Arguments)"
    Write-Output "  WorkingDirectory: $($Action.WorkingDirectory)"
    Write-Output ""
    Write-Output "IMPORTANT REQUIREMENT:"
    Write-Output "  The PC must be turned ON and you must be SIGNED IN to Windows for"
    Write-Output "  this scheduled task to run at $Time (or ASAP after a missed start)."
    Write-Output "================================================================="
    Write-Output "[DRY RUN] No scheduled task was created or modified."
    exit 0
}

if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description $Description `
    -Force | Out-Null

Write-Output "[OK] Registered Windows Scheduled Task '$TaskName'."
Write-Output "  Schedule:          Daily at $Time (local PC time)"
Write-Output "  Command:           $PythonExe run_daily.py --morning"
Write-Output "  Working Directory: $RepoRoot"
Write-Output "  Log File:          $LogsDir\morning_<YYYY-MM-DD>.log"
Write-Output "  Missed Start:      StartWhenAvailable enabled"
Write-Output "  Principal:         $CurrentUser (Interactive - PC must be ON and signed in)"
