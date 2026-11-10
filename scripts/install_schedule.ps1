<#
.SYNOPSIS
    Installs (or previews with -DryRun) the Windows Scheduled Task "NepalNewsBot-Morning".

.DESCRIPTION
    Registers a daily scheduled task that runs at 07:00 local time from C:\news-auto:
      C:\news-auto\.venv\Scripts\python.exe run_daily.py --morning
    with WorkingDirectory set to C:\news-auto (or -ProjectRoot), stdout/stderr appended to:
      logs\morning_<YYYY-MM-DD>.log
    configured to:
      - wake the PC from sleep/hibernation to run (-WakeToRun),
      - run as soon as possible after a missed start (-StartWhenAvailable),
      - run on battery without stopping (-AllowStartIfOnBatteries / -DontStopIfGoingOnBatteries),
      - run under the current user's signed-in session (LogonType Interactive).

    IMPORTANT REQUIREMENT:
      - "-WakeToRun" can wake a PC from Windows Sleep/Modern Standby or Hibernation
        if Windows Power Options ("Allow wake timers") and BIOS wake timers allow it.
      - Windows CANNOT wake a completely powered-off (shut down) PC.
      - Because LogonType is Interactive, you must remain signed into your Windows
        account (screen locked is usually fine while signed in, unless wake requires logon
        re-authentication or the account is completely signed out).

.PARAMETER DryRun
    Prints the exact task definition that would be registered without creating or
    modifying any Windows scheduled task.

.PARAMETER TaskName
    Name of the scheduled task (default: "NepalNewsBot-Morning").

.PARAMETER Time
    Local daily run time in HH:mm format (default: "07:00").

.PARAMETER ProjectRoot
    Project directory containing run_daily.py and .venv (default: "C:\news-auto").
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$TaskName = "NepalNewsBot-Morning",
    [string]$Time = "07:00",
    [string]$ProjectRoot = "C:\news-auto"
)

$ErrorActionPreference = "Stop"

if (Test-Path $ProjectRoot) {
    $RepoRoot = (Resolve-Path $ProjectRoot).Path
} else {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
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
    "Daily Nepal News Bot morning prep ($PythonExe run_daily.py --morning in $RepoRoot). " +
    "WakeToRun enabled for sleep timers; requires signed-in Interactive session. Logs to logs\morning_<date>.log."
)

$Action = New-ScheduledTaskAction -Execute $ActionExecute -Argument $ActionArguments -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
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
    Write-Output "Wake PC:           WakeToRun (wakes PC from sleep if wake timers enabled)"
    Write-Output "Logon Condition:   Interactive (runs ONLY while '$CurrentUser' is signed in)"
    Write-Output "Executable:        $PythonExe"
    Write-Output "Python Arguments:  run_daily.py --morning"
    Write-Output "Working Directory: $RepoRoot"
    Write-Output "Log File Pattern:  $LogsDir\morning_<YYYY-MM-DD>.log ($LogPattern)"
    Write-Output ""
    Write-Output "Task Scheduler Settings:"
    Write-Output "  StartWhenAvailable:        $($Settings.StartWhenAvailable)"
    Write-Output "  WakeToRun:                 $($Settings.WakeToRun)"
    Write-Output "  AllowStartIfOnBatteries:   $($Settings.DisallowStartIfOnBatteries -eq $false)"
    Write-Output "  DontStopIfGoingOnBatteries: $($Settings.StopIfGoingOnBatteries -eq $false)"
    Write-Output "  ExecutionTimeLimit:        $($Settings.ExecutionTimeLimit)"
    Write-Output ""
    Write-Output "Task Scheduler Wrapper Action:"
    Write-Output "  Execute:          $($Action.Execute)"
    Write-Output "  Arguments:        $($Action.Arguments)"
    Write-Output "  WorkingDirectory: $($Action.WorkingDirectory)"
    Write-Output ""
    Write-Output "IMPORTANT REQUIREMENT:"
    Write-Output "  WakeToRun wakes the PC from Windows Sleep (when Power Options 'Allow wake timers'"
    Write-Output "  and BIOS wake timers are enabled), but the PC must NOT be completely powered off"
    Write-Output "  and '$CurrentUser' must be SIGNED IN to Windows for the Interactive task to run at $Time"
    Write-Output "  (or ASAP after a missed start)."
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
Write-Output "  Wake PC:           WakeToRun enabled"
Write-Output "  Principal:         $CurrentUser (Interactive - user session must be signed in)"
