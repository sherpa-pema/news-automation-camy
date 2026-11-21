<#
.SYNOPSIS
    Installs (or previews with -DryRun) the Windows Scheduled Tasks:
    1. NepalNewsBot-Morning (daily morning editorial prep)
    2. NepalNewsBot-Cleanup (weekly SSD retention and cache cleanup)

.DESCRIPTION
    Registers:
    1. Daily scheduled task running at 07:00 local time from C:\news-auto:
         C:\news-auto\.venv\Scripts\python.exe run_daily.py --morning
       appending logs to logs\morning_<YYYY-MM-DD>.log
    2. Weekly scheduled task running on Sundays at 03:00 local time:
         C:\news-auto\.venv\Scripts\python.exe run_daily.py --cleanup --keep-runs 14
       appending logs to logs\cleanup_<YYYY-MM-DD>.log

    Configured to:
      - wake the PC from sleep/hibernation to run (-WakeToRun),
      - run as soon as possible after a missed start (-StartWhenAvailable),
      - run on battery without stopping (-AllowStartIfOnBatteries / -DontStopIfGoingOnBatteries),
      - run under the current user's signed-in session (LogonType Interactive).

.PARAMETER DryRun
    Prints the exact task definitions that would be registered without creating or
    modifying any Windows scheduled tasks.

.PARAMETER TaskName
    Name of the morning scheduled task (default: "NepalNewsBot-Morning").

.PARAMETER Time
    Local daily run time in HH:mm format (default: "07:00").

.PARAMETER CleanupTaskName
    Name of the cleanup scheduled task (default: "NepalNewsBot-Cleanup").

.PARAMETER CleanupTime
    Local weekly run time in HH:mm format (default: "03:00").

.PARAMETER CleanupDay
    Day of week for weekly cleanup (default: "Sunday").

.PARAMETER KeepRuns
    Retention threshold in dated runs to keep intact (default: 14).

.PARAMETER ProjectRoot
    Project directory containing run_daily.py and .venv (default: "C:\news-auto").
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$TaskName = "NepalNewsBot-Morning",
    [string]$Time = "07:00",
    [string]$CleanupTaskName = "NepalNewsBot-Cleanup",
    [string]$CleanupTime = "03:00",
    [string]$CleanupDay = "Sunday",
    [int]$KeepRuns = 14,
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
$EscapedPython = $PythonExe.Replace("'", "''")

# 1. Morning Task Definition
$PsCommand = (
    "`$env:PYTHONIOENCODING='utf-8'; " +
    "New-Item -ItemType Directory -Force -Path 'logs' | Out-Null; " +
    "`$d = Get-Date -Format 'yyyy-MM-dd'; " +
    "`$log = Join-Path 'logs' ('morning_' + `$d + '.log'); " +
    "& '$EscapedPython' run_daily.py --morning --automated 2>&1 | " +
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

# 2. Weekly Cleanup Task Definition
$CleanupPsCommand = (
    "`$env:PYTHONIOENCODING='utf-8'; " +
    "New-Item -ItemType Directory -Force -Path 'logs' | Out-Null; " +
    "`$d = Get-Date -Format 'yyyy-MM-dd'; " +
    "`$log = Join-Path 'logs' ('cleanup_' + `$d + '.log'); " +
    "& '$EscapedPython' run_daily.py --cleanup --keep-runs $KeepRuns 2>&1 | " +
    "ForEach-Object { if (`$_ -is [System.Management.Automation.ErrorRecord]) { `$_.Exception.Message } else { `$_ } } | " +
    "Out-File -FilePath `$log -Append -Encoding utf8; " +
    "exit `$LASTEXITCODE"
)
$CleanupActionArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$CleanupPsCommand`""
$CleanupDescription = (
    "Weekly Nepal News Bot retention cleanup ($PythonExe run_daily.py --cleanup --keep-runs $KeepRuns in $RepoRoot). " +
    "Purges temporary artifacts and stale render/cache/*.mp4 files older than $KeepRuns runs. Logs to logs\cleanup_<date>.log."
)

$CleanupAction = New-ScheduledTaskAction -Execute $ActionExecute -Argument $CleanupActionArguments -WorkingDirectory $RepoRoot
$CleanupTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $CleanupTime
$CleanupSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

if ($DryRun) {
    Write-Output "================================================================="
    Write-Output " DRY RUN - Windows Scheduled Task Preview (NOT REGISTERED)"
    Write-Output "================================================================="
    Write-Output "Task 1:            $TaskName"
    Write-Output "Schedule:          Daily at $Time (local PC time)"
    Write-Output "Missed Start:      StartWhenAvailable (runs as soon as possible)"
    Write-Output "Wake PC:           WakeToRun (wakes PC from sleep if wake timers enabled)"
    Write-Output "Logon Condition:   Interactive (runs ONLY while '$CurrentUser' is signed in)"
    Write-Output "Executable:        $PythonExe"
    Write-Output "Python Arguments:  run_daily.py --morning"
    Write-Output "Working Directory: $RepoRoot"
    Write-Output "Log File Pattern:  $LogsDir\morning_<YYYY-MM-DD>.log ($LogPattern)"
    Write-Output "-----------------------------------------------------------------"
    Write-Output "Task 2:            $CleanupTaskName"
    Write-Output "Schedule:          Weekly on $CleanupDay at $CleanupTime (local PC time)"
    Write-Output "Retention Window:  Keep last $KeepRuns dated runs intact"
    Write-Output "Python Arguments:  run_daily.py --cleanup --keep-runs $KeepRuns"
    Write-Output "Working Directory: $RepoRoot"
    Write-Output "Log File Pattern:  $LogsDir\cleanup_<YYYY-MM-DD>.log"
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

Register-ScheduledTask `
    -TaskName $CleanupTaskName `
    -Action $CleanupAction `
    -Trigger $CleanupTrigger `
    -Settings $CleanupSettings `
    -Principal $Principal `
    -Description $CleanupDescription `
    -Force | Out-Null

Write-Output "[OK] Registered Windows Scheduled Tasks:"
Write-Output "  1. '$TaskName' (Daily at $Time, run_daily.py --morning)"
Write-Output "  2. '$CleanupTaskName' (Weekly on $CleanupDay at $CleanupTime, run_daily.py --cleanup --keep-runs $KeepRuns)"
Write-Output "  Working Directory: $RepoRoot"
Write-Output "  Missed Start:      StartWhenAvailable enabled"
Write-Output "  Wake PC:           WakeToRun enabled"
Write-Output "  Principal:         $CurrentUser (Interactive - user session must be signed in)"
