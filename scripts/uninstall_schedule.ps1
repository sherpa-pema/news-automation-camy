<#
.SYNOPSIS
    Unregisters (or previews with -DryRun) the Windows Scheduled Task "NepalNewsBot-Morning".

.PARAMETER DryRun
    Shows whether the task exists and what would be removed without making changes.

.PARAMETER TaskName
    Name of the scheduled task to remove (default: "NepalNewsBot-Morning").
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$TaskName = "NepalNewsBot-Morning"
)

$ErrorActionPreference = "Stop"

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($DryRun) {
    if ($null -eq $Existing) {
        Write-Output "[DRY RUN] Scheduled task '$TaskName' is not currently registered."
    } else {
        Write-Output "[DRY RUN] Would unregister Windows Scheduled Task '$TaskName' (State: $($Existing.State))."
    }
    exit 0
}

if ($null -eq $Existing) {
    Write-Output "[INFO] Scheduled task '$TaskName' is not registered; nothing to remove."
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Output "[OK] Unregistered Windows Scheduled Task '$TaskName'."
