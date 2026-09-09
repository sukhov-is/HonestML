<#
.SYNOPSIS
  Prune per-round driver logs, keeping the newest runs.

.DESCRIPTION
  Every round writes a stream log and an error log per segment, so ops/loop/logs grows by tens
  of MB across a few autonomous nights and buries the run you actually care about.

  Files are grouped by the run stamp that prefixes their name (YYYYMMDD-HHMMSS): the newest
  -KeepRuns stamps stay, older ones are deleted. Only round logs carry that prefix, so the
  artifacts sharing the directory — the round-guard marker, the gates verdict, the junit
  results read by ops/loop/report_test_cost.py — are never matched. Each of those is written
  once per run under a fixed name, so it needs no rotation and must survive one.
  Supports -WhatIf.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateRange(1, 100)]
    [int]$KeepRuns = 5
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$logDir = Join-Path $root 'ops\loop\logs'
if (-not (Test-Path -LiteralPath $logDir)) { exit 0 }

$stampPattern = '^(?<stamp>\d{8}-\d{6})-.*\.(out|err)$'
$logs = @(Get-ChildItem -LiteralPath $logDir -File |
        Where-Object { $_.Name -match $stampPattern })
$stamps = @($logs |
        ForEach-Object { [regex]::Match($_.Name, $stampPattern).Groups['stamp'].Value } |
        Sort-Object -Unique)
if ($stamps.Count -le $KeepRuns) {
    Write-Output "logs: $($stamps.Count) run(s) on disk, keep $KeepRuns - nothing to prune"
    exit 0
}

$kept = @($stamps | Select-Object -Last $KeepRuns)
$doomed = @($stamps | Where-Object { $kept -notcontains $_ })
$removed = 0
$freed = 0L
foreach ($log in $logs) {
    $stamp = [regex]::Match($log.Name, $stampPattern).Groups['stamp'].Value
    if ($doomed -notcontains $stamp) { continue }
    if ($PSCmdlet.ShouldProcess($log.Name, 'remove round log')) {
        $freed += $log.Length
        Remove-Item -LiteralPath $log.FullName -Force
        $removed++
    }
}
Write-Output ("logs: pruned {0} file(s) from {1} run(s), {2:N1} MB freed, newest {3} run(s) kept" -f
    $removed, $doomed.Count, ($freed / 1MB), $KeepRuns)
