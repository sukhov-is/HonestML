<#
.SYNOPSIS
  Answers what the loop's rounds cost, in one query across both drivers.

.DESCRIPTION
  Both drivers append one line per round to ops/loop/metrics.jsonl. This reader answers from it
  what a closed slice cost, whether the cache still holds and whether a round type is drifting
  longer, instead of leaving those questions to impressions.

  Rows are grouped by provider and round kind, because those are the two things that change what
  a round costs: a design round and an implementation round are different work, and the two
  providers price differently.

  Medians, not averages. A single killed round or one 4-hour outlier moves a mean far enough to
  hide the shape of everything else, and the question here is what a typical round of this kind
  costs. Totals are reported alongside for the one question a median cannot answer: what was
  actually spent.

  A round killed before its provider reported usage carries nulls, never zeroes. Those rounds are
  counted in a column of their own instead of being dropped, because their number is itself the
  finding: a period where a third of rounds have no usage is not a period of cheap rounds, it is
  one whose measurement is broken. A line too damaged to read at all - a record cut in half by a
  killed round - is counted the same way, on its own line above the table.

.PARAMETER Since
  Report only rounds recorded on or after this date. This is what makes a before/after comparison
  possible: run it twice around the change under test rather than eyeballing the tail of the file.
#>
[CmdletBinding()]
param(
    [datetime]$Since = [datetime]::MinValue
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$metricsPath = Join-Path $root 'ops\loop\metrics.jsonl'

function Get-FirstValue([object]$record, [string[]]$names) {
    if ($null -eq $record) { return $null }
    foreach ($name in $names) {
        $property = $record.PSObject.Properties[$name]
        if ($property -and $null -ne $property.Value) { return $property.Value }
    }
    return $null
}

# one flat view of a record whichever driver wrote it: the two spell the token fields differently
function ConvertTo-RoundRecord([object]$raw) {
    return [pscustomobject]@{
        Timestamp = [datetime](Get-FirstValue $raw @('ts'))
        Provider  = Get-FirstValue $raw @('provider')
        Kind      = Get-FirstValue $raw @('kind')
        Duration  = Get-FirstValue $raw @('duration_s')
        Cost      = Get-FirstValue $raw @('cost_usd')
        Input     = Get-FirstValue $raw @('input_tokens')
        Cached    = Get-FirstValue $raw @('cached_input_tokens', 'cache_read')
        Output    = Get-FirstValue $raw @('output_tokens', 'out_tokens')
        Turns     = Get-FirstValue $raw @('turns')
        Waits     = Get-FirstValue $raw @('wait_calls')
    }
}

function Get-Median([object[]]$values) {
    $measured = @($values | Where-Object { $null -ne $_ } | Sort-Object)
    if ($measured.Count -eq 0) { return $null }
    $middle = [int][math]::Floor($measured.Count / 2)
    if ($measured.Count % 2 -eq 1) { return $measured[$middle] }
    return ($measured[$middle - 1] + $measured[$middle]) / 2
}

function Format-Number([object]$value, [int]$decimals) {
    if ($null -eq $value) { return '-' }
    return ('{0:N' + $decimals + '}') -f $value
}

$row = '{0,-9} {1,-14} {2,7} {3,10} {4,10} {5,11} {6,10} {7,12} {8,7} {9,9}'
Write-Output ($row -f 'provider', 'kind', 'rounds', 'median s', 'median $', 'total $',
    'cache hit', 'median out', 'turns', 'no usage')

if (-not (Test-Path -LiteralPath $metricsPath)) {
    Write-Output 'no telemetry recorded'
    exit 0
}

$rounds = @()
$unreadable = 0
foreach ($line in [System.IO.File]::ReadLines($metricsPath)) {
    if (-not $line.Trim()) { continue }
    try { $rounds += ConvertTo-RoundRecord ($line | ConvertFrom-Json -ErrorAction Stop) }
    catch { $unreadable++ }
}
$rounds = @($rounds | Where-Object { $_.Timestamp -ge $Since })
if ($unreadable -gt 0) { Write-Output "$unreadable line(s) unreadable" }
if ($rounds.Count -eq 0) {
    Write-Output 'No rounds matched. Telemetry is written by a round, not by a self-test.'
    exit 0
}

$span = @($rounds | ForEach-Object { $_.Timestamp } | Sort-Object)
Write-Output ("--- {0} rounds | {1:yyyy-MM-dd} .. {2:yyyy-MM-dd}" -f
    $rounds.Count, $span[0], $span[$span.Count - 1])

foreach ($group in ($rounds | Group-Object { "$($_.Provider) $($_.Kind)" } | Sort-Object Name)) {
    $members = @($group.Group)
    $costs = @($members | ForEach-Object { $_.Cost } | Where-Object { $null -ne $_ })
    $totalCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Sum).Sum } else { $null }
    # the cache-hit share is a ratio of the group's sums, not a median of per-round ratios: one
    # short round at a poor ratio must not weigh the same as a round twenty times its size. Both
    # sums are taken over the SAME rounds - the ones reporting each half of the ratio - so a group
    # mixing schemas reports the share of what it measured, not one record set divided by another.
    $paired = @($members | Where-Object { $null -ne $_.Input -and $null -ne $_.Cached })
    $inputSum = ($paired | ForEach-Object { $_.Input } | Measure-Object -Sum).Sum
    $cachedSum = ($paired | ForEach-Object { $_.Cached } | Measure-Object -Sum).Sum
    $cacheHit = if ($inputSum) { '{0:N1}%' -f ($cachedSum / $inputSum * 100) } else { '-' }
    $unmeasured = @($members | Where-Object { $null -eq $_.Duration }).Count

    Write-Output ($row -f
        $members[0].Provider,
        $members[0].Kind,
        $members.Count,
        (Format-Number (Get-Median @($members | ForEach-Object { $_.Duration })) 0),
        (Format-Number (Get-Median $costs) 2),
        (Format-Number $totalCost 2),
        $cacheHit,
        (Format-Number (Get-Median @($members | ForEach-Object { $_.Output })) 0),
        (Format-Number (Get-Median @($members | ForEach-Object { $_.Turns })) 0),
        $unmeasured)
}
exit 0
