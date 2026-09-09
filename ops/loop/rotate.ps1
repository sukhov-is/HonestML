<#
.SYNOPSIS
  Rotate the loop ledger: move old rows of the "Раунды" table from docs/loop/state.md
  to docs/loop/state-archive.md, keeping the newest -Keep rows.

.DESCRIPTION
  The ledger grows by one multi-KB row per round; unrotated, every round pays for the
  whole history on every model turn. Rotation triggers only when the table holds more
  than -TriggerAt data rows (hysteresis: one rotation commit per ~Keep rounds, not per round).

  Driver contract preserved: data rows keep matching '^\|\s*\d+\s*\|' in file order and the
  newest row stays last (Get-RoundStatuses in both loop drivers reads statuses in file order and
  only uses the tail). The pointer line inserted between the heading and the table header is prose
  and never matches the row regex. Files are written UTF-8 without BOM.

  Fails loud (non-zero) if the table layout is not the expected contiguous block —
  never rewrites a file it does not fully understand.

.PARAMETER Keep      Rows to keep in state.md after a rotation (default 20).
.PARAMETER TriggerAt Rotate only when the table holds more than this many rows (default 40).
#>
[CmdletBinding()]
param(
    [int]$Keep = 20,
    [int]$TriggerAt = 40
)

$ErrorActionPreference = 'Stop'
$utf8        = [System.Text.UTF8Encoding]::new($false)
$root        = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$statePath   = Join-Path $root 'docs\loop\state.md'
$archivePath = Join-Path $root 'docs\loop\state-archive.md'

$lines = [System.IO.File]::ReadAllLines($statePath, $utf8)

# locate the rounds table: '## Раунды' heading, then the '| round |' header + separator
$iHead = -1
for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^##\s+Раунды') { $iHead = $i; break } }
if ($iHead -lt 0) { throw "state.md: '## Раунды' heading not found" }
$iCols = -1
for ($i = $iHead + 1; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^\|\s*round\s*\|') { $iCols = $i; break } }
if ($iCols -lt 0) { throw "state.md: '| round |' table header not found" }
$iSep = $iCols + 1
if ($lines[$iSep] -notmatch '^\|[-:\s|]+$') { throw "state.md: separator row expected at line $($iSep + 1)" }

# data rows (same regex as the driver's status parser)
$dataIdx = @()
for ($i = $iSep + 1; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^\|\s*\d+\s*\|') { $dataIdx += $i } }
if ($dataIdx.Count -le $TriggerAt) { exit 0 }

# refuse to rewrite a table we do not fully understand: between the separator and the last
# data row only data rows and blank lines are allowed (blanks are dropped on rebuild)
for ($i = $iSep + 1; $i -le $dataIdx[-1]; $i++) {
    if ($lines[$i] -notmatch '^\|\s*\d+\s*\|' -and $lines[$i].Trim() -ne '') {
        throw "state.md: unexpected non-row line $($i + 1) inside the rounds table — rotate manually"
    }
}

$cut     = $dataIdx.Count - $Keep
$oldRows = @($dataIdx[0..($cut - 1)] | ForEach-Object { $lines[$_] })
$keepRows = @($dataIdx[$cut..($dataIdx.Count - 1)] | ForEach-Object { $lines[$_] })

# archived round-number range, merged with a previous pointer if present
$nums = @($oldRows | ForEach-Object { [int]((($_ -split '\|')[1]).Trim()) })
$min = ($nums | Measure-Object -Minimum).Minimum
$max = ($nums | Measure-Object -Maximum).Maximum
for ($i = $iHead + 1; $i -lt $iCols; $i++) {
    if ($lines[$i] -match 'Раунды\s+(\d+)\D(\d+)\s') {
        if ([int]$Matches[1] -lt $min) { $min = [int]$Matches[1] }
        if ([int]$Matches[2] -gt $max) { $max = [int]$Matches[2] }
    }
}

# append to the archive (create with a matching table header on first rotation)
if (-not (Test-Path $archivePath)) {
    $header = @(
        '# HonestML Loop — Архив гроссбуха',
        '',
        'Старые строки таблицы «Раунды» из `docs/loop/state.md` (переносит `ops/loop/rotate.ps1`;',
        'порядок строк — исходный файловый). Живой хвост таблицы — в `state.md`.',
        '',
        $lines[$iCols],
        $lines[$iSep]
    )
    [System.IO.File]::WriteAllLines($archivePath, [string[]]$header, $utf8)
}
[System.IO.File]::AppendAllLines($archivePath, [string[]]$oldRows, $utf8)

# rebuild state.md: prefix through the heading, fresh pointer, table header, kept rows, original tail
$pointer = "Раунды $min–$max — в ``docs/loop/state-archive.md`` (ротация драйвером; здесь — последние строки, свежайшая внизу)."
$tail = @()
if ($dataIdx[-1] + 1 -le $lines.Count - 1) { $tail = @($lines[($dataIdx[-1] + 1)..($lines.Count - 1)]) }
$new = [System.Collections.Generic.List[string]]::new()
$new.AddRange([string[]]$lines[0..$iHead])
$new.Add('')
$new.Add($pointer)
$new.Add('')
$new.Add($lines[$iCols])
$new.Add($lines[$iSep])
$new.AddRange([string[]]$keepRows)
if ($tail.Count -gt 0) { $new.AddRange([string[]]$tail) }
[System.IO.File]::WriteAllLines($statePath, $new, $utf8)

Write-Output "rotated: $cut rows (rounds $min-$max range) -> docs/loop/state-archive.md; kept last $Keep"
