<#
.SYNOPSIS
  One-call round orientation for the HonestML loop (LOOP.md §0): prints exactly what a round
  needs to pick its work, instead of full reads of the large state files.

.DESCRIPTION
  Every model turn re-reads the whole context, so orientation ballast is paid on every
  subsequent turn of the round — this script collapses 6-10 exploratory reads into one
  compact tool result.

  Prints, in order:
    - graceful STOP marker state (if present);
    - the git baseline: the last commit and, when the tree is dirty, the first paths of the
      interrupted round's WIP;
    - non-done rows of the ledger's items table, in full (they carry the design-package
      paths an IMPLEMENT round needs), plus a one-line count of done rows;
    - the archive pointer and the last -Tail rows of the rounds table — older rows clipped
      to -RowClip chars, the newest row printed in full (its note carries the handoff);
    - blockers: every open one in full (they are live state), plus the newest resolved
      decisions in full and older ones as a heading line;
    - backlog: the epic summary (## Сводка, whose row order is the order of picking work)
      and the planning-discovered additions.
  Backlog epic sections (### EPIC-) and design-package files are intentionally NOT printed —
  the round reads those точечно (Grep anchor + Read offset/limit) after choosing a slice.

.PARAMETER Tail    Rounds-table rows to print (default 10).
.PARAMETER RowClip Char clip for all but the newest rounds row (default 500).
.PARAMETER KeepResolvedBlockers
                   Newest resolved blockers printed in full (default 5); older ones print as
                   their identity alone — number, item, state. Open blockers are always
                   printed in full.
#>
[CmdletBinding()]
param(
    [int]$Tail = 10,
    [int]$RowClip = 500,
    [ValidateRange(0, 1000)]
    [int]$KeepResolvedBlockers = 5
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Read-Lines([string]$rel) { [System.IO.File]::ReadAllLines((Join-Path $root $rel), $utf8) }

if (Test-Path (Join-Path $root 'docs\loop\STOP')) {
    Write-Output 'INFO docs/loop/STOP присутствует: текущий раунд заверши и закоммить; драйвер остановится перед следующим.'
}

# working-tree baseline: an interrupted round leaves its WIP on disk — make that visible
# to the fresh context instead of letting it discover (or miss) the leftovers ad hoc
$lastCommit = [string](git -C $root log -1 --format='%h %s')
if ($lastCommit.Length -gt 220) { $lastCommit = $lastCommit.Substring(0, 220) + '…' }
Write-Output '=== База (git) ==='
Write-Output "Последний коммит: $lastCommit"
$dirty = @(git -C $root status --porcelain)
if ($dirty.Count -gt 0) {
    Write-Output "!! Рабочее дерево ГРЯЗНОЕ ($($dirty.Count) путей) — вероятно, WIP оборванного раунда. Продолжай от файлов, не отбрасывай:"
    $dirty | Select-Object -First 10 | Write-Output
} else {
    Write-Output 'Рабочее дерево чистое.'
}
Write-Output ''

$state = Read-Lines 'docs\loop\state.md'
$iItems = -1; $iRounds = -1
for ($i = 0; $i -lt $state.Count; $i++) {
    if ($iItems -lt 0 -and $state[$i] -match '^##\s+Статус пунктов') { $iItems = $i }
    if ($state[$i] -match '^##\s+Раунды') { $iRounds = $i; break }
}
if ($iItems -lt 0 -or $iRounds -lt 0) { throw 'state.md: таблицы «Статус пунктов»/«Раунды» не найдены' }

# --- items table: non-done rows in full, done rows as a count ---
$done = 0
$notDone = [System.Collections.Generic.List[string]]::new()
for ($i = $iItems + 1; $i -lt $iRounds; $i++) {
    $row = $state[$i]
    if ($row -notmatch '^\|') { continue }
    if ($row -match '^\|[-:\s|]+$') { continue }
    if ($row -match 'id пункта') { continue }
    $phase = (($row -split '\|')[2]).Trim()
    if ($phase -eq 'done') { $done++ } else { $notDone.Add($row) }
}
Write-Output "=== Статус пунктов: не-done (done: $done строк — целиком в docs/loop/state.md) ==="
$notDone | Write-Output

# --- rounds table: archive pointer + tail ---
Write-Output ''
Write-Output "=== Раунды: последние $Tail (старшие обрезаны до $RowClip симв., последняя — целиком) ==="
for ($i = $iRounds + 1; $i -lt $state.Count; $i++) {
    if ($state[$i] -match 'state-archive\.md') { Write-Output $state[$i]; break }
    if ($state[$i] -match '^\|') { break }
}
$rows = @()
for ($i = $iRounds + 1; $i -lt $state.Count; $i++) { if ($state[$i] -match '^\|\s*\d+\s*\|') { $rows += $state[$i] } }
if ($rows.Count -gt $Tail) { $rows = $rows[($rows.Count - $Tail)..($rows.Count - 1)] }
for ($i = 0; $i -lt $rows.Count; $i++) {
    $r = $rows[$i]
    if ($i -lt $rows.Count - 1 -and $r.Length -gt $RowClip) { $r = $r.Substring(0, $RowClip) + '…' }
    Write-Output $r
}

# --- blockers: every open one in full, resolved ones clipped to their heading past the newest few ---
# An open blocker is live state and is never clipped. A resolved one is a decision already taken:
# the newest stay in full because a round may still be acting on them, older ones keep only their
# heading so orientation stops growing with the project's whole decision history.
$blockerRel = 'docs/loop/blockers.md'
$blockerLines = Read-Lines 'docs\loop\blockers.md'
$heads = @()
for ($i = 0; $i -lt $blockerLines.Count; $i++) { if ($blockerLines[$i] -match '^##\s+BLK-') { $heads += $i } }
$resolved = @()
for ($s = 0; $s -lt $heads.Count; $s++) { if ($blockerLines[$heads[$s]] -notmatch '\bopen\b') { $resolved += $s } }
$clipped = @()
if ($resolved.Count -gt $KeepResolvedBlockers) {
    $clipped = @($resolved[0..($resolved.Count - $KeepResolvedBlockers - 1)])
}
Write-Output ''
Write-Output "=== Блокеры ($blockerRel; открытые целиком, решения — свежайшие $KeepResolvedBlockers целиком, остальные номером и пунктом) ==="
if ($heads.Count -eq 0) {
    $blockerLines | Write-Output
} else {
    if ($heads[0] -gt 0) { $blockerLines[0..($heads[0] - 1)] | Write-Output }
    for ($s = 0; $s -lt $heads.Count; $s++) {
        $from = $heads[$s]
        $to = if ($s -lt $heads.Count - 1) { $heads[$s + 1] - 1 } else { $blockerLines.Count - 1 }
        if ($clipped -contains $s) {
            # the owner's decision is written into the heading, so identity here is the heading up
            # to the state word: the number and the item are what a round navigates by, the
            # decision itself is fetched with the Grep the line names
            $head = $blockerLines[$from]
            if ($head -match '^##\s+BLK-\S+\s+·\s+\S+\s+·\s+answered') { $head = $Matches[0] }
            elseif ($head.Length -gt $RowClip) { $head = $head.Substring(0, $RowClip) + '…' }
            Write-Output "$head  — решён; текст: Grep по $blockerRel"
        } else {
            $blockerLines[$from..$to] | Write-Output
        }
    }
}

# --- backlog: сводка эпиков и реестр «дыр полноты» ---
# Секции адресуются именем, а не номером: раунд читает backlog каждый раз, и перенумерация
# секций не должна его ослеплять. Не найденная секция называет себя и не отменяет остальную
# ориентацию — граница секции всюду одна и та же, следующий заголовок уровня `##`.
$b = Read-Lines 'docs\loop\backlog.md'
$sections = @(
    @{ Title = '=== Backlog — сводка эпиков (## Сводка, docs/loop/backlog.md; порядок строк = порядок выбора работы) ==='; Anchor = 'Сводка' },
    @{ Title = '=== Backlog — Planning-discovered дополнения ==='; Anchor = 'Planning-discovered' }
)
foreach ($section in $sections) {
    $start = -1
    for ($i = 0; $i -lt $b.Count; $i++) {
        if ($b[$i] -match '^##\s' -and $b[$i] -match [regex]::Escape($section.Anchor)) { $start = $i; break }
    }
    Write-Output ''
    if ($start -lt 0) {
        Write-Output "$($section.Title) НЕ НАЙДЕНА — искали заголовок '##' со словами '$($section.Anchor)' в docs/loop/backlog.md"
        continue
    }
    $end = $b.Count
    for ($i = $start + 1; $i -lt $b.Count; $i++) { if ($b[$i] -match '^##\s') { $end = $i; break } }
    Write-Output $section.Title
    $b[$start..($end - 1)] | Write-Output
}

Write-Output ''
Write-Output '=== Дальше по контракту (§2a): backlog — только секция выбранного эпика (Grep ''^### EPIC-'' → Read offset/limit); дизайн-пакет — только файл нужной фазы в docs/architecture/<пакет>/. Полные файлы состояния целиком не читай. ==='
