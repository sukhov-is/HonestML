<#
.SYNOPSIS
  Verifies that the two instruction files differ only in their provider layer.

.DESCRIPTION
  One contract is carried twice: AGENTS.md drives Codex, .CLAUDE/CLAUDE.md drives Claude Code.
  Every rule must hold in both, so only the lines naming a runtime's own tools may differ.
  Divergence beyond that means the two drivers run different rules, which surfaces as an
  unexplained behaviour gap many rounds later, with no diff anyone was watching.

  Both files are diffed with zero context. Every hunk must name at least one provider token
  (Agent/spawn_agent, run_in_background, context7/current-documentation, ...); a hunk without one
  is substantive drift. The structural tags themselves must match as an ordered list: a block
  present in one file and absent in the other is drift even when its body names a provider tool.

  Exit code 0 when the files agree, 1 on drift.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$codexFile = Join-Path $root 'AGENTS.md'
$claudeFile = Join-Path $root '.CLAUDE\CLAUDE.md'

$providerTokens = 'spawn_agent|wait_agent|followup_task|fork_turns|update_plan|apply_patch|' +
    'shell_command|SendMessage|TodoWrite|NotebookEdit|run_in_background|general-purpose|' +
    'context7|Explore|Agent|Bash|Write|Edit|Read|Grep|Glob|rg |bash|shell|' +
    'subagent mechanism|current-documentation'

function Get-StructuralTags([string]$path) {
    @(Get-Content -LiteralPath $path -Encoding UTF8 |
        Where-Object { $_ -match '^<[a-zA-Z_]+>\s*$' } |
        ForEach-Object { $_.Trim() })
}

$failures = @()
foreach ($file in @($codexFile, $claudeFile)) {
    if (-not (Test-Path -LiteralPath $file)) { $failures += "missing: $file" }
}

if ($failures.Count -eq 0) {
    $codexTags = Get-StructuralTags $codexFile
    $claudeTags = Get-StructuralTags $claudeFile
    if (Compare-Object $codexTags $claudeTags -SyncWindow 0) {
        $failures += "structural tags differ: AGENTS.md [$($codexTags -join ', ')] vs CLAUDE.md [$($claudeTags -join ', ')]"
    }

    $diff = @(& git -C $root diff --no-index --unified=0 --ignore-cr-at-eol $codexFile $claudeFile)
    $hunk = ''
    $named = $false
    foreach ($line in $diff) {
        if ($line -like '@@*') {
            if ($hunk -and -not $named) { $failures += "drifting hunk  $hunk" }
            $hunk = $line
            $named = $false
            continue
        }
        if (-not $hunk -or $line -match '^(\+\+\+|---)') { continue }
        if ($line -match '^[+-]' -and $line -cmatch $providerTokens) { $named = $true }
    }
    if ($hunk -and -not $named) { $failures += "drifting hunk  $hunk" }
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Output "FAIL instruction parity: $failure" }
    exit 1
}
Write-Output "PASS instruction parity ($($codexTags.Count) blocks, provider layer only)"
exit 0
