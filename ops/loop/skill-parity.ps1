<#
.SYNOPSIS
  Verifies that the two skill trees differ only in their provider layer.

.DESCRIPTION
  The same workflows are carried twice: .CLAUDE/skills drives Claude Code, .agents/skills
  drives Codex. Their instructions must agree in substance -- only the lines naming a
  runtime's own tools may differ. Divergence beyond that means the two drivers run
  different contracts, which surfaces as an unexplained behaviour gap many rounds later,
  with no diff anyone was watching.

  Both trees are diffed with zero context. Every hunk must name at least one provider
  token on either side (Agent/spawn_agent, TodoWrite/update_plan, Edit/apply_patch, ...);
  a hunk without one is substantive drift. Eval fixtures are out of scope -- they feed
  skill-creator runs, not agents -- and so are skill-creator workspaces.

  Exit code 0 when the trees agree, 1 on drift or on a differing file set.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$codexTree = Join-Path $root '.agents\skills'
$claudeTree = Join-Path $root '.CLAUDE\skills'

$providerTokens = 'spawn_agent|wait_agent|followup_task|fork_turns|update_plan|apply_patch|' +
    'shell_command|SendMessage|TodoWrite|NotebookEdit|run_in_background|general-purpose|' +
    'context7|Explore|Agent|Bash|Write|Edit|Read|Grep|Glob|rg |bash|shell|' +
    # how a provider hands back a long-running cell is its own layer: codex yields and resumes a
    # cell, the Claude harness returns the finished call
    'yield_time_ms|ячейк|cell_id'

function Get-SkillFiles([string]$treeRoot) {
    Get-ChildItem -LiteralPath $treeRoot -Recurse -File |
        ForEach-Object { ($_.FullName.Substring($treeRoot.Length + 1)) -replace '\\', '/' } |
        Where-Object { $_ -notmatch '(^|/)evals/' -and $_ -notmatch '-workspace/' } |
        Sort-Object
}

function Get-DriftingHunks([string]$relativePath) {
    $left = Join-Path $codexTree ($relativePath -replace '/', '\')
    $right = Join-Path $claudeTree ($relativePath -replace '/', '\')
    $diff = @(& git -C $root diff --no-index --unified=0 --ignore-cr-at-eol $left $right)
    $drifting = @()
    $hunk = ''
    $named = $false
    foreach ($line in $diff) {
        if ($line -like '@@*') {
            if ($hunk -and -not $named) { $drifting += "$relativePath  $hunk" }
            $hunk = $line
            $named = $false
            continue
        }
        if (-not $hunk -or $line -match '^(\+\+\+|---)') { continue }
        if ($line -match '^[+-]' -and $line -cmatch $providerTokens) { $named = $true }
    }
    if ($hunk -and -not $named) { $drifting += "$relativePath  $hunk" }
    $drifting
}

$codexFiles = @(Get-SkillFiles $codexTree)
$claudeFiles = @(Get-SkillFiles $claudeTree)
$shared = @($codexFiles | Where-Object { $claudeFiles -contains $_ })
$failures = @()
$failures += @($codexFiles | Where-Object { $claudeFiles -notcontains $_ } |
    ForEach-Object { "missing in .CLAUDE/skills: $_" })
$failures += @($claudeFiles | Where-Object { $codexFiles -notcontains $_ } |
    ForEach-Object { "missing in .agents/skills: $_" })
$failures += @($shared | ForEach-Object { Get-DriftingHunks $_ })

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Output "FAIL skill parity: $failure" }
    exit 1
}
Write-Output "PASS skill parity ($($shared.Count) files, provider layer only)"
exit 0
