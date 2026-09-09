<#
.SYNOPSIS
  DoD gate runner for the HonestML loop: loop-contract (fatal) -> skill-parity ->
  instruction-parity -> config-check (advisory) -> registers -> check-index -> ruff-format ->
  ruff -> lint-imports -> mypy -> pre-commit -> pytest;
  one foreground call, every gate judged.

.DESCRIPTION
  The DoD is one foreground call: every model turn re-reads the whole context, so it must
  cost one tool call, not dozens.
  For the same reason every gate runs even after one of them goes red: the gates
  do not depend on each other, and learning about ruff, mypy and the suite one
  round at a time costs three full contexts to save a few seconds of CPU.

  Call in the FOREGROUND with a generous timeout: the full suite runs 28-40 minutes, so
  allow >= 45 min (2700000 ms); BASH_MAX_TIMEOUT_MS is raised in .CLAUDE/settings.json for
  this. Never run it in the background with a poll loop — leftover background tasks wake
  finished rounds and burn cache rewrites.

  Output: one PASS line per green gate; on failure — head+tail of that gate's
  output. The last line names every gate that failed. Exit code = the first
  failure's exit code, 0 when all green.
  A gate that fails to launch reports "did not run" and exits 127: an empty output
  is never evidence of a pass. An advisory gate reports WARN and the run continues;
  its verdict is named on the final line but does not colour the DoD.

.PARAMETER Scope
  Which tier of the suite to judge. `Full` runs it whole and is what a DoD answers with; `Fast`
  deselects the slow marker for the working contour, and cannot stand in for a full verdict.

.PARAMETER Item
  The item this round is answerable for; scopes the register check's verdict to its rows.

.PARAMETER NoCache
  Measure even when the inputs are unchanged. The cache is keyed on the content of everything
  a verdict can depend on (ops/loop/gate_inputs.py); pass this to judge the tree again anyway.
#>
[CmdletBinding()]
param(
    [ValidateSet('Fast', 'Full')]
    [string]$Scope = 'Full',
    [ValidatePattern('^$|^[A-Za-z0-9._-]+$')]
    [string]$Item = '',
    [switch]$NoCache
)

$ErrorActionPreference = 'Continue'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$env:PYTHONUTF8 = '1'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root

# An advisory gate reports and lets the run continue; a fatal one stops the run where it stands,
# because everything after it would be judged against a contract that no longer holds.
# A cacheable gate is one whose verdict is a function of the input closure alone: rerunning it
# over a tree it already judged costs the round half an hour and learns nothing.
# The registers gate answers for the round's own item: --item scopes its verdict to those rows,
# so a disagreement someone else left behind is reported and returned to its owner instead of
# holding this round's commit. With no item named there is nobody to hold answerable, and the
# gate falls back to advisory.
$registerArgs = @('run', 'python', 'ops/loop/check_registers.py')
if ($Item) { $registerArgs += @('--item', $Item) }

# The suite splits at the slow marker and nowhere else: everything runs in one process against
# synthetic data, so the fast contour is the full one minus its longest cases.
$pytestArgs = @('run', 'pytest', '-q')
if ($Scope -eq 'Fast') { $pytestArgs += @('-m', 'not slow') }
$pytestArgs += @('-o', 'junit_family=xunit1', '--junitxml=ops/loop/logs/pytest-junit.xml')

$gates = @(
    @{ Name = 'loop-contract'; Exe = 'uv'; Fatal = $true
       Args = @('run', 'python', 'ops/loop/implementation_round_guard.py', '--self-test') },
    @{ Name = 'skill-parity'; Exe = 'powershell'; Advisory = $true
       Args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'ops/loop/skill-parity.ps1') },
    @{ Name = 'instruction-parity'; Exe = 'powershell'; Advisory = $true
       Args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'ops/loop/instruction-parity.ps1') },
    @{ Name = 'config-check';  Exe = 'powershell'; Advisory = $true
       Args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'ops/loop/config-check.ps1') },
    @{ Name = 'registers';    Exe = 'uv'; Advisory = (-not $Item); Args = $registerArgs },
    # the decision index is a pure projection of docs/architecture: a round that added or changed
    # an ADR regenerates it with --write, and this is the backstop that says so
    @{ Name = 'check-index';  Exe = 'uv'
       Args = @('run', 'python', 'ops/loop/index_decisions.py', '--check') },
    @{ Name = 'ruff-format';  Exe = 'uv'; Args = @('run', 'ruff', 'format', '--check', 'src', 'tests') },
    @{ Name = 'ruff';         Exe = 'uv'; Args = @('run', 'ruff', 'check', 'src', 'tests') },
    @{ Name = 'lint-imports'; Exe = 'uv'; Args = @('run', 'lint-imports') },
    @{ Name = 'mypy';         Exe = 'uv'; Args = @('run', 'mypy', 'src') },
    @{ Name = 'pre-commit';   Exe = 'uv'; Args = @('run', 'pre-commit', 'run', '--all-files') },
    @{ Name = 'pytest';       Exe = 'uv'; Cacheable = $true
       Clears = @('ops/loop/logs/pytest-junit.xml')
       Artifacts = @('ops/loop/logs/pytest-junit.xml')
       Args = $pytestArgs }
)

# The verdict file is what lets the ledger check a claimed DoD against a run that actually
# happened. It is removed before the gates start: a stale green from an earlier run would
# otherwise vouch for this one, which is the failure the file exists to prevent.
# A gate's own artefacts (`Clears`) are removed the moment before it runs, for the same reason
# and not earlier: a gate served from cache leaves the results of the run its verdict came from,
# which describe the same tree. xunit1 is the junit family carrying each case's source file, and
# that file is what ops/loop/report_test_cost.py builds its replay command from.
$verdictPath = Join-Path $root 'ops/loop/logs/gates-verdict.json'
New-Item -ItemType Directory -Force -Path (Split-Path $verdictPath) | Out-Null
Remove-Item -LiteralPath $verdictPath -ErrorAction SilentlyContinue

# The cache answers one question: has anything a verdict depends on moved since it was recorded.
# A digest that could not be taken means "unknown", and unknown must run — never skip.
# Only the suite is cacheable, and that is the whole of the assumption: its verdict is a function
# of the input closure alone. The static gates cost seconds, so measuring them again is cheaper
# than reasoning about whether a recorded verdict still describes the tree.
$cachePath = Join-Path $root 'ops/loop/logs/gates-cache.json'
$inputsHash = ''
if (-not $NoCache) {
    $global:LASTEXITCODE = $null
    $digest = (& uv run python ops/loop/gate_inputs.py 2>&1 | Select-Object -Last 1)
    if ($LASTEXITCODE -eq 0) { $inputsHash = "$digest".Trim() }
    else { Write-Output "=== cache: inputs could not be digested ($digest) — measuring everything" }
}
$cache = @{}
if ($inputsHash -and (Test-Path -LiteralPath $cachePath)) {
    try {
        (Get-Content -Raw -LiteralPath $cachePath | ConvertFrom-Json).PSObject.Properties |
            ForEach-Object { $cache[$_.Name] = $_.Value }
    } catch {
        Write-Output "=== cache: unreadable, measuring everything ($($_.Exception.Message))"
    }
}

function Save-GatesCache {
    try {
        [System.IO.File]::WriteAllText($cachePath, ($cache | ConvertTo-Json -Depth 4), $utf8)
    } catch {
        Write-Output "=== cache: could not be written ($($_.Exception.Message))"
    }
}

function Write-GatesVerdict([string]$verdict, [string]$gate, [double]$seconds) {
    try {
        $record = [ordered]@{
            ts      = (Get-Date).ToString('o')
            verdict = $verdict
            scope   = $Scope
            gate    = $gate
            seconds = [math]::Round($seconds, 1)
        }
        [System.IO.File]::WriteAllText($verdictPath, ($record | ConvertTo-Json -Compress), $utf8)
    } catch {
        Write-Output "=== verdict: could not be written ($($_.Exception.Message))"
    }
}

function Get-ArtifactRecords([object[]]$relativePaths) {
    $records = @()
    foreach ($relativePath in $relativePaths) {
        $path = Join-Path $root $relativePath
        try {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return @() }
            if ((Get-Item -LiteralPath $path -ErrorAction Stop).Length -eq 0) { return @() }
            $sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash
        } catch {
            return @()
        }
        if ([string]::IsNullOrWhiteSpace($sha256)) { return @() }
        $records += [ordered]@{
            path = $relativePath
            sha256 = $sha256.ToLowerInvariant()
        }
    }
    return $records
}

function Test-CachedArtifacts($artifactRecords, [object[]]$expectedPaths) {
    if ($null -eq $artifactRecords) { return $false }
    $records = @($artifactRecords)
    if ($records.Count -ne $expectedPaths.Count) { return $false }
    $seenPaths = @{}
    foreach ($artifact in $records) {
        if ($null -eq $artifact -or -not $artifact.path -or -not $artifact.sha256) { return $false }
        $relativePath = [string]$artifact.path
        if ($expectedPaths -notcontains $relativePath -or $seenPaths.ContainsKey($relativePath)) {
            return $false
        }
        $seenPaths[$relativePath] = $true
        $path = Join-Path $root $relativePath
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
        if ((Get-Item -LiteralPath $path).Length -eq 0) { return $false }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$artifact.sha256).ToLowerInvariant()) { return $false }
    }
    return $seenPaths.Count -eq $expectedPaths.Count
}

$total = [System.Diagnostics.Stopwatch]::StartNew()
$advisoryWarnings = @()
$failures = @()
$served = @()
foreach ($g in $gates) {
    # keyed by scope: a fast verdict describes a smaller suite and must never serve a full run
    $key = "$($g.Name)|$Scope"
    $entry = if ($g.Cacheable -and $inputsHash -and $cache.ContainsKey($key)) { $cache[$key] } else { $null }
    $cacheHit = $entry -and $entry.hash -eq $inputsHash
    if ($cacheHit -and $g.Artifacts) {
        $cacheHit = Test-CachedArtifacts -artifactRecords $entry.artifacts -expectedPaths @($g.Artifacts)
        if (-not $cacheHit) { Write-Output "=== cache: $($g.Name) artifacts missing or changed — measuring again" }
    }
    if ($cacheHit) {
        # Same inputs, so the recorded verdict still describes this tree — including a red one:
        # rerunning a suite that failed on bytes nobody changed buys the round nothing.
        $code = [int]$entry.code
        $out = @($entry.tail)
        $secs = [double]$entry.seconds
        $mark = "cached, measured in ${secs}s"
        $served += $g.Name
    } else {
        if ($g.Clears) {
            foreach ($relativePath in @($g.Clears)) {
                Remove-Item -LiteralPath (Join-Path $root $relativePath) -ErrorAction SilentlyContinue
            }
        }
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        # A command that never launched leaves the PREVIOUS gate's code in $LASTEXITCODE, so a
        # green predecessor would paint this gate PASS over an empty output. $null is the only
        # honest reading of "did not run", and it must fail the DoD like any other red.
        $global:LASTEXITCODE = $null
        $out = @(& $g.Exe @($g.Args) 2>&1 | ForEach-Object { "$_" })
        $code = $LASTEXITCODE
        $sw.Stop()
        $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        $mark = "${secs}s"
        $artifacts = @()
        if ($null -ne $code -and $code -eq 0 -and $g.Artifacts) {
            $artifacts = @(Get-ArtifactRecords @($g.Artifacts))
            if ($artifacts.Count -ne @($g.Artifacts).Count) {
                $out += "expected artifacts were missing or empty after $($g.Name)"
                $code = 127
            }
        }
        if ($g.Cacheable -and $inputsHash -and $null -ne $code) {
            $cache[$key] = [ordered]@{
                hash    = $inputsHash
                code    = $code
                seconds = $secs
                tail    = @($out | Where-Object { $_.Trim() } | Select-Object -Last 20)
                artifacts = $artifacts
            }
            Save-GatesCache
        }
    }
    if ($null -ne $code -and $code -eq 0) {
        $last = ($out | Where-Object { $_.Trim() } | Select-Object -Last 1)
        Write-Output "=== GATE $($g.Name): PASS ($mark) | $last"
        # a green gate can still have something to say: -Item scopes the register verdict to the
        # round's own rows, and someone else's disagreement comes back as WARN for its note
        $out | Where-Object { $_ -match '^WARN ' } | Write-Output
    } elseif ($g.Advisory) {
        $advisoryWarnings += $g.Name
        Write-Output "=== GATE $($g.Name): WARN exit=$code ($mark) — advisory, run continues"
        $out | Select-Object -First 20 | Write-Output
    } else {
        if ($null -eq $code) {
            Write-Output "=== GATE $($g.Name): FAIL — did not run ($($g.Exe) not launched, $mark)"
        } else {
            Write-Output "=== GATE $($g.Name): FAIL exit=$code ($mark)"
        }
        if ($out.Count -le 60) {
            $out | Write-Output
        } else {
            $out | Select-Object -First 20 | Write-Output
            Write-Output "... ($($out.Count - 60) lines omitted) ..."
            $out | Select-Object -Last 40 | Write-Output
        }
        # `exit $null` would leave the shell reporting 0 — a gate that never ran must exit red.
        $failures += @{ Name = $g.Name; Code = if ($null -eq $code) { 127 } else { $code } }
        if ($g.Fatal) {
            Write-Output "=== DoD: RED ($($g.Name), fatal) after $([math]::Round($total.Elapsed.TotalSeconds, 1))s"
            Write-GatesVerdict 'red' $g.Name $total.Elapsed.TotalSeconds
            exit $failures[0].Code
        }
    }
}
$elapsed = [math]::Round($total.Elapsed.TotalSeconds, 1)
$note = "[$Scope]"
if ($advisoryWarnings.Count -gt 0) { $note += " | advisory WARN: $($advisoryWarnings -join ', ')" }
# A cached verdict is evidence about this tree, but it is not a measurement this call made, and
# the DoD line must not read as though it were.
if ($served.Count -gt 0) { $note += " | from cache: $($served -join ', ')" }
if ($failures.Count -gt 0) {
    $named = ($failures | ForEach-Object { $_.Name }) -join ', '
    Write-Output "=== DoD: RED ($named) after ${elapsed}s $note"
    Write-GatesVerdict 'red' $named $total.Elapsed.TotalSeconds
    exit $failures[0].Code
}
$judged = ($gates | Where-Object { -not $_.Advisory } | ForEach-Object { $_.Name }) -join ', '
Write-Output "=== DoD: GREEN ($judged) in ${elapsed}s $note"
Write-GatesVerdict 'green' '' $total.Elapsed.TotalSeconds
exit 0
