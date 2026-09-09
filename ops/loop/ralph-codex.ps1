<#
.SYNOPSIS
  HonestML Autonomous Loop driver for Codex (one persistent thread per round).

.DESCRIPTION
  Each iteration launches a fresh non-interactive Codex thread and feeds it the shared
  docs/loop/LOOP.md round contract. If a model turn ends before the repository round is
  complete, the driver resumes that same thread within the original wall-clock budget.

  A round succeeds only when the process exits with code 0, emits a terminal
  `turn.completed` JSONL event with no unfinished command, creates the required commit,
  and leaves the repository completion contract satisfied. Unknown failures and timeouts
  halt the loop with work-in-progress preserved for inspection.

  The driver uses codex.cmd on Windows so an npm installation works when PowerShell script
  execution is restricted. Sessions are persisted and each new repository round gets a fresh
  thread; resume is used only for incomplete turns inside that round.

  Stop markers: docs/loop/STOP is the graceful switch - an in-flight round keeps resuming until
  it completes and commits, then the loop halts before the next round. docs/loop/STOP-NOW is the
  kill switch - the in-flight round process tree is terminated immediately, WIP kept.

.PARAMETER MaxIterations
  Round cap per launch. 0 = unlimited.
.PARAMETER BypassPermissions
  Required for autonomous edit/run/commit access. Passes the Codex approvals/sandbox bypass
  and hook-trust bypass flags. Use only in an isolated, clean checkout.
.PARAMETER Prompt
  Shared round-contract file (default docs\loop\LOOP.md).
.PARAMETER Model
  Pin every round to one model. Omit to retain separate design/implementation selection.
.PARAMETER Effort
  Reasoning effort in single-model mode.
.PARAMETER DesignModel
  Model for design rounds. Default = gpt-6-astra; empty = active Codex default.
.PARAMETER ImplModel
  Model for implementation rounds. Default = gpt-6-astra; empty = active Codex default.
.PARAMETER DesignEffort
  Reasoning effort for design rounds.
.PARAMETER ImplEffort
  Reasoning effort for implementation rounds.
.PARAMETER MaxRoundMinutes
  Wall-clock limit for the complete round across its initial process and any resumes. Timeout
  exits with synthetic code 124 and keeps WIP.
.PARAMETER MaxTransientRetries
  Consecutive transport-level failures (dropped connection, 5xx) that relaunch the same round in a
  fresh thread before the loop halts (default 3). The round's WIP is kept and resumed from files.
.PARAMETER Force
  Allow a dirty working tree at startup. The round may include those changes in its commit.
.PARAMETER CodexCommand
  Optional explicit path/name for the Codex CLI. Windows defaults to codex.cmd.
.PARAMETER Preflight
  Check the CLI, authentication, prompt, repository and STOP state without starting a round.
.PARAMETER SelfTest
  Run parser/argument assertions without invoking Codex.

.EXAMPLE
  powershell -NoProfile -File ops/loop/ralph-codex.ps1 -SelfTest
.EXAMPLE
  powershell -NoProfile -File ops/loop/ralph-codex.ps1 -Preflight
.EXAMPLE
  powershell -NoProfile -File ops/loop/ralph-codex.ps1 -BypassPermissions -MaxIterations 1
#>
[CmdletBinding()]
param(
    [ValidateRange(0, [int]::MaxValue)]
    [int]$MaxIterations = 0,
    [switch]$BypassPermissions,
    [string]$Prompt = 'docs\loop\LOOP.md',
    [AllowEmptyString()]
    [ValidatePattern('^$|^[A-Za-z0-9][A-Za-z0-9._:/-]*$')]
    [string]$Model = 'gpt-6-astra',
    [ValidateSet('minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra')]
    [string]$Effort = 'low',
    [AllowEmptyString()]
    [ValidatePattern('^$|^[A-Za-z0-9][A-Za-z0-9._:/-]*$')]
    [string]$DesignModel = 'gpt-6-astra',
    [AllowEmptyString()]
    [ValidatePattern('^$|^[A-Za-z0-9][A-Za-z0-9._:/-]*$')]
    [string]$ImplModel = 'gpt-6-astra',
    [ValidateSet('minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra')]
    [string]$DesignEffort = 'medium',
    [ValidateSet('minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra')]
    [string]$ImplEffort = 'medium',
    [ValidateRange(1, 1440)]
    [int]$MaxRoundMinutes = 400,
    [switch]$Force,
    [string]$CodexCommand = '',
    [ValidateRange(0, 20)]
    [int]$MaxTransientRetries = 3,
    [switch]$Preflight,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'

$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$loopPrompt = Join-Path $root $Prompt
$stopFile = Join-Path $root 'docs\loop\STOP'
$stopNowFile = Join-Path $root 'docs\loop\STOP-NOW'
$ledger = Join-Path $root 'docs\loop\state.md'
$logDir = Join-Path $root 'ops\loop\logs'

# the two trees a round's code lands in; everything else a commit touches (docs, ops, the ledger)
# is not code volume
$script:CodeVolumeAreas = @('src', 'tests')

function Resolve-CodexCli([string]$preferred) {
    if ($preferred) {
        if (Test-Path -LiteralPath $preferred -PathType Leaf) {
            return (Resolve-Path -LiteralPath $preferred).Path
        }
        $explicit = Get-Command $preferred -CommandType Application -ErrorAction SilentlyContinue
        if ($explicit) { return $explicit.Source }
        throw "Codex CLI not found: $preferred"
    }

    $names = if ($env:OS -eq 'Windows_NT') {
        @('codex.cmd', 'codex.exe')
    } else {
        @('codex')
    }
    foreach ($name in $names) {
        $found = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
    }
    throw 'Codex CLI not found. Install it and ensure codex.cmd/codex is on PATH.'
}

function Invoke-CodexProbe([string]$command, [string[]]$arguments) {
    $ErrorActionPreference = 'Continue'
    $output = & $command @arguments 2>&1
    $exitCode = $LASTEXITCODE
    $text = (@($output) | ForEach-Object { $_.ToString() }) -join "`n"
    return [pscustomobject]@{ ExitCode = $exitCode; Text = $text.Trim() }
}

# `git diff --numstat` rows ("added<TAB>removed<TAB>path", '-' where the file is binary) and
# `git grep -Ic '^'` rows ("path:lines", one per tracked text file) folded into the volume line of a
# round note: what the round added and removed under each area, against the size that area now has.
# Empty while src/ and tests/ stayed untouched, so a design round carries no volume line.
function Format-CodeVolumeNote([string[]]$numstatRows, [string[]]$lineCountRows) {
    $added = @{}; $removed = @{}; $total = @{}
    foreach ($area in $script:CodeVolumeAreas) { $added[$area] = 0; $removed[$area] = 0; $total[$area] = 0 }
    foreach ($row in $numstatRows) {
        $cells = ([string]$row) -split "`t"
        if ($cells.Count -lt 3) { continue }
        $area = ($cells[2] -split '/')[0]
        if (-not $added.ContainsKey($area)) { continue }
        if ($cells[0] -ne '-') { $added[$area] += [int]$cells[0] }
        if ($cells[1] -ne '-') { $removed[$area] += [int]$cells[1] }
    }
    if (-not ($script:CodeVolumeAreas | Where-Object { $added[$_] -or $removed[$_] })) { return '' }
    foreach ($row in $lineCountRows) {
        $text = [string]$row
        $split = $text.LastIndexOf(':')
        if ($split -lt 1) { continue }
        $area = ($text.Substring(0, $split) -split '/')[0]
        if ($total.ContainsKey($area)) { $total[$area] += [int]$text.Substring($split + 1) }
    }
    $parts = foreach ($area in $script:CodeVolumeAreas) {
        "$area +$($added[$area])/-$($removed[$area]) of $($total[$area])"
    }
    return 'lines: ' + ($parts -join ', ')
}

# measured, never taken from what the round said about itself. A note may not halt a round, so an
# unreadable git answer degrades to no volume line.
function Get-CodeVolumeNote([string]$sinceSha) {
    try {
        return Format-CodeVolumeNote @(git diff --numstat $sinceSha HEAD -- src tests) `
                                     @(git grep -Ic '^' -- src tests)
    } catch { return '' }
}

# Best-effort note to whoever is not watching the console. The tag names the project as well as
# the driver: one Telegram chat carries several repositories, and "[loop codex]" alone would not
# say which one is reporting.
function Send-LoopNote([string]$text) {
    try { & (Join-Path $PSScriptRoot 'notify.ps1') -Message "[honestml codex] $text" } catch {}
}

# A crash leaves nobody watching the console; say so before the process dies.
trap { Send-LoopNote "driver crashed: $($_.Exception.Message)"; break }

# graceful STOP only halts at a round boundary, so an in-flight round still reaches its
# commit; STOP-NOW is the emergency switch and applies at any moment
function Test-LoopStopRequested(
    [bool]$betweenRounds,
    [bool]$softStopPresent,
    [bool]$emergencyStopPresent
) {
    return $emergencyStopPresent -or ($betweenRounds -and $softStopPresent)
}

# rotate.ps1 stays silent when the table is short enough: only a reported rotation may be
# committed, so a round's own ledger edits never land under the rotation commit
function Test-RotationPerformed([string[]]$messages) {
    return $messages.Count -gt 0
}

function ConvertTo-NativeArgument([string]$value) {
    if ($null -eq $value -or $value.Length -eq 0) { return '""' }
    if ($value -match '^[A-Za-z0-9_./:\\=-]+$') { return $value }

    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $slashes = 0
    foreach ($char in $value.ToCharArray()) {
        if ($char -eq '\') {
            $slashes++
            continue
        }
        if ($char -eq '"') {
            [void]$builder.Append(('\' * (($slashes * 2) + 1)))
            [void]$builder.Append('"')
            $slashes = 0
            continue
        }
        if ($slashes -gt 0) {
            [void]$builder.Append(('\' * $slashes))
            $slashes = 0
        }
        [void]$builder.Append($char)
    }
    if ($slashes -gt 0) { [void]$builder.Append(('\' * ($slashes * 2))) }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function New-CodexBaseArguments([string]$roundModel, [string]$roundEffort) {
    $arguments = [System.Collections.Generic.List[string]]::new()
    $arguments.Add('--dangerously-bypass-approvals-and-sandbox')
    $arguments.Add('--dangerously-bypass-hook-trust')
    $arguments.Add('--enable')
    $arguments.Add('multi_agent')
    $arguments.Add('-C')
    $arguments.Add($root)
    if ($roundModel) {
        $arguments.Add('-m')
        $arguments.Add($roundModel)
    }
    $arguments.Add('-c')
    $arguments.Add("model_reasoning_effort=$roundEffort")
    return ,$arguments
}

function New-CodexArguments([string]$roundModel, [string]$roundEffort) {
    $arguments = New-CodexBaseArguments $roundModel $roundEffort
    $arguments.Add('exec')
    $arguments.Add('--strict-config')
    $arguments.Add('--color')
    $arguments.Add('never')
    $arguments.Add('--json')
    $arguments.Add('-')
    return $arguments.ToArray()
}

function New-CodexResumeArguments(
    [string]$roundModel,
    [string]$roundEffort,
    [string]$threadId
) {
    $arguments = New-CodexBaseArguments $roundModel $roundEffort
    $arguments.Add('exec')
    $arguments.Add('resume')
    $arguments.Add('--strict-config')
    $arguments.Add('--json')
    $arguments.Add($threadId)
    $arguments.Add('-')
    return $arguments.ToArray()
}

function Limit-Text([string]$text, [int]$limit = 240) {
    if (-not $text -or $text.Length -le $limit) { return $text }
    return $text.Substring(0, $limit - 3) + '...'
}

function Get-CodexErrorText([object]$event) {
    $value = if ($null -ne $event.error) { $event.error } else { $event.message }
    if ($null -eq $value) { return '' }
    if ($value -is [string]) { return $value }
    if ($null -ne $value.message) { return [string]$value.message }
    try { return ($value | ConvertTo-Json -Compress -Depth 8) } catch { return [string]$value }
}

function Format-CodexStreamEvent([string]$line) {
    if (-not $line) { return $null }
    try { $event = $line | ConvertFrom-Json -ErrorAction Stop } catch { return $null }
    switch ($event.type) {
        'thread.started' { return "[thread: $($event.thread_id)]" }
        'item.completed' {
            if ($null -eq $event.item) { return $null }
            switch ($event.item.type) {
                'agent_message' { return [string]$event.item.text }
                'command_execution' {
                    $command = if ($event.item.command -is [array]) {
                        $event.item.command -join ' '
                    } else {
                        [string]$event.item.command
                    }
                    return "  -> $(Limit-Text $command)"
                }
                'file_change' { return '  -> file change' }
                'mcp_tool_call' { return "  -> MCP $($event.item.server)/$($event.item.tool)" }
                default { return $null }
            }
        }
        'turn.completed' { return '[turn completed]' }
        'turn.failed' { return "[turn failed: $(Limit-Text (Get-CodexErrorText $event))]" }
        'error' { return "[error: $(Limit-Text (Get-CodexErrorText $event))]" }
        default { return $null }
    }
}

function Get-CodexRunSummaryFromLines([string[]]$lines) {
    $threadId = $null
    $terminal = 'missing'
    $usage = $null
    $errors = [System.Collections.Generic.List[string]]::new()
    $inFlightCommands = [ordered]@{}
    foreach ($line in $lines) {
        if (-not $line) { continue }
        try { $event = $line | ConvertFrom-Json -ErrorAction Stop } catch { continue }
        switch ($event.type) {
            'thread.started' { $threadId = [string]$event.thread_id }
            'item.started' {
                if ($null -ne $event.item -and $event.item.type -eq 'command_execution') {
                    $inFlightCommands[[string]$event.item.id] = [string]$event.item.command
                }
            }
            'item.completed' {
                if ($null -ne $event.item -and $event.item.type -eq 'command_execution') {
                    $inFlightCommands.Remove([string]$event.item.id)
                }
            }
            'turn.completed' {
                $terminal = 'completed'
                $usage = $event.usage
            }
            'turn.failed' {
                $terminal = 'failed'
                $message = Get-CodexErrorText $event
                if ($message) { $errors.Add($message) }
            }
            'error' {
                $message = Get-CodexErrorText $event
                if ($message) { $errors.Add($message) }
            }
        }
    }
    return [pscustomobject]@{
        ThreadId = $threadId
        Terminal = $terminal
        Usage = $usage
        Errors = @($errors)
        InFlightCommandIds = @($inFlightCommands.Keys)
    }
}

function Read-CodexRunSummary([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) {
        return Get-CodexRunSummaryFromLines @()
    }
    return Get-CodexRunSummaryFromLines ([System.IO.File]::ReadAllLines($path, $utf8))
}

function Test-CodexRateLimited([object]$summary) {
    $text = ($summary.Errors -join "`n")
    return $text -match '(?i)(rate[ _-]?limit|usage[ _-]?limit|too many requests|\b429\b)'
}

# transport-level death of the round process (dropped connection, 5xx, overloaded upstream) leaves the
# round's WIP intact on disk, so the same round is worth another process. A process that dies without
# emitting a terminal event says nothing in JSONL - stderr carries the reason, so both are inspected.
function Get-RoundFailureText([object]$summary, [string]$errPath) {
    $text = (@($summary.Errors) -join "`n")
    if ($errPath -and (Test-Path -LiteralPath $errPath)) {
        $lines = @([System.IO.File]::ReadAllLines($errPath, $utf8))
        if ($lines.Count -gt 30) { $lines = $lines[-30..-1] }
        $text = $text + "`n" + ($lines -join "`n")
    }
    return $text
}

function Test-CodexTransientFailure([string]$text) {
    if (-not $text) { return $false }
    return $text -match "(?i)(connection closed|connection error|connection reset|socket hang up|ECONNRESET|ETIMEDOUT|EPIPE|fetch failed|network error|stream disconnected|overloaded|internal server error|50[234]\s|Bad Gateway|Service Unavailable|Gateway Time-?out)"
}

# one column of the ledger rounds table, in file order:
# 0 round | 1 item | 2 type | 3 gate | 4 DoD | 5 dtests | 6 commit | 7 status | 8 note
function Get-RoundColumnFromLines([string[]]$lines, [int]$columnIndex) {
    $values = @()
    foreach ($line in $lines) {
        if ($line -match '^\|\s*\d+\s*\|') {
            $cells = ($line.Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
            if ($cells.Count -gt $columnIndex) { $values += $cells[$columnIndex] }
        }
    }
    return $values
}

function Get-RoundStatuses {
    if (-not (Test-Path $ledger)) { return @() }
    return Get-RoundColumnFromLines ([System.IO.File]::ReadAllLines($ledger, $utf8)) 7
}

# Whether the contract this run drives keeps the rounds table of docs/loop/state.md. Only the LOOP
# contract does; another one keeps its own register (the debug contract records a finding row in
# docs/audit/.../05-fixes.md and never touches the ledger), so the verdicts read from that table
# describe someone else's work and say nothing about the round that just ran.
function Test-KeepsLoopLedger([string]$contractPath, [string]$loopContractPath) {
    return [System.IO.Path]::GetFullPath($contractPath) -ieq
        [System.IO.Path]::GetFullPath($loopContractPath)
}

# What to call a round the ledger does not name. The contract file names it, so a third contract
# labels itself without the driver learning about it.
function Get-ContractKind([string]$contractPath) {
    return [System.IO.Path]::GetFileNameWithoutExtension($contractPath).ToLowerInvariant()
}

function Get-RoundTypes {
    if (-not (Test-Path $ledger)) { return @() }
    return Get-RoundColumnFromLines ([System.IO.File]::ReadAllLines($ledger, $utf8)) 2
}

# Phases of the items table. An item row is the one whose first cell is an id: round rows lead
# with a number, separators with dashes, and the only other id-shaped first cell in the ledger is
# the rounds header's own word 'round'.
function Get-ItemPhasesFromLines([string[]]$lines) {
    $phases = @()
    foreach ($line in $lines) {
        if ($line -notmatch '^\|') { continue }
        $cells = ($line.Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
        if ($cells.Count -ge 2 -and $cells[0] -match '^[A-Za-z][A-Za-z0-9._-]*$' -and $cells[0] -ne 'round') {
            $phases += $cells[1]
        }
    }
    return $phases
}

function Get-ItemPhases {
    if (-not (Test-Path $ledger)) { return @() }
    return Get-ItemPhasesFromLines ([System.IO.File]::ReadAllLines($ledger, $utf8))
}

# What the loop still owes work to, not what it did last: a package awaiting code (designed)
# or a round stopped by a declared obstacle (needs-fix) is an impl round waiting to happen.
function Test-ImplKind([string[]]$phases) {
    return @($phases | Where-Object { $_ -eq 'designed' -or $_ -eq 'needs-fix' }).Count -gt 0
}

function Get-NextRoundKind {
    if (Test-ImplKind (Get-ItemPhases)) { return 'impl' }
    return 'design'
}

function New-CodexRoundPrompt([string]$contract) {
    $adapter = @'
<codex_provider_adapter>
- When the contract says to invoke skill X with a Skill tool, read and apply the available repository skill X. Codex has no separate Skill tool.
- Grep, Read, Bash and other provider-specific tool names mean the available Codex equivalent.
- References to Opus mean the implementation model already selected by the driver. Do not switch models or combine design and implementation rounds.
- Do not start persistent or background processes. Wait for every subagent and command before the final commit and exit.
- Set timeout_ms=2700000 on every foreground test or gate command that can exceed two minutes. A tool timeout is not a test failure: terminate its command tree and rerun it with the required timeout before deciding the round outcome.
</codex_provider_adapter>
'@
    return $adapter + "`r`n" + $contract
}

function Test-CodexRoundWorkloadActive([string]$repoRoot) {
    if ($env:OS -ne 'Windows_NT') { return $false }
    $escapedRoot = [regex]::Escape($repoRoot.TrimEnd('\', '/'))
    try {
        foreach ($process in @(Get-CimInstance Win32_Process -ErrorAction Stop)) {
            if ($process.ProcessId -eq $PID -or [string]::IsNullOrWhiteSpace($process.CommandLine)) {
                continue
            }
            if ($process.CommandLine -match $escapedRoot -and
                $process.CommandLine -match '(?i)(gates\.ps1|uv(?:\.exe)?\s+run|pytest|ruff|mypy|lint-imports)') {
                return $true
            }
        }
        return $false
    } catch {
        return $true
    }
}

function Get-RoundCompletionViolations(
    [string]$preHead,
    [string]$postHead,
    [string]$roundKind,
    [string[]]$statuses,
    [bool]$workingTreeDirty,
    [bool]$roundWorkloadActive = $false,
    [object]$summary
) {
    $violations = [System.Collections.Generic.List[string]]::new()
    $commitCreated = $postHead -ne $preHead
    $terminalStatus = $roundKind -ne 'impl' -or
        ($statuses.Count -gt 0 -and $statuses[-1] -in @('done', 'blocked'))
    $roundContractComplete = $commitCreated -and -not $workingTreeDirty -and $terminalStatus
    if (-not $roundContractComplete -or $roundWorkloadActive) {
        foreach ($commandId in @($summary.InFlightCommandIds)) {
            $violations.Add("command_execution $commandId has no completed event")
        }
    }
    if (-not $commitCreated) {
        $violations.Add('the round has not created a commit')
    }
    if ($workingTreeDirty) {
        $violations.Add('the working tree is dirty')
    }
    if ($roundKind -eq 'impl') {
        if ($statuses.Count -eq 0) {
            $violations.Add('the loop ledger has no terminal implementation status')
        } elseif ($statuses[-1] -notin @('done', 'blocked', 'needs-fix')) {
            $violations.Add("the implementation status is '$($statuses[-1])', expected done, blocked or needs-fix")
        }
    }
    return $violations.ToArray()
}

function New-CodexContinuationPrompt([string[]]$violations, [int]$remainingSeconds) {
    $items = ($violations | ForEach-Object { "- $_" }) -join "`n"
    return @"
<codex_round_continuation>
The previous Codex turn ended, but the autonomous round is not complete:
$items

Continue this same round in the same thread. Preserve the current WIP and do not start another
backlog item. A command without a completed event outlived the prior turn, so its result was not
captured; verify that no child process remains and rerun the check when its result is required.
Finish all review/fix/verification work, update the round artifacts, create the required commit,
and leave a clean working tree before stopping. Approximately $remainingSeconds seconds remain in
the original round wall-clock budget.
</codex_round_continuation>
"@
}

function Stop-ProcessTree([System.Diagnostics.Process]$process) {
    try {
        if ($process.HasExited) { return $false }
    } catch {
        return $false
    }

    $treeKillConfirmed = $false
    if ($env:OS -eq 'Windows_NT') {
        $killInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $killInfo.FileName = Join-Path $env:SystemRoot 'System32\taskkill.exe'
        $killInfo.Arguments = "/PID $($process.Id) /T /F"
        $killInfo.UseShellExecute = $false
        $killInfo.CreateNoWindow = $true
        $killInfo.RedirectStandardOutput = $true
        $killInfo.RedirectStandardError = $true
        $killer = [System.Diagnostics.Process]::new()
        $killer.StartInfo = $killInfo
        try {
            if ($killer.Start()) {
                if ($killer.WaitForExit(10000)) {
                    $treeKillConfirmed = $killer.ExitCode -eq 0
                } else {
                    try { $killer.Kill() } catch {}
                    try { [void]$killer.WaitForExit(1000) } catch {}
                }
            }
        } catch {
        } finally {
            $killer.Dispose()
        }
        try {
            if (-not $process.HasExited) { $process.Kill() }
        } catch {
        }
    } else {
        try { $process.Kill() } catch {}
    }
    $rootStopped = $false
    try {
        $rootStopped = $process.WaitForExit(10000)
    } catch {
    }
    if (-not $rootStopped) {
        try { $rootStopped = $process.HasExited } catch { $rootStopped = $false }
    }
    if ($env:OS -eq 'Windows_NT') { return $rootStopped -and $treeKillConfirmed }
    return $rootStopped
}

function Invoke-CodexRoundProcess(
    [string]$command,
    [string[]]$arguments,
    [string]$promptText,
    [string]$outFile,
    [string]$errFile,
    [int]$timeoutSeconds,
    [string]$roundKind,
    [string]$stopMarker = '',
    [int]$terminalGraceSeconds = 30
) {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $command
    $startInfo.Arguments = (($arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
    $startInfo.WorkingDirectory = $root
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = $utf8
    $startInfo.StandardErrorEncoding = $utf8
    $startInfo.EnvironmentVariables['HONESTML_LOOP_ROUND'] = '1'
    $startInfo.EnvironmentVariables['HONESTML_LOOP_KIND'] = $roundKind

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $outWriter = [System.IO.StreamWriter]::new($outFile, $false, $utf8)
    $errWriter = [System.IO.StreamWriter]::new($errFile, $false, $utf8)
    $outWriter.AutoFlush = $true
    $errWriter.AutoFlush = $true
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    $started = $false
    $timedOut = $false
    $stopRequested = $false
    $processTreeStopConfirmed = $true
    $exitCode = 1
    $stdinClosed = $false
    $lingered = $false
    $terminalClock = $null
    $exitDrainClock = $null
    # Streams outlive the agent whenever a child inherits its pipes, so EOF is not a completion
    # signal. Both waits below are what bounds a round: the turn's terminal event ends it, and a
    # dead process gets only a short drain for output still in flight.
    $exitDrainSeconds = 5

    try {
        $started = $process.Start()
        $bytes = $utf8.GetBytes($promptText + "`n")
        $stdoutTask = $process.StandardOutput.ReadLineAsync()
        $stderrTask = $process.StandardError.ReadLineAsync()
        $stdinTask = $process.StandardInput.BaseStream.WriteAsync($bytes, 0, $bytes.Length)
        while ($true) {
            $handled = $false
            if (-not $stdinClosed -and $stdinTask.IsCompleted) {
                if ($stdinTask.IsFaulted) { throw $stdinTask.Exception.GetBaseException() }
                if ($stdinTask.IsCanceled) { throw 'Codex prompt write was canceled.' }
                $process.StandardInput.Close()
                $stdinClosed = $true
                $handled = $true
            }
            if ($null -ne $stdoutTask -and $stdoutTask.IsCompleted) {
                $line = $stdoutTask.Result
                if ($null -eq $line) {
                    $stdoutTask = $null
                } else {
                    $outWriter.WriteLine($line)
                    $pretty = Format-CodexStreamEvent $line
                    if ($pretty) { Write-Host $pretty }
                    if ($null -eq $terminalClock -and
                        (Get-CodexRunSummaryFromLines @($line)).Terminal -ne 'missing') {
                        $terminalClock = [System.Diagnostics.Stopwatch]::StartNew()
                    }
                    $stdoutTask = $process.StandardOutput.ReadLineAsync()
                }
                $handled = $true
            }
            if ($null -ne $stderrTask -and $stderrTask.IsCompleted) {
                $line = $stderrTask.Result
                if ($null -eq $line) {
                    $stderrTask = $null
                } else {
                    $errWriter.WriteLine($line)
                    $stderrTask = $process.StandardError.ReadLineAsync()
                }
                $handled = $true
            }

            if ($stopMarker -and (Test-Path -LiteralPath $stopMarker)) {
                $stopRequested = $true
                if (-not $stdinClosed) {
                    try { $process.StandardInput.Close() } catch {}
                    $stdinClosed = $true
                }
                $processRunning = $false
                try { $processRunning = -not $process.HasExited } catch { $processRunning = $true }
                if ($processRunning) {
                    $processTreeStopConfirmed = Stop-ProcessTree $process
                }
                break
            }
            if (-not $timedOut -and $clock.Elapsed.TotalSeconds -ge $timeoutSeconds) {
                $timedOut = $true
                if (-not $stdinClosed) {
                    try { $process.StandardInput.Close() } catch {}
                    $stdinClosed = $true
                }
                $processTreeStopConfirmed = Stop-ProcessTree $process
                break
            }
            if ($null -ne $terminalClock -and
                $terminalClock.Elapsed.TotalSeconds -ge $terminalGraceSeconds) {
                $lingered = $true
                if (-not $stdinClosed) {
                    try { $process.StandardInput.Close() } catch {}
                    $stdinClosed = $true
                }
                $processRunning = $false
                try { $processRunning = -not $process.HasExited } catch { $processRunning = $true }
                if ($processRunning) {
                    $processTreeStopConfirmed = Stop-ProcessTree $process
                }
                break
            }
            if ($process.HasExited) {
                if ($null -eq $stdoutTask -and $null -eq $stderrTask) { break }
                if ($null -eq $exitDrainClock) {
                    $exitDrainClock = [System.Diagnostics.Stopwatch]::StartNew()
                } elseif ($exitDrainClock.Elapsed.TotalSeconds -ge $exitDrainSeconds) {
                    break
                }
            }
            if (-not $handled) { Start-Sleep -Milliseconds 25 }
        }
        if ($timedOut) {
            $exitCode = 124
        } elseif ($stopRequested -or $lingered) {
            $exitCode = 0
        } else {
            [void]$process.WaitForExit(10000)
            $exitCode = $process.ExitCode
        }
    } finally {
        $clock.Stop()
        if ($started) {
            if (-not $stdinClosed) {
                try { $process.StandardInput.Close() } catch {}
            }
            $processRunning = $false
            try { $processRunning = -not $process.HasExited } catch { $processRunning = $true }
            if ($processRunning -and -not (Stop-ProcessTree $process)) {
                $processTreeStopConfirmed = $false
            }
        }
        $outWriter.Close()
        $errWriter.Close()
        $process.Dispose()
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        TimedOut = $timedOut
        StopRequested = $stopRequested
        ProcessTreeStopConfirmed = $processTreeStopConfirmed
        DurationSeconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
    }
}

function Get-UsageValue([object]$usage, [string]$name) {
    if ($null -eq $usage) { return $null }
    $property = $usage.PSObject.Properties[$name]
    if ($null -eq $property -or $null -eq $property.Value) { return $null }
    return [long]$property.Value
}

function Write-RoundMetrics(
    [int]$round,
    [string]$kind,
    [string]$roundModel,
    [string]$roundEffort,
    [int]$segments,
    [object]$processResult,
    [object]$summary,
    [string]$headSha
) {
    try {
        $record = [ordered]@{
            ts = (Get-Date).ToString('s')
            provider = 'codex'
            round = $round
            kind = $kind
            model = if ($roundModel) { $roundModel } else { $null }
            effort = $roundEffort
            segments = $segments
            exit = $processResult.ExitCode
            terminal = $summary.Terminal
            timed_out = $processResult.TimedOut
            stop_requested = $processResult.StopRequested
            thread_id = $summary.ThreadId
            duration_s = $processResult.DurationSeconds
            input_tokens = Get-UsageValue $summary.Usage 'input_tokens'
            cached_input_tokens = Get-UsageValue $summary.Usage 'cached_input_tokens'
            output_tokens = Get-UsageValue $summary.Usage 'output_tokens'
            reasoning_output_tokens = Get-UsageValue $summary.Usage 'reasoning_output_tokens'
            head = $headSha
        }
        [System.IO.File]::AppendAllText(
            (Join-Path $PSScriptRoot 'metrics.jsonl'),
            ($record | ConvertTo-Json -Compress) + "`n",
            $utf8
        )
    } catch {
        Write-Host "metrics: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

if ($SelfTest) {
    $failures = 0
    function Check([string]$name, [bool]$condition) {
        if ($condition) { "PASS $name" } else { $script:failures++; "FAIL $name" }
    }

    $completedLines = @(
        '{"type":"thread.started","thread_id":"thread-1"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
        '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":4,"output_tokens":2,"reasoning_output_tokens":1}}'
    )
    $completed = Get-CodexRunSummaryFromLines $completedLines
    Check 'summary completed' ($completed.Terminal -eq 'completed')
    Check 'summary thread' ($completed.ThreadId -eq 'thread-1')
    Check 'summary usage' ((Get-UsageValue $completed.Usage 'cached_input_tokens') -eq 4)

    $premature = Get-CodexRunSummaryFromLines @(
        '{"type":"thread.started","thread_id":"thread-early"}',
        '{"type":"item.started","item":{"id":"command-1","type":"command_execution","status":"in_progress"}}',
        '{"type":"turn.completed","usage":{"input_tokens":20}}'
    )
    Check 'summary keeps unfinished command at turn completion' (
        $premature.InFlightCommandIds -contains 'command-1'
    )

    $failed = Get-CodexRunSummaryFromLines @(
        '{broken',
        '{"type":"unknown"}',
        '{"type":"turn.failed","error":{"message":"rate limit exceeded"}}'
    )
    Check 'summary failed' ($failed.Terminal -eq 'failed')
    Check 'rate-limit hint' (Test-CodexRateLimited $failed)
    Check 'rate-limit no false-positive' (-not (Test-CodexRateLimited $completed))
    Check 'transient: dropped connection' (Test-CodexTransientFailure 'stream disconnected before completion: Connection closed')
    Check 'transient: upstream 503' (Test-CodexTransientFailure 'unexpected status 503 Service Unavailable')
    Check 'content error is not transient' (-not (Test-CodexTransientFailure 'invalid request: unsupported model'))
    Check 'rate limit is not transient' (-not (Test-CodexTransientFailure 'You have hit your usage limit'))
    Check 'failure text merges summary errors and stderr' (
        (Get-RoundFailureText ([pscustomobject]@{ Errors = @('boom') }) '') -match 'boom'
    )

    Check 'format agent message' ((Format-CodexStreamEvent $completedLines[1]) -eq 'done')
    Check 'format malformed' ($null -eq (Format-CodexStreamEvent '{broken'))
    # the kind follows work still owed, not the order rows happen to sit in
    Check 'designed anywhere -> impl' (Test-ImplKind @('done', 'designed', 'done'))
    Check 'needs-fix anywhere -> impl' (Test-ImplKind @('done', 'needs-fix'))
    Check 'all settled -> design' (-not (Test-ImplKind @('done', 'blocked', 'done')))
    Check 'no items -> design' (-not (Test-ImplKind @()))
    # the rounds header (| round | item | ...) has an id-shaped first cell too
    $itemFixture = @(
        '| FR-X | designed | `docs/architecture/x/` | n |',
        '|---|---|---|---|',
        '| round | item | type | gate | DoD | d | commit | status | note |',
        '| 289 | FR-Y | impl | go | green | +3 | - | done | n |'
    )
    $itemPhases = @(Get-ItemPhasesFromLines $itemFixture)
    Check 'item phases stay in their section' ($itemPhases.Count -eq 1 -and $itemPhases[0] -eq 'designed')
    Check 'graceful STOP waits for the round boundary' (
        -not (Test-LoopStopRequested $false $true $false)
    )
    Check 'graceful STOP halts between rounds' (
        Test-LoopStopRequested $true $true $false
    )
    Check 'emergency STOP interrupts an active round' (
        Test-LoopStopRequested $false $false $true
    )
    Check 'dirty ledger without rotation stays uncommitted' (
        -not (Test-RotationPerformed @())
    )
    Check 'reported rotation is committed' (
        Test-RotationPerformed @('rotated')
    )

    $arguments = @(New-CodexArguments 'test-model' 'high')
    $execIndex = [Array]::IndexOf([object[]]$arguments, 'exec')
    Check 'args bypass approvals' ($arguments[0] -eq '--dangerously-bypass-approvals-and-sandbox')
    Check 'args global before exec' ([Array]::IndexOf([object[]]$arguments, '-C') -lt $execIndex)
    Check 'args model effort' ($arguments -contains 'model_reasoning_effort=high')
    Check 'args JSON stdin' (($arguments[-2] -eq '--json') -and ($arguments[-1] -eq '-'))
    Check 'args persistent fresh thread' (-not ($arguments -contains '--ephemeral') -and -not ($arguments -contains 'resume'))
    Check 'quote path with spaces' ((ConvertTo-NativeArgument 'C:\Repo With Space') -eq '"C:\Repo With Space"')

    $resumeArguments = @(New-CodexResumeArguments 'test-model' 'high' 'thread-early')
    $resumeIndex = [Array]::IndexOf([object[]]$resumeArguments, 'resume')
    Check 'resume args keep the same thread' (
        $resumeIndex -gt 0 -and
        $resumeArguments[$resumeIndex + 1] -eq '--strict-config' -and
        $resumeArguments[-2] -eq 'thread-early' -and
        $resumeArguments[-1] -eq '-'
    )

    $prematureViolations = @(Get-RoundCompletionViolations `
        -preHead 'head-1' `
        -postHead 'head-1' `
        -roundKind 'impl' `
        -statuses @('designed') `
        -workingTreeDirty $true `
        -summary $premature)
    # a declared external obstacle is a legitimate impl ending; 'designed' never is
    $needsFixViolations = @(Get-RoundCompletionViolations `
        -preHead 'head-1' `
        -postHead 'head-2' `
        -roundKind 'impl' `
        -statuses @('needs-fix') `
        -workingTreeDirty $false `
        -summary $completed)
    Check 'needs-fix ends an impl round' (($needsFixViolations -match 'implementation status').Count -eq 0)
    Check 'premature turn requires same-round continuation' (
        $prematureViolations.Count -eq 4 -and
        ($prematureViolations -match 'command_execution').Count -eq 1
    )
    $staleCommandViolations = @(Get-RoundCompletionViolations `
        -preHead 'head-1' `
        -postHead 'head-2' `
        -roundKind 'impl' `
        -statuses @('done') `
        -workingTreeDirty $false `
        -roundWorkloadActive $false `
        -summary $premature)
    Check 'stale command event does not resume a completed round' (
        $staleCommandViolations.Count -eq 0
    )
    $liveCommandViolations = @(Get-RoundCompletionViolations `
        -preHead 'head-1' `
        -postHead 'head-2' `
        -roundKind 'impl' `
        -statuses @('done') `
        -workingTreeDirty $false `
        -roundWorkloadActive $true `
        -summary $premature)
    Check 'live workload behind an unfinished command still resumes' (
        $liveCommandViolations.Count -eq 1 -and
        $liveCommandViolations[0] -match 'command_execution'
    )
    $completeViolations = @(Get-RoundCompletionViolations `
        -preHead 'head-1' `
        -postHead 'head-2' `
        -roundKind 'impl' `
        -statuses @('done') `
        -workingTreeDirty $false `
        -summary $completed)
    Check 'committed clean implementation completes the round' ($completeViolations.Count -eq 0)
    $continuation = New-CodexContinuationPrompt $prematureViolations 120
    Check 'continuation preserves round and time budget' (
        $continuation -match 'same round in the same thread' -and
        $continuation -match '120 seconds remain'
    )

    $adapted = New-CodexRoundPrompt 'CONTRACT'
    Check 'prompt adapter' ($adapted -match '<codex_provider_adapter>' -and $adapted.EndsWith('CONTRACT'))
    $source = [System.IO.File]::ReadAllText($PSCommandPath, $utf8)
    Check 'PowerShell 5.1 source is ASCII-only' (-not ($source -match '[^\x00-\x7F]'))
    if (-not $PSBoundParameters.ContainsKey('DesignModel') -and
        -not $PSBoundParameters.ContainsKey('ImplModel')) {
        Check 'default split models use astra for design and impl' (
            $DesignModel -eq 'gpt-6-astra' -and $ImplModel -eq 'gpt-6-astra'
        )
    }
    if (-not $PSBoundParameters.ContainsKey('DesignEffort') -and
        -not $PSBoundParameters.ContainsKey('ImplEffort') -and
        -not $PSBoundParameters.ContainsKey('MaxRoundMinutes')) {
        Check 'default split efforts use medium for design and impl' (
            $DesignEffort -eq 'medium' -and $ImplEffort -eq 'medium'
        )
        Check 'default round timeout is 400 minutes' ($MaxRoundMinutes -eq 400)
    }

    if ($env:OS -eq 'Windows_NT') {
        $stderrProbe = Invoke-CodexProbe $env:ComSpec @('/d', '/c', 'echo probe 1>&2')
        Check 'probe accepts successful native stderr' (
            $stderrProbe.ExitCode -eq 0 -and $stderrProbe.Text -eq 'probe'
        )
        $failedProbe = Invoke-CodexProbe $env:ComSpec @('/d', '/c', 'echo failed 1>&2 & exit /b 7')
        Check 'probe preserves native exit code' (
            $failedProbe.ExitCode -eq 7 -and $failedProbe.Text -eq 'failed'
        )

        $runnerOut = [System.IO.Path]::GetTempFileName()
        $runnerErr = [System.IO.Path]::GetTempFileName()
        $timeoutOut = [System.IO.Path]::GetTempFileName()
        $timeoutErr = [System.IO.Path]::GetTempFileName()
        $stopOut = [System.IO.Path]::GetTempFileName()
        $stopErr = [System.IO.Path]::GetTempFileName()
        $stopMarker = [System.IO.Path]::GetTempFileName()
        $lingerOut = [System.IO.Path]::GetTempFileName()
        $lingerErr = [System.IO.Path]::GetTempFileName()
        $orphanOut = [System.IO.Path]::GetTempFileName()
        $orphanErr = [System.IO.Path]::GetTempFileName()
        try {
            $hostPowerShell = (Get-Command powershell.exe -CommandType Application).Source
            $fixtureCommand = '$input | Out-Null; @{ type = ''thread.started''; thread_id = $env:HONESTML_LOOP_KIND } | ConvertTo-Json -Compress; Write-Output ''{"type":"turn.completed"}'''
            $runnerResult = Invoke-CodexRoundProcess `
                -command $hostPowerShell `
                -arguments @('-NoProfile', '-Command', $fixtureCommand) `
                -promptText 'fixture prompt' `
                -outFile $runnerOut `
                -errFile $runnerErr `
                -timeoutSeconds 10 `
                -roundKind 'fixture'
            $runnerSummary = Read-CodexRunSummary $runnerOut
            Check 'runner streams JSONL and round kind' (
                $runnerResult.ExitCode -eq 0 -and
                -not $runnerResult.TimedOut -and
                $runnerSummary.ThreadId -eq 'fixture' -and
                $runnerSummary.Terminal -eq 'completed'
            )

            $timeoutResult = Invoke-CodexRoundProcess `
                -command $hostPowerShell `
                -arguments @('-NoProfile', '-Command', 'Start-Sleep -Seconds 30') `
                -promptText 'fixture prompt' `
                -outFile $timeoutOut `
                -errFile $timeoutErr `
                -timeoutSeconds 1 `
                -roundKind 'fixture'
            $timeoutOk = (
                $timeoutResult.ExitCode -eq 124 -and
                $timeoutResult.TimedOut -and
                $timeoutResult.ProcessTreeStopConfirmed -and
                $timeoutResult.DurationSeconds -lt 15
            )
            if (-not $timeoutOk) { $timeoutResult | Format-List | Out-String | Write-Host }
            Check 'runner timeout is bounded and kills the tree' $timeoutOk

            $stopResult = Invoke-CodexRoundProcess `
                -command $hostPowerShell `
                -arguments @('-NoProfile', '-Command', 'Start-Sleep -Seconds 30') `
                -promptText 'fixture prompt' `
                -outFile $stopOut `
                -errFile $stopErr `
                -timeoutSeconds 10 `
                -roundKind 'fixture' `
                -stopMarker $stopMarker
            $stopOk = (
                $stopResult.ExitCode -eq 0 -and
                $stopResult.StopRequested -and
                -not $stopResult.TimedOut -and
                $stopResult.ProcessTreeStopConfirmed -and
                $stopResult.DurationSeconds -lt 15
            )
            if (-not $stopOk) { $stopResult | Format-List | Out-String | Write-Host }
            Check 'runner STOP-NOW is bounded and kills the tree' $stopOk

            # A turn that ends while the agent process stays alive: the round is over, so the
            # runner must return on the terminal event instead of waiting out the wall clock.
            $lingerCommand = '$input | Out-Null; Write-Output ''{"type":"thread.started","thread_id":"linger"}''; Write-Output ''{"type":"turn.completed"}''; Start-Sleep -Seconds 30'
            $lingerResult = Invoke-CodexRoundProcess `
                -command $hostPowerShell `
                -arguments @('-NoProfile', '-Command', $lingerCommand) `
                -promptText 'fixture prompt' `
                -outFile $lingerOut `
                -errFile $lingerErr `
                -timeoutSeconds 60 `
                -roundKind 'fixture' `
                -terminalGraceSeconds 2
            $lingerSummary = Read-CodexRunSummary $lingerOut
            $lingerOk = (
                $lingerResult.ExitCode -eq 0 -and
                -not $lingerResult.TimedOut -and
                $lingerResult.ProcessTreeStopConfirmed -and
                $lingerResult.DurationSeconds -lt 15 -and
                $lingerSummary.Terminal -eq 'completed'
            )
            if (-not $lingerOk) { $lingerResult | Format-List | Out-String | Write-Host }
            Check 'runner returns on the terminal event when the process lingers' $lingerOk

            # A child that outlives the agent keeps the inherited pipes open, so EOF never
            # arrives; once the agent itself is gone the runner drains briefly and returns.
            $orphanCommand = '$input | Out-Null; Start-Process -NoNewWindow -FilePath powershell.exe -ArgumentList ''-NoProfile'', ''-Command'', ''Start-Sleep -Seconds 20'''
            $orphanResult = Invoke-CodexRoundProcess `
                -command $hostPowerShell `
                -arguments @('-NoProfile', '-Command', $orphanCommand) `
                -promptText 'fixture prompt' `
                -outFile $orphanOut `
                -errFile $orphanErr `
                -timeoutSeconds 60 `
                -roundKind 'fixture'
            $orphanOk = (
                $orphanResult.ExitCode -eq 0 -and
                -not $orphanResult.TimedOut -and
                $orphanResult.DurationSeconds -lt 15
            )
            if (-not $orphanOk) { $orphanResult | Format-List | Out-String | Write-Host }
            Check 'runner returns when a surviving child holds the pipes open' $orphanOk
        } finally {
            Remove-Item -LiteralPath `
                $runnerOut, $runnerErr, $timeoutOut, $timeoutErr, $stopOut, $stopErr, $stopMarker, `
                $lingerOut, $lingerErr, $orphanOut, $orphanErr `
                -Force
        }
    }

    # the halt vocabulary of the ledger: 'blocked-only' is the round TYPE, its status is 'blocked'
    $ledgerFixture = @('| 9 | FR-X | blocked-only | - | - | - | - | blocked | waiting |')
    $fixtureTypes = @(Get-RoundColumnFromLines $ledgerFixture 2)
    $fixtureStatuses = @(Get-RoundColumnFromLines $ledgerFixture 7)
    Check 'ledger type identifies blocked-only' (
        $fixtureTypes.Count -eq 1 -and $fixtureTypes[0] -eq 'blocked-only'
    )
    Check 'ledger status of a blocked-only round' (
        $fixtureStatuses.Count -eq 1 -and $fixtureStatuses[0] -eq 'blocked'
    )
    # the volume line is measured from git: numstat rows carry the round's delta, `git grep -c` rows
    # the size of each tree; a binary file has no lines, and paths outside src/ and tests/ no volume
    $volumeRows = @(
        "12`t3`tsrc/honestml/adapters/onnx_export.py",
        "-`t-`ttests/fixtures/native_artifacts/catboost/artifact/model.cbm",
        "40`t0`ttests/unit/test_onnx_export.py",
        "9`t1`tdocs/loop/state.md"
    )
    $volumeSizes = @('src/honestml/adapters/onnx_export.py:100', 'tests/unit/test_onnx_export.py:250', 'docs/loop/state.md:900')
    Check 'volume line counts src and tests only' (
        (Format-CodeVolumeNote $volumeRows $volumeSizes) -eq 'lines: src +12/-3 of 100, tests +40/-0 of 250'
    )
    Check 'a round that touched no code gets no volume line' (
        (Format-CodeVolumeNote @("9`t1`tdocs/loop/state.md") @('docs/loop/state.md:900')) -eq ''
    )
    # the ledger verdicts belong to the LOOP contract; another contract keeps its own register and
    # would be judged every round by a row it never wrote
    $loopContract = Join-Path $root 'docs\loop\LOOP.md'
    Check 'loop contract keeps the ledger' (Test-KeepsLoopLedger $loopContract $loopContract)
    Check 'same contract via mixed separators' (
        Test-KeepsLoopLedger (Join-Path $root 'docs/loop/LOOP.md') $loopContract
    )
    Check 'debug contract keeps its own register' (
        -not (Test-KeepsLoopLedger (Join-Path $root 'docs\loop\DEBUG.md') $loopContract)
    )
    Check 'a foreign round is named by its contract' (
        (Get-ContractKind (Join-Path $root 'docs\loop\DEBUG.md')) -eq 'debug'
    )
    # rounds here read .agents/skills while the Claude driver reads .CLAUDE/skills;
    # substantive divergence would make the two drivers run different contracts unnoticed
    $parityOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot 'skill-parity.ps1'))
    Check 'skill trees differ only in the provider layer' ($LASTEXITCODE -eq 0)
    if ($LASTEXITCODE -ne 0) { $parityOutput | ForEach-Object { "     $_" } }
    # the same contract is carried by AGENTS.md and CLAUDE.md; a rule that reaches only one
    # runtime is drift no one is watching
    $instructionOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot 'instruction-parity.ps1'))
    Check 'instruction files differ only in the provider layer' ($LASTEXITCODE -eq 0)
    if ($LASTEXITCODE -ne 0) { $instructionOutput | ForEach-Object { "     $_" } }
    $configOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot 'config-check.ps1'))
    Check 'hook config and skill frontmatter are valid' ($LASTEXITCODE -eq 0)
    if ($LASTEXITCODE -ne 0) { $configOutput | ForEach-Object { "     $_" } }

    if ($failures -gt 0) { "SELFTEST FAILED ($failures)"; exit 1 }
    'SELFTEST OK'
    exit 0
}

if ($Preflight) {
    $preflightFailures = 0
    if (-not (Test-Path $loopPrompt)) {
        Write-Host "FAIL prompt missing: $loopPrompt" -ForegroundColor Red
        $preflightFailures++
    } else {
        Write-Host "PASS prompt: $loopPrompt" -ForegroundColor Green
    }
    try {
        $resolvedCodex = Resolve-CodexCli $CodexCommand
        $version = Invoke-CodexProbe $resolvedCodex @('--version')
        $auth = Invoke-CodexProbe $resolvedCodex @('login', 'status')
        if ($version.ExitCode -eq 0) {
            Write-Host "PASS CLI: $($version.Text) [$resolvedCodex]" -ForegroundColor Green
        } else {
            Write-Host "FAIL CLI: $($version.Text)" -ForegroundColor Red
            $preflightFailures++
        }
        if ($auth.ExitCode -eq 0) {
            Write-Host "PASS auth: $($auth.Text)" -ForegroundColor Green
        } else {
            Write-Host "FAIL auth: $($auth.Text)" -ForegroundColor Red
            $preflightFailures++
        }
    } catch {
        Write-Host "FAIL CLI: $($_.Exception.Message)" -ForegroundColor Red
        $preflightFailures++
    }

    Push-Location $root
    try {
        $gitRoot = (git rev-parse --show-toplevel 2>$null)
        if ($LASTEXITCODE -eq 0) {
            Write-Host "PASS git: $gitRoot" -ForegroundColor Green
            $dirty = [bool](git status --porcelain)
            Write-Host "INFO working_tree=$(if ($dirty) { 'dirty' } else { 'clean' })"
        } else {
            Write-Host 'FAIL git repository unavailable' -ForegroundColor Red
            $preflightFailures++
        }
    } finally {
        Pop-Location
    }
    Write-Host "INFO stop_file=$(if (Test-Path $stopFile) { 'present' } else { 'absent' })"
    Write-Host "INFO stop_now_file=$(if (Test-Path $stopNowFile) { 'present' } else { 'absent' })"
    if ($preflightFailures -gt 0) { "PREFLIGHT FAILED ($preflightFailures)"; exit 1 }
    'PREFLIGHT OK'
    exit 0
}

if (-not (Test-Path $loopPrompt)) { throw "Round contract not found: $loopPrompt" }

$keepsLedger = Test-KeepsLoopLedger $loopPrompt (Join-Path $root 'docs\loop\LOOP.md')
if (-not $keepsLedger) {
    Write-Host "Contract $Prompt keeps its own register - ledger verdicts are off this run." -ForegroundColor DarkCyan
}

if (-not $BypassPermissions) {
    Write-Host 'WARNING: autonomous rounds must edit, run commands and commit without prompts.' -ForegroundColor Yellow
    Write-Host 'Re-run with -BypassPermissions after reviewing the driver and project hooks.' -ForegroundColor Yellow
    Write-Host 'Recommended first run: -BypassPermissions -MaxIterations 1' -ForegroundColor Yellow
    exit 2
}

Set-Location $root
if (Test-LoopStopRequested $true (Test-Path $stopFile) (Test-Path $stopNowFile)) {
    Write-Host 'STOP marker present - halting.' -ForegroundColor Yellow
    Send-LoopNote 'halted: STOP marker present.'
    return
}
if ((git status --porcelain) -and -not $Force) {
    Write-Host 'Working tree is dirty. Commit/stash first; use -Force only if the round may commit those changes.' -ForegroundColor Red
    exit 2
}

$resolvedCodex = Resolve-CodexCli $CodexCommand
$version = Invoke-CodexProbe $resolvedCodex @('--version')
if ($version.ExitCode -ne 0) { throw "Codex CLI failed: $($version.Text)" }
$auth = Invoke-CodexProbe $resolvedCodex @('login', 'status')
if ($auth.ExitCode -ne 0) { throw "Codex authentication required: $($auth.Text)" }

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
& (Join-Path $PSScriptRoot 'prune-logs.ps1') | Write-Host  # keep the log dir readable across nights
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$splitMode = -not $PSBoundParameters.ContainsKey('Model')
$iteration = 0
$transientRetries = 0
# commit subject of the previous completed round; feeds the no-op repetition guard
$lastSubject = ''
$driverExitCode = 0
$activeRoundFile = Join-Path $logDir 'active-round.json'

if ($splitMode) {
    $designLabel = if ($DesignModel) { $DesignModel } else { '<default>' }
    $implLabel = if ($ImplModel) { $ImplModel } else { '<default>' }
    Write-Host "HonestML Codex loop @ $root | split: design=$designLabel/$DesignEffort impl=$implLabel/$ImplEffort" -ForegroundColor Cyan
} else {
    $modelLabel = if ($Model) { $Model } else { '<default>' }
    Write-Host "HonestML Codex loop @ $root | single: model=$modelLabel effort=$Effort" -ForegroundColor Cyan
}

while ($true) {
    if (Test-LoopStopRequested $true (Test-Path $stopFile) (Test-Path $stopNowFile)) {
        Write-Host 'STOP marker present - halting.' -ForegroundColor Yellow
        Send-LoopNote 'halted: STOP marker present.'
        break
    }
    if ($MaxIterations -gt 0 -and $iteration -ge $MaxIterations) {
        Write-Host "Reached -MaxIterations ($MaxIterations)." -ForegroundColor Yellow
        Send-LoopNote "finished: reached -MaxIterations ($MaxIterations)."
        break
    }

    # rotation commits the ledger, so it must not run over an uncommitted one: a half-written round
    # row would ride along into the rotation commit and stop reading as new since HEAD.
    if (git status --porcelain -- docs/loop/state.md) {
        Write-Host 'Ledger has uncommitted changes - skipping rotation this round.' -ForegroundColor DarkYellow
    } else {
        $rotationMessages = @(& (Join-Path $PSScriptRoot 'rotate.ps1'))
        $rotationMessages | ForEach-Object { Write-Host $_ -ForegroundColor DarkCyan }
        if (Test-RotationPerformed $rotationMessages) {
            git add -- docs/loop/state.md docs/loop/state-archive.md
            if ($LASTEXITCODE -ne 0) { throw 'Failed to stage ledger rotation.' }
            git commit -q -m 'chore(loop): rotate ledger (old rounds -> docs/loop/state-archive.md)' | Out-Null
            if ($LASTEXITCODE -ne 0) { throw 'Failed to commit ledger rotation.' }
        }
    }

    $iteration++
    if ($splitMode) {
        # Only the LOOP ledger can say whether work is owed design or code. Under another contract
        # the item phases describe a queue this run is not working, and calling the round 'impl'
        # would arm the Stop hook with a completion contract it was never given.
        $kind = if ($keepsLedger) { Get-NextRoundKind } else { Get-ContractKind $loopPrompt }
        if ($kind -eq 'impl') {
            $roundModel = $ImplModel
            $roundEffort = $ImplEffort
        } else {
            $roundModel = $DesignModel
            $roundEffort = $DesignEffort
        }
    } else {
        $kind = 'single'
        $roundModel = $Model
        $roundEffort = $Effort
    }

    $roundModelLabel = if ($roundModel) { $roundModel } else { '<default>' }
    Write-Host "=== Codex round $iteration ($(Get-Date -Format 'HH:mm:ss')) | $kind $roundModelLabel/$roundEffort ===" -ForegroundColor Cyan
    # relaunches after a transport failure get their own suffix: the log of the failed attempt survives
    $attemptSuffix = if ($transientRetries -gt 0) { "-t$transientRetries" } else { '' }
    $baseLogName = "$stamp-codex-r$iteration$attemptSuffix"
    $contract = [System.IO.File]::ReadAllText($loopPrompt, $utf8)
    $roundPrompt = New-CodexRoundPrompt $contract
    $preHead = (git rev-parse HEAD).Trim()
    $roundClock = [System.Diagnostics.Stopwatch]::StartNew()
    $segment = 0
    $threadId = $null
    $continuationPrompt = $null
    $roundComplete = $false
    $roundStopped = $false
    $summary = Get-CodexRunSummaryFromLines @()
    $processResult = [pscustomobject]@{
        ExitCode = 1
        TimedOut = $false
        StopRequested = $false
        ProcessTreeStopConfirmed = $true
        DurationSeconds = 0
    }
    $outFile = $null
    $errFile = $null

    $activeRound = [ordered]@{
        kind = $kind
        expires_at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + ($MaxRoundMinutes * 60) + 300
    } | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($activeRoundFile, $activeRound, $utf8)
    try {
        while (-not $roundComplete) {
            if (Test-Path -LiteralPath $stopNowFile) {
                $roundStopped = $true
                $processResult.ExitCode = 0
                $processResult.StopRequested = $true
                break
            }
            $remainingSeconds = [int][math]::Ceiling(
                ($MaxRoundMinutes * 60) - $roundClock.Elapsed.TotalSeconds
            )
            if ($remainingSeconds -le 0) {
                $processResult = [pscustomobject]@{
                    ExitCode = 124
                    TimedOut = $true
                    StopRequested = $false
                    ProcessTreeStopConfirmed = $true
                    DurationSeconds = [math]::Round($roundClock.Elapsed.TotalSeconds, 1)
                }
                break
            }

            $segment++
            $segmentSuffix = if ($segment -eq 1) { '' } else { "-s$segment" }
            $outFile = Join-Path $logDir "$baseLogName$segmentSuffix.out"
            $errFile = Join-Path $logDir "$baseLogName$segmentSuffix.err"
            if ($segment -eq 1) {
                $arguments = @(New-CodexArguments $roundModel $roundEffort)
                $promptText = $roundPrompt
            } else {
                $arguments = @(New-CodexResumeArguments $roundModel $roundEffort $threadId)
                $promptText = $continuationPrompt
                Write-Host "=== Resume round $iteration, segment $segment ($remainingSeconds s remain) ===" -ForegroundColor Cyan
            }

            $segmentResult = Invoke-CodexRoundProcess `
                -command $resolvedCodex `
                -arguments $arguments `
                -promptText $promptText `
                -outFile $outFile `
                -errFile $errFile `
                -timeoutSeconds $remainingSeconds `
                -roundKind $kind `
                -stopMarker $stopNowFile
            $processResult = [pscustomobject]@{
                ExitCode = $segmentResult.ExitCode
                TimedOut = $segmentResult.TimedOut
                StopRequested = $segmentResult.StopRequested
                ProcessTreeStopConfirmed = $segmentResult.ProcessTreeStopConfirmed
                DurationSeconds = [math]::Round($roundClock.Elapsed.TotalSeconds, 1)
            }
            $summary = Read-CodexRunSummary $outFile
            if ($summary.ThreadId) {
                if ($threadId -and $summary.ThreadId -ne $threadId) {
                    throw "Codex resume changed thread from $threadId to $($summary.ThreadId)."
                }
                $threadId = $summary.ThreadId
            }

            if ($processResult.StopRequested) {
                $roundStopped = $true
                break
            }
            if ($processResult.TimedOut -or
                $processResult.ExitCode -ne 0 -or
                $summary.Terminal -ne 'completed') {
                break
            }

            $postHead = (git rev-parse HEAD).Trim()
            $statuses = @(Get-RoundStatuses)
            $roundWorkloadActive = $false
            if (@($summary.InFlightCommandIds).Count -gt 0) {
                $roundWorkloadActive = Test-CodexRoundWorkloadActive $root
            }
            $violations = @(Get-RoundCompletionViolations `
                -preHead $preHead `
                -postHead $postHead `
                -roundKind $kind `
                -statuses $statuses `
                -workingTreeDirty ([bool](git status --porcelain)) `
                -roundWorkloadActive $roundWorkloadActive `
                -summary $summary)
            if ($violations.Count -eq 0) {
                if (@($summary.InFlightCommandIds).Count -gt 0) {
                    Write-Host "Codex emitted stale command events, but the round contract is complete and no workload process remains." -ForegroundColor DarkYellow
                }
                $roundComplete = $true
                break
            }
            if (Test-LoopStopRequested $false (Test-Path -LiteralPath $stopFile) (Test-Path -LiteralPath $stopNowFile)) {
                # emergency stop may arrive after a segment exits but before the next resume
                $roundStopped = $true
                $processResult.StopRequested = $true
                break
            }
            if (-not $threadId) {
                Write-Host 'Codex turn is incomplete and emitted no thread id; cannot resume it.' -ForegroundColor Red
                $processResult.ExitCode = 1
                break
            }
            Write-Host "Codex turn ended before round completion:" -ForegroundColor Yellow
            $violations | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
            $remainingSeconds = [int][math]::Max(
                0,
                [math]::Floor(($MaxRoundMinutes * 60) - $roundClock.Elapsed.TotalSeconds)
            )
            $continuationPrompt = New-CodexContinuationPrompt $violations $remainingSeconds
        }
    } finally {
        $roundClock.Stop()
        Remove-Item -LiteralPath $activeRoundFile -Force -ErrorAction SilentlyContinue
    }
    $postHead = (git rev-parse HEAD).Trim()
    $processResult.DurationSeconds = [math]::Round($roundClock.Elapsed.TotalSeconds, 1)
    Write-RoundMetrics `
        -round $iteration `
        -kind $kind `
        -roundModel $roundModel `
        -roundEffort $roundEffort `
        -segments $segment `
        -processResult $processResult `
        -summary $summary `
        -headSha $postHead

    if ($roundStopped) {
        Write-Host 'STOP marker present - halting.' -ForegroundColor Yellow
        Send-LoopNote 'halted: STOP marker present.'
        if (-not $processResult.ProcessTreeStopConfirmed) {
            Write-Host 'WARNING: the Codex process tree did not confirm termination.' -ForegroundColor Red
        }
        break
    }
    if ($processResult.TimedOut) {
        Write-Host "Round $iteration exceeded $MaxRoundMinutes minute(s); process tree stopped, WIP kept." -ForegroundColor Red
        if (-not $processResult.ProcessTreeStopConfirmed) {
            Write-Host 'WARNING: the Codex process tree did not confirm termination.' -ForegroundColor Red
        }
        Send-LoopNote "halted: round $iteration exceeded $MaxRoundMinutes min; process tree stopped, WIP kept."
        $driverExitCode = 124
        break
    }
    if ($processResult.ExitCode -ne 0 -or $summary.Terminal -ne 'completed') {
        if (Test-CodexRateLimited $summary) {
            Write-Host 'Codex reported a rate/usage limit. Halting with WIP kept; relaunch after capacity resets.' -ForegroundColor Yellow
            Send-LoopNote "halted: round $iteration hit a rate/usage limit; WIP kept, relaunch after reset."
        } else {
            $failureText = Get-RoundFailureText $summary $errFile
            if ((Test-CodexTransientFailure $failureText) -and $transientRetries -lt $MaxTransientRetries) {
                $transientRetries++
                Write-Host "Transport failure (exit $($processResult.ExitCode), terminal $($summary.Terminal)); relaunching round $iteration on its WIP ($transientRetries/$MaxTransientRetries)." -ForegroundColor Yellow
                $iteration--  # same round number; a fresh thread resumes from the files the round already wrote
                continue
            }
            Write-Host "Codex round failed: exit=$($processResult.ExitCode), terminal=$($summary.Terminal). WIP kept." -ForegroundColor Red
            Send-LoopNote "halted: round $iteration failed (exit=$($processResult.ExitCode), terminal=$($summary.Terminal)); WIP kept."
        }
        Write-Host "Logs: $outFile | $errFile" -ForegroundColor DarkYellow
        $driverExitCode = if ($processResult.ExitCode -gt 0) { $processResult.ExitCode } else { 1 }
        break
    }
    if (-not $roundComplete) {
        Write-Host "Round $iteration could not reach its repository completion contract. WIP kept." -ForegroundColor Red
        Send-LoopNote "halted: round $iteration could not reach its completion contract; WIP kept."
        $driverExitCode = 1
        break
    }
    $transientRetries = 0

    # a round whose commit repeats the previous subject added no new work, only another ledger
    # row: one repetition is the whole signal, so the second identical round ends the loop
    $subject = [string](git log -1 --format=%s)
    if ($subject -eq $lastSubject) {
        Write-Host "Round $iteration repeated the previous round's commit subject - no-progress loop, halting:" -ForegroundColor Red
        Write-Host "  $subject" -ForegroundColor Red
        Send-LoopNote "halted: round $iteration repeated the previous commit subject (no-progress loop): $subject"
        $driverExitCode = 1
        break
    }
    $lastSubject = $subject

    # Round boundaries are the loop's only regular report: nobody watches the console for hours, and
    # a driver that speaks only when it halts leaves every finished round invisible until it stops.
    # A run under another contract has no ledger status of its own and carries '?' instead.
    $statuses = @()
    $roundTypes = @()
    if ($keepsLedger) { $statuses = @(Get-RoundStatuses); $roundTypes = @(Get-RoundTypes) }
    $roundStatus = if ($statuses.Count -gt 0) { $statuses[-1] } else { '?' }
    $roundMinutes = [math]::Round($processResult.DurationSeconds / 60)
    $roundNote = "round $iteration ($kind, $roundStatus) done in $roundMinutes min: $subject"
    $volumeNote = Get-CodeVolumeNote $preHead
    if ($volumeNote) { $roundNote = "$roundNote`n$volumeNote" }
    Send-LoopNote $roundNote

    # The verdict below is read from the rounds table of the LOOP ledger. A run driving another
    # contract has no row there to be judged by, and the table's last row describes someone else's
    # work. Such a round is judged by its own register and by the commit checks above, which hold
    # for every contract.
    if (-not $keepsLedger) { continue }

    # 'blocked-only' is a round TYPE (LOOP.md section 5); the status column of such a row is 'blocked'.
    if ($roundTypes.Count -gt 0 -and $roundTypes[-1] -eq 'blocked-only') {
        Write-Host 'Last round = blocked-only: nothing ready - halting.' -ForegroundColor Yellow
        Send-LoopNote 'halted: blocked-only round, nothing ready.'
        break
    }
}

$finishColor = if ($driverExitCode -eq 0) { 'Green' } else { 'Red' }
Write-Host "Codex loop finished after $iteration round(s), exit=$driverExitCode. Ledger: docs/loop/state.md | blockers: docs/loop/blockers.md" -ForegroundColor $finishColor
exit $driverExitCode
