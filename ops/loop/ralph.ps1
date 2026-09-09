<#
.SYNOPSIS
  HonestML Autonomous Loop driver (Ralph; fresh process per round) with rate-limit wait and
  resume of a round that ended without its commit.

.DESCRIPTION
  Each iteration launches a FRESH `claude -p` process, feeding it the round contract
  docs/loop/LOOP.md. Fresh process = clean context: the heavy work (architecture-design,
  implementation) restarts from zero every round and never hits the window limit. Continuity
  lives in files (docs/loop/state.md, docs/loop/backlog.md) and git history. The driver itself
  burns no LLM context.

  Rate limits: headless `claude -p` draws from the SAME 5h/weekly subscription pool as
  interactive Claude Code. On a limit it exits non-zero and prints e.g.
  "You've hit your session limit . resets 3:45pm". There is no built-in wait-until-reset,
  so this driver detects the message, parses the reset time, sleeps until then (+buffer), and
  resumes the SAME round on its own session over the files it already wrote. A limit is a pause,
  not a verdict: the round's uncommitted work is never discarded. Waits longer than
  -MaxWaitHours (e.g. weekly resets, days away) exit cleanly with the reset time so a
  scheduled relaunch (see README) can resume later.

  Ledger rotation: before each round the driver moves old rows of the state.md rounds table
  to docs/loop/state-archive.md (ops/loop/rotate.ps1) and commits the rotation with its own
  chore(loop) commit — orientation reads stay cheap no matter how long the loop runs.

  Transport failures: a dropped connection or 5xx kills the CLI process mid-round, but the round's WIP
  stays on disk — the driver relaunches the SAME round (up to -MaxTransientRetries in a row, WIP kept,
  the fresh process resumes from files) instead of halting the run.

  Stop conditions: file docs/loop/STOP | -MaxIterations reached | last round type = blocked-only
  (nothing ready) | a round that repeats the previous round's commit subject (the loop is
  re-committing the same no-op) | a round that exits green with no commit even after its one resume
  (no-progress stall) | the same EXTERNAL-BLOCKER declared twice in a row | reset too far away |
  an implementation protocol violation | repeated Usage-Policy refusals (content-level stop, not
  capacity: retrying re-hits the same wall; WIP is kept in the tree for inspection).

.PARAMETER MaxIterations          Round cap per run. 0 = unlimited.
.PARAMETER MaxTurns               Per-round --max-turns cap (default 800). An aborted round keeps its WIP.
.PARAMETER SkipPermissions        Required for unattended runs (passes --dangerously-skip-permissions).
.PARAMETER Prompt                 Round-contract file fed each round (default docs\loop\LOOP.md).
.PARAMETER Model                  Pin ALL rounds to one model (single-model mode). Omit to use the design/impl split.
.PARAMETER Effort                 Reasoning effort for single-model mode: low|medium|high|xhigh|max (default high).
.PARAMETER DesignModel            Split mode: model for design rounds (default claude-opus-5[1m] — 1M window so the
                                  whole design package stays in one round).
.PARAMETER ImplModel              Split mode: model for impl rounds (default claude-opus-5[1m] — 1M window
                                  so a recommended 25-75 test slice and its review fit one round without
                                  compaction; the range is advisory).
.PARAMETER DesignEffort           Split mode: effort for design rounds (default high — the ADR/verdict synthesis).
.PARAMETER ImplEffort             Split mode: effort for impl rounds (default high).
.PARAMETER SubagentModel          Force ALL fan-out subagents (research/review) onto this model via
                                  CLAUDE_CODE_SUBAGENT_MODEL (default claude-opus-5[1m]). Empty = no override.
.PARAMETER MaxWaitHours           Longest rate-limit wait to sit through; beyond it, exit cleanly (default 6).
.PARAMETER BufferMinutes          Extra minutes added after the parsed reset time (default 2).
.PARAMETER DefaultBackoffMinutes  Wait used when a limit is detected but the reset time can't be parsed (default 30).
.PARAMETER MaxTransientRetries    Consecutive transport-level failures (dropped connection, 5xx) that relaunch the
                                  same round before the loop halts (default 3). The round's WIP is kept.
.PARAMETER MaxBackoffRetries      Consecutive rate-limit waits whose reset time could not be parsed before the loop
                                  halts (default 4). Each such wait doubles the previous one.
.PARAMETER Force                  Proceed even if the working tree is dirty at startup; the round continues from
                                  the files that are there.
.PARAMETER SelfTest               Run internal parser assertions and exit.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File ops/loop/ralph.ps1 -SkipPermissions -MaxIterations 1  # supervised single round
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File ops/loop/ralph.ps1 -SkipPermissions  # autonomous until STOP/blockers/limit
#>
[CmdletBinding()]
param(
    [int]$MaxIterations = 0,
    [ValidateRange(1, 10000)]
    [int]$MaxTurns = 800,
    [switch]$SkipPermissions,
    [string]$Prompt = 'docs\loop\LOOP.md',
    [string]$Model = 'claude-opus-5',
    [string]$Effort = 'high',
    [string]$DesignModel = 'claude-opus-5[1m]',
    [string]$ImplModel = 'claude-opus-5[1m]',
    [string]$DesignEffort = 'high',
    [string]$ImplEffort = 'high',
    [string]$SubagentModel = 'claude-opus-5[1m]',
    [double]$MaxWaitHours = 6,
    [int]$BufferMinutes = 2,
    [int]$DefaultBackoffMinutes = 30,
    [ValidateRange(0, 20)]
    [int]$MaxTransientRetries = 3,
    [int]$MaxBackoffRetries = 4,
    [switch]$Force,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'

# feed the Russian LOOP.md to claude as UTF-8 regardless of PowerShell edition
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8

$root       = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$loopPrompt = Join-Path $root $Prompt
$stopFile   = Join-Path $root 'docs\loop\STOP'
$ledger     = Join-Path $root 'docs\loop\state.md'
$logDir     = Join-Path $root 'ops\loop\logs'

# the two trees a round's code lands in; everything else a commit touches (docs, ops, the ledger)
# is not code volume
$script:CodeVolumeAreas = @('src', 'tests')

# --- helpers -----------------------------------------------------------------

# one readable console line from a stream-json event; $null = nothing worth printing.
# defensive: unknown/non-JSON lines -> $null (full raw stream still goes to the .out log).
function Format-StreamEvent([string]$line) {
    if (-not $line) { return $null }
    try { $o = $line | ConvertFrom-Json -ErrorAction Stop } catch { return $null }
    switch ($o.type) {
        'assistant' {
            $parts = @()
            foreach ($b in $o.message.content) {
                if ($b.type -eq 'text' -and $b.text.Trim()) { $parts += $b.text.Trim() }
                elseif ($b.type -eq 'tool_use') { $parts += "  -> $($b.name)" }
            }
            if ($parts) { return ($parts -join "`n") } else { return $null }
        }
        'system' { if ($o.subtype -eq 'api_retry') { return "  (retry: $($o.error) $($o.attempt)/$($o.max_retries))" } else { return $null } }
        'result' { return "[result: $($o.subtype)]" }
        default  { return $null }
    }
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

# Best-effort note to whoever is not watching the console. The tag names the project as well as the
# driver: one Telegram chat carries several repositories, and "[loop claude]" alone would not say
# which one is reporting.
function Send-LoopNote([string]$text) {
    try { & (Join-Path $PSScriptRoot 'notify.ps1') -Message "[honestml claude] $text" } catch {}
}

# A reset time may arrive with or without minutes ("resets 10am", "resets Fri 3:30pm"); a driver that
# only knows the long form treats the short one as no limit at all and halts on the next exit code.
function Test-RateLimited([string]$text) {
    if (-not $text) { return $false }
    return ($text -match "(?i)hit your (session|weekly|usage) limit") -or
           ($text -match "(?i)\bresets\s+(\w{3}\s+)?\d{1,2}(:\d{2})?\s*[ap]m")
}

# transport-level failure of the CLI process (dropped connection, 5xx, overloaded upstream): the round's
# WIP is intact on disk, so the same round is worth another process. Content-level errors (bad request,
# refusals, protocol violations) are NOT transient and must keep halting the loop.
function Test-TransientApiFailure([string]$text) {
    if (-not $text) { return $false }
    return $text -match "(?i)(connection closed|connection error|socket hang up|ECONNRESET|ETIMEDOUT|EPIPE|fetch failed|network error|overloaded|internal server error|50[234]\s|Bad Gateway|Service Unavailable|Gateway Time-?out)"
}

# repeated AUP refusals = content-level stop (the round's framing trips the model's policy layer)
function Test-PolicyRefusal([string]$text) {
    if (-not $text) { return $false }
    return ([regex]::Matches($text, 'violate our Usage Policy')).Count -ge 2
}

# What the driver is allowed to judge a round's ENDING by: the last stream events plus stderr.
# A limit or refusal message arrives there, while the body of the stream carries tool results —
# echoed file contents that merely QUOTE the limit or refusal wording. Judging the whole stream
# would let any such file halt the loop; this driver quotes the refusal wording itself, so a round
# that reads it would otherwise stop the loop and blame the round contract.
function Get-StreamVerdictText([string]$outText, [string]$errText, [int]$lines = 30) {
    return ((($outText -split "`n") | Select-Object -Last $lines) -join "`n") + "`n" + $errText
}

# the reset stamp both helpers below read; minutes are optional ("resets 10am")
$script:resetTimeRegex = "(?i)resets\s+(?:(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap]m)"

# whether the limit message named a reset time at all: a wait derived from a stamp is trustworthy,
# a wait that fell back to the flat default is a guess and must not be repeated unchanged
function Test-ResetTimeParsed([string]$text) {
    if (-not $text) { return $false }
    return [regex]::IsMatch($text, $script:resetTimeRegex)
}

# seconds to wait until the reported reset (+buffer); $DefaultBackoffMinutes if message present but unparseable
function Get-ResetWaitSeconds([string]$text, [int]$bufferMin, [int]$defaultBackoffMin) {
    $m = [regex]::Match($text, $script:resetTimeRegex)
    if (-not $m.Success) { return $defaultBackoffMin * 60 }
    $wd   = $m.Groups[1].Value
    $hour = [int]$m.Groups[2].Value
    $min  = if ($m.Groups[3].Success) { [int]$m.Groups[3].Value } else { 0 }
    $ampm = $m.Groups[4].Value.ToLower()
    if ($ampm -eq 'pm' -and $hour -ne 12) { $hour += 12 }
    if ($ampm -eq 'am' -and $hour -eq 12) { $hour = 0 }
    $now = Get-Date
    if ($wd) {
        $map = @{ Sun = 0; Mon = 1; Tue = 2; Wed = 3; Thu = 4; Fri = 5; Sat = 6 }
        $delta  = ($map[$wd] - [int]$now.DayOfWeek + 7) % 7
        $target = $now.Date.AddDays($delta).AddHours($hour).AddMinutes($min)
        if ($target -le $now) { $target = $target.AddDays(7) }
    } else {
        $target = $now.Date.AddHours($hour).AddMinutes($min)
        if ($target -le $now) { $target = $target.AddDays(1) }
    }
    $secs = [int]($target - $now).TotalSeconds + ($bufferMin * 60)
    if ($secs -lt 0) { return $defaultBackoffMin * 60 }
    return $secs
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

# The item column carries an id followed by a human label ("FR-020-x (adaptive_cascade_sizing)"); every
# comparison against it must take the id alone, or it matches nothing and says so to no one.
function Get-RoundItems {
    if (-not (Test-Path $ledger)) { return @() }
    return @(Get-RoundColumnFromLines ([System.IO.File]::ReadAllLines($ledger, $utf8)) 1 |
        ForEach-Object { ($_ -split '\s+')[0] })
}

function Get-RoundNotes {
    if (-not (Test-Path $ledger)) { return @() }
    return Get-RoundColumnFromLines ([System.IO.File]::ReadAllLines($ledger, $utf8)) 8
}

# What a needs-fix round declared it was blocked by, keyed by item: the same obstacle reported
# twice in a row means nothing moved it, and another round will not either.
function Get-ExternalBlockerCarryover([string]$item, [string]$note) {
    $marker = [regex]::Match($note, '(?i)EXTERNAL-BLOCKER:\s*(.+)$')
    if (-not $marker.Success) { return '' }
    return "$item|" + ($marker.Groups[1].Value.Trim() -replace '\s+', ' ')
}

function Get-RoundTypes {
    if (-not (Test-Path $ledger)) { return @() }
    return Get-RoundColumnFromLines ([System.IO.File]::ReadAllLines($ledger, $utf8)) 2
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

# How a round of a given RECORDED type is allowed to end. The type comes from the row the round
# itself wrote, never from the kind guessed before launch: a round is free to do design work under
# an impl label when the ledger says the design was not ready, and only its own row says what it
# actually did. An unknown or absent type ('—' in the type column) carries no expectation.
function Get-RoundTerminalStatuses([string]$recordedType) {
    switch ($recordedType) {
        'design'       { return @('designed', 'blocked') }
        'impl'         { return @('done', 'blocked', 'needs-fix') }
        'blocked-only' { return @('blocked') }
        default        { return @() }
    }
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

# What the loop owes work to, not what it did last: a package awaiting code (designed) or a
# round stopped by a declared obstacle (needs-fix) is an impl round waiting to happen. The
# tail of the rounds table only says what the previous round was, which is a different question.
function Test-ImplKind([string[]]$phases) {
    return @($phases | Where-Object { $_ -eq 'designed' -or $_ -eq 'needs-fix' }).Count -gt 0
}

function Get-NextRoundKind {
    if (Test-ImplKind (Get-ItemPhases)) { return 'impl' } else { return 'design' }
}

# rotate.ps1 stays silent when the table is short enough: only a reported rotation may be
# committed, so a round's own ledger edits never land under the rotation commit
function Test-RotationPerformed([string[]]$messages) {
    return $messages.Count -gt 0
}

# session id of the round's last result event: lets the driver resume that exact session instead
# of losing a round whose model ended its turn one step short of the commit
function Get-LastSessionId([string]$OutFile) {
    try {
        $sid = $null
        foreach ($line in [System.IO.File]::ReadAllLines($OutFile, $utf8)) {
            if ($line -match '"type"\s*:\s*"result"') {
                try {
                    $o = $line | ConvertFrom-Json -ErrorAction Stop
                    if ($o.session_id) { $sid = [string]$o.session_id }
                } catch {}
            }
        }
        return $sid
    } catch { return $null }
}

# append one JSON line of round telemetry to ops/loop/metrics.jsonl (git-ignored), parsed from
# the round's final stream-json result event; a telemetry failure never breaks the loop
function Write-RoundMetrics(
    [string]$OutFile, [int]$Round, [string]$Kind, [string]$RoundModel,
    [string]$RoundEffort, [int]$ExitCode, [string]$HeadSha
) {
    try {
        # background tasks can wake a finished session: the .out then holds SEVERAL result
        # events (one per segment). Cost is cumulative (take the last), turns are per-segment
        # (take the max = the main run); segment count is itself a wake-up telemetry signal
        $res = $null; $maxTurns = 0; $segments = 0; $maxDurMs = 0
        # two cost signals the result event does not carry. A wait call (Monitor/TaskOutput) makes
        # the round re-pay its whole context for a poll, and a compaction means the round outgrew
        # its window; both are invisible in tokens alone. The compaction subtype is matched loosely
        # because this stream has yet to produce one to name exactly.
        $waitCalls = 0; $compacts = 0
        foreach ($line in [System.IO.File]::ReadAllLines($OutFile, $utf8)) {
            if ($line -match '"type"\s*:\s*"(result|assistant|system)"') {
                try {
                    $o = $line | ConvertFrom-Json -ErrorAction Stop
                    if ($o.type -eq 'result') {
                        $res = $o; $segments++
                        if ([int]$o.num_turns -gt $maxTurns) { $maxTurns = [int]$o.num_turns }
                        if ([long]$o.duration_ms -gt $maxDurMs) { $maxDurMs = [long]$o.duration_ms }
                    } elseif ($o.type -eq 'assistant') {
                        foreach ($b in $o.message.content) {
                            if ($b.type -eq 'tool_use' -and $b.name -match '^(Monitor|TaskOutput|TaskGet|TaskList)$') { $waitCalls++ }
                        }
                    } elseif ($o.type -eq 'system' -and $o.subtype -match 'compact') {
                        $compacts++
                    }
                } catch {}
            }
        }
        # result.modelUsage is the only sound token source in this stream: the per-message
        # assistant usage counts neither subagent calls nor completed output. A round killed
        # before its result event therefore has UNKNOWN usage — report null, never 0, so a
        # lost measurement can never read as a round that cost nothing.
        $cr = $null; $cw = $null; $outTok = $null; $inTok = $null
        if ($res -and $res.modelUsage) {
            $cr = [long]0; $cw = [long]0; $outTok = [long]0; $inTok = [long]0
            foreach ($p in $res.modelUsage.PSObject.Properties) {
                $cr += [long]$p.Value.cacheReadInputTokens
                $cw += [long]$p.Value.cacheCreationInputTokens
                $outTok += [long]$p.Value.outputTokens
                # the whole input, cache included: paired with cache_read it answers "how much
                # context did this round carry" and "did the cache hold" in one ratio
                $inTok += [long]$p.Value.inputTokens + [long]$p.Value.cacheReadInputTokens +
                    [long]$p.Value.cacheCreationInputTokens
            }
        }
        $rec = [ordered]@{
            ts          = (Get-Date).ToString('s')
            provider    = 'claude'
            round       = $Round
            kind        = $Kind
            model       = $RoundModel
            effort      = $RoundEffort
            exit        = $ExitCode
            subtype     = if ($res) { $res.subtype } else { $null }
            turns       = if ($res) { $maxTurns } else { $null }
            segments    = $segments
            duration_s  = if ($res) { [math]::Round($maxDurMs / 1000) } else { $null }
            cost_usd    = if ($res) { [math]::Round($res.total_cost_usd, 2) } else { $null }
            input_tokens = $inTok
            cache_read  = $cr
            cache_write = $cw
            out_tokens  = $outTok
            wait_calls  = $waitCalls
            compacts    = $compacts
            head        = $HeadSha
        }
        $json = ($rec | ConvertTo-Json -Compress -Depth 5) + "`n"
        [System.IO.File]::AppendAllText((Join-Path $PSScriptRoot 'metrics.jsonl'), $json, $utf8)
    } catch {
        Write-Host "metrics: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

# --- self-test (parser assertions, no claude calls) --------------------------

if ($SelfTest) {
    $fail = 0
    function Check($name, $cond) { if ($cond) { "PASS $name" } else { $script:fail++; "FAIL $name" } }
    Check "detect session"  (Test-RateLimited "You've hit your session limit . resets 3:45pm")
    Check "detect weekly"   (Test-RateLimited "You've hit your weekly limit . resets Mon 12:00am")
    Check "no false-positive" (-not (Test-RateLimited "Round done, DoD green, committed."))
    $s = Get-ResetWaitSeconds "resets 3:45pm" 2 30      # parseable -> not the default backoff
    Check "session parses"  ($s -gt 0 -and $s -ne 1800)
    $w = Get-ResetWaitSeconds "weekly limit . resets Mon 12:00am" 2 30
    Check "weekly parses"   ($w -gt 0)
    $d = Get-ResetWaitSeconds "limit reached, try later" 2 30
    Check "unparseable->backoff" ($d -eq 1800)
    # a reset stamp may omit minutes; treating that form as "no limit" would halt the run instead
    Check "hour-only detected"  (Test-RateLimited "You've hit your session limit . resets 10am")
    $h = Get-ResetWaitSeconds "resets 10am" 2 30
    Check "hour-only parses"    ($h -gt 0 -and $h -ne 1800)
    Check "stamp is recognised" (Test-ResetTimeParsed "resets 10am")
    Check "no stamp, no parse"  (-not (Test-ResetTimeParsed "limit reached, try later"))
    # blind waits double per streak, so a wall that keeps answering the same way is not hammered
    Check "blind wait doubles"  ([int](1800 * [math]::Pow(2, 3 - 1)) -eq 7200)
    Check "policy refusal x2"   (Test-PolicyRefusal "API Error: ...violate our Usage Policy... retry ...violate our Usage Policy...")
    Check "policy x1 -> no"     (-not (Test-PolicyRefusal "a doc quoting 'violate our Usage Policy' once"))
    # the verdict text must exclude the stream body, where tool results echo file contents that
    # merely quote the limit/refusal wording (this driver's own source is such a file)
    $echoed = ((("tool_result: violate our Usage Policy and resets 9:50am") + "`n") * 60) + (("clean tail line" + "`n") * 30)
    $echoedVerdict = Get-StreamVerdictText $echoed ''
    Check "verdict ignores refusal quoted in the body" (-not (Test-PolicyRefusal $echoedVerdict))
    Check "verdict ignores limit quoted in the body"   (-not (Test-RateLimited $echoedVerdict))
    $realRefusal = (("work line" + "`n") * 60) + "API Error: violate our Usage Policy`nretry failed: violate our Usage Policy"
    Check "real refusal at the end still halts" (Test-PolicyRefusal (Get-StreamVerdictText $realRefusal ''))
    Check "refusal reported only on stderr still halts" (
        Test-PolicyRefusal (Get-StreamVerdictText (("work line" + "`n") * 60) "violate our Usage Policy / violate our Usage Policy")
    )
    $realLimit = (("work line" + "`n") * 60) + '{"type":"result","subtype":"error"} You have hit your session limit . resets 3:45pm'
    Check "real limit at the end still detected" (Test-RateLimited (Get-StreamVerdictText $realLimit ''))
    # the kind follows work still owed, not the order rows happen to sit in
    Check "designed anywhere->impl"  (Test-ImplKind @('done','designed','done'))
    Check "needs-fix anywhere->impl" (Test-ImplKind @('done','needs-fix'))
    Check "all settled->design"      (-not (Test-ImplKind @('done','blocked','done')))
    Check "no items->design"         (-not (Test-ImplKind @()))
    # the rounds header (| round | item | ...) has an id-shaped first cell too
    $itemFixture = @(
        '| FR-X | designed | `docs/architecture/x/` | n |',
        '|---|---|---|---|',
        '| round | item | type | gate | DoD | d | commit | status | note |',
        '| 289 | FR-Y | impl | go | green | +3 | - | done | n |'
    )
    $itemPhases = @(Get-ItemPhasesFromLines $itemFixture)
    Check "item phases stay in their section" ($itemPhases.Count -eq 1 -and $itemPhases[0] -eq 'designed')
    # the halt vocabulary of the ledger: 'blocked-only' is the round TYPE, its status is 'blocked'
    $ledgerRow   = @('| 9 | FR-X | blocked-only | — | — | — | — | blocked | waiting |')
    $rowTypes    = @(Get-RoundColumnFromLines $ledgerRow 2)
    $rowStatuses = @(Get-RoundColumnFromLines $ledgerRow 7)
    Check "ledger type identifies blocked-only" ($rowTypes.Count -eq 1 -and $rowTypes[0] -eq 'blocked-only')
    Check "ledger status of a blocked-only round" ($rowStatuses.Count -eq 1 -and $rowStatuses[0] -eq 'blocked')
    $implRow = @('| 8 | FR-X | impl | go | green | +20 | — | done | shipped |')
    # a round is judged by the type IT recorded, not by the kind guessed before launch
    Check "design closes at designed"   ('designed' -in (Get-RoundTerminalStatuses 'design'))
    Check "design cannot claim done"    ('done' -notin (Get-RoundTerminalStatuses 'design'))
    Check "impl closes at done"         ('done' -in (Get-RoundTerminalStatuses 'impl'))
    Check "impl may stop at needs-fix"  ('needs-fix' -in (Get-RoundTerminalStatuses 'impl'))
    Check "design cannot need fixing"   ('needs-fix' -notin (Get-RoundTerminalStatuses 'design'))
    # the item column carries a label after the id; a carryover key built from the whole cell
    # would never match the next round's and the repeat would pass unnoticed
    Check "carryover keys by item id"   ((Get-ExternalBlockerCarryover 'FR-1' 'EXTERNAL-BLOCKER: pyarrow missing') -eq 'FR-1|pyarrow missing')
    Check "carryover ignores plain note" ((Get-ExternalBlockerCarryover 'FR-1' 'ordinary progress note') -eq '')
    Check "carryover folds whitespace"  ((Get-ExternalBlockerCarryover 'FR-1' 'EXTERNAL-BLOCKER:  pyarrow   missing') -eq 'FR-1|pyarrow missing')
    Check "round items drop the label"  ((('FR-020-x (adaptive_cascade_sizing)' -split '\s+')[0]) -eq 'FR-020-x')
    Check "impl cannot claim designed"  ('designed' -notin (Get-RoundTerminalStatuses 'impl'))
    Check "blocked ends any round"      (('blocked' -in (Get-RoundTerminalStatuses 'design')) -and ('blocked' -in (Get-RoundTerminalStatuses 'impl')))
    Check "blocked-only closes blocked" ((Get-RoundTerminalStatuses 'blocked-only') -join ',' -eq 'blocked')
    # a row carrying '—' in the type column names no expectation and must not halt the loop
    Check "unknown type expects none"   ((Get-RoundTerminalStatuses '—').Count -eq 0)
    # the ledger verdicts belong to the LOOP contract; another contract keeps its own register and
    # would be halted every round by a rule it was never given
    $loopContract = Join-Path $root 'docs\loop\LOOP.md'
    Check "loop contract keeps the ledger" (Test-KeepsLoopLedger $loopContract $loopContract)
    Check "same contract via mixed separators" (
        Test-KeepsLoopLedger (Join-Path $root 'docs/loop/LOOP.md') $loopContract)
    Check "debug contract keeps its own register" (
        -not (Test-KeepsLoopLedger (Join-Path $root 'docs\loop\DEBUG.md') $loopContract))
    Check "a foreign round is named by its contract" (
        (Get-ContractKind (Join-Path $root 'docs\loop\DEBUG.md')) -eq 'debug')
    Check "ledger column split stays aligned" (
        (@(Get-RoundColumnFromLines $implRow 2))[0] -eq 'impl' -and
        (@(Get-RoundColumnFromLines $implRow 7))[0] -eq 'done'
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
    Check "volume line counts src and tests only" (
        (Format-CodeVolumeNote $volumeRows $volumeSizes) -eq 'lines: src +12/-3 of 100, tests +40/-0 of 250'
    )
    Check "a round that touched no code gets no volume line" (
        (Format-CodeVolumeNote @("9`t1`tdocs/loop/state.md") @('docs/loop/state.md:900')) -eq ''
    )
    $ev = Format-StreamEvent '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"},{"type":"tool_use","name":"Bash"}]}}'
    Check "format assistant" ($ev -match 'hi' -and $ev -match 'Bash')
    Check "dirty ledger without rotation stays uncommitted" (-not (Test-RotationPerformed @()))
    Check "reported rotation is committed" (Test-RotationPerformed @('rotated'))
    Check "transient: dropped connection" (Test-TransientApiFailure 'API Error: Connection closed mid-response. The response above may be incomplete.')
    Check "transient: upstream 503"       (Test-TransientApiFailure 'API Error 503 Service Unavailable')
    Check "transient: overloaded"         (Test-TransientApiFailure '{"type":"error","error":{"type":"overloaded_error"}}')
    Check "content error is not transient" (-not (Test-TransientApiFailure 'API Error 400: invalid request - max_tokens too large'))
    Check "protocol violation is not transient" (-not (Test-TransientApiFailure 'Round ended at needs-fix, not done'))
    # rounds here read .CLAUDE/skills while the Codex driver reads .agents/skills;
    # substantive divergence would make the two drivers run different contracts unnoticed
    $parityOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot 'skill-parity.ps1'))
    Check "skill trees differ only in the provider layer" ($LASTEXITCODE -eq 0)
    if ($LASTEXITCODE -ne 0) { $parityOutput | ForEach-Object { "     $_" } }
    # the same contract is carried by AGENTS.md and CLAUDE.md; a rule that reaches only one
    # runtime is drift no one is watching
    $instructionOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot 'instruction-parity.ps1'))
    Check "instruction files differ only in the provider layer" ($LASTEXITCODE -eq 0)
    if ($LASTEXITCODE -ne 0) { $instructionOutput | ForEach-Object { "     $_" } }
    $configOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot 'config-check.ps1'))
    Check "hook config and skill frontmatter are valid" ($LASTEXITCODE -eq 0)
    if ($LASTEXITCODE -ne 0) { $configOutput | ForEach-Object { "     $_" } }
    if ($fail -gt 0) { "SELFTEST FAILED ($fail)"; exit 1 } else { "SELFTEST OK"; exit 0 }
}

# --- preflight ---------------------------------------------------------------

if (-not (Test-Path $loopPrompt)) { throw "Round contract not found: $loopPrompt" }

$keepsLedger = Test-KeepsLoopLedger $loopPrompt (Join-Path $root 'docs\loop\LOOP.md')
if (-not $keepsLedger) {
    Write-Host "Contract $Prompt keeps its own register - ledger verdicts are off this run." -ForegroundColor DarkCyan
}

if (-not $SkipPermissions) {
    Write-Host "WARNING: the autonomous loop needs permission to edit/run/commit." -ForegroundColor Yellow
    Write-Host "Re-run with -SkipPermissions (claude --dangerously-skip-permissions)." -ForegroundColor Yellow
    Write-Host "Recommended first run, supervised: -SkipPermissions -MaxIterations 1" -ForegroundColor Yellow
    return
}

Set-Location $root
if ((git status --porcelain) -and -not $Force) {
    Write-Host "Working tree is dirty. Commit/stash first, or pass -Force to continue from the files that are there." -ForegroundColor Red
    return
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
& (Join-Path $PSScriptRoot 'prune-logs.ps1') | Write-Host  # keep the log dir readable across nights
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'  # unique per launch -> logs never overwrite across runs

# A crash leaves nobody watching the console; say so before the process dies.
trap { Send-LoopNote "driver crashed: $($_.Exception.Message)"; break }

# --- main loop ---------------------------------------------------------------

# Model per round: split mode (default) runs design rounds on DesignModel and impl rounds on ImplModel,
# chosen from the ledger BEFORE launch (the driver fixes the model per process). Passing -Model pins one
# model for every round (single mode). NO --fallback-model and no rate-limit model switch on purpose —
# never silently downgrade; a limit means wait until reset, or exit cleanly for the scheduled resumer.
$splitMode = -not $PSBoundParameters.ContainsKey('Model')
# force every round's fan-out subagents onto one model regardless of CLI alias mapping or what
# the orchestrator passes; child `claude -p` processes inherit this env var
if ($SubagentModel) { $env:CLAUDE_CODE_SUBAGENT_MODEL = $SubagentModel }
# round sessions must show up in the IDE's Local session history: the list hides
# entrypoint=sdk-cli transcripts (treated as programmatic SDK spawns), so pin the
# entrypoint explicitly no matter what environment the driver was launched from
$env:CLAUDE_CODE_ENTRYPOINT = 'claude-vscode'
# marks the session as a loop round for scoped hooks (no_background_in_rounds.py
# denies run_in_background only when this is set; interactive sessions unaffected)
$env:HONESTML_LOOP_ROUND = '1'
# transient server overload (HTTP 429/529) is retried inside the round rather than surfacing as a
# non-zero exit, which the driver would treat as a fatal halt. Default on; pre-set
# CLAUDE_CODE_RETRY_WATCHDOG in the environment to override.
if (-not $env:CLAUDE_CODE_RETRY_WATCHDOG) { $env:CLAUDE_CODE_RETRY_WATCHDOG = '1' }
$subInfo = if ($SubagentModel) { "subagents=$SubagentModel" } else { "subagents=inherit" }
$iter = 0
$transientRetries = 0
# rate-limit state: how many times this round already waited out a limit (log suffix), how many of
# those waits were blind guesses (backoff growth), and the session the wait should resume into
$rateLimitAttempts = 0
$backoffStreak = 0
$pendingResumeSid = ''
# the external blocker the previous round declared; a repeat of it halts the loop
$lastExternalBlocker = ''
# commit subject of the previous completed round; feeds the no-op repetition guard
$lastSubject = ''
# nudge for the resume-once path below. It points at the round contract instead of restating it,
# so LOOP.md stays the single source of the rules.
$resumePrompt = 'The round is NOT finished: no commit was created. Do not start a new item. Finish THIS round per LOOP.md section 5 - wait for or stop your own background tasks, get the DoD verdict, append the round row to the ledger, and create the phase commit.'
$rateLimitResumePrompt = 'The usage limit that interrupted you has reset. Continue THIS round from the files you already wrote - do not start a new item and do not redo finished work. Re-read your own uncommitted changes first, then carry the round through to the ledger row and the phase commit per LOOP.md section 5.'
if ($splitMode) {
    Write-Host "HonestML loop @ $root | split: design=$DesignModel/$DesignEffort impl=$ImplModel/$ImplEffort | $subInfo" -ForegroundColor Cyan
} else {
    Write-Host "HonestML loop @ $root | single: model=$Model effort=$Effort | $subInfo" -ForegroundColor Cyan
}

while ($true) {
    if (Test-Path $stopFile) { Write-Host "STOP file present - halting." -ForegroundColor Yellow; Send-LoopNote 'halted: STOP file present.'; break }
    if ($MaxIterations -gt 0 -and $iter -ge $MaxIterations) { Write-Host "Reached -MaxIterations ($MaxIterations)." -ForegroundColor Yellow; Send-LoopNote "finished: reached -MaxIterations ($MaxIterations)."; break }

    # keep the ledger cheap to read: rotate old round rows into docs/loop/state-archive.md before
    # the round starts. Rotation commits the ledger, so it must not run over an uncommitted ledger:
    # a half-written round row would ride along into the rotation commit, and every guard that asks
    # "is this row new since HEAD" would answer no for a row the round never finished.
    if (git status --porcelain -- docs/loop/state.md) {
        Write-Host "Ledger has uncommitted changes - skipping rotation this round." -ForegroundColor DarkYellow
    } else {
        $rotationMessages = @(& (Join-Path $PSScriptRoot 'rotate.ps1'))
        $rotationMessages | ForEach-Object { Write-Host $_ -ForegroundColor DarkCyan }
        if (Test-RotationPerformed $rotationMessages) {
            git add -- docs/loop/state.md docs/loop/state-archive.md
            git commit -q -m 'chore(loop): rotate ledger (old rounds -> docs/loop/state-archive.md)' | Out-Null
        }
    }

    $iter++
    if ($splitMode) {
        # Only the LOOP ledger can say whether work is owed design or code. Under another contract
        # the item phases describe a queue this run is not working, and calling the round 'impl'
        # would arm the Stop hook with a completion contract it was never given.
        $kind = if ($keepsLedger) { Get-NextRoundKind } else { Get-ContractKind $loopPrompt }
        if ($kind -eq 'impl') { $roundModel = $ImplModel; $roundEffort = $ImplEffort }
        else                  { $roundModel = $DesignModel; $roundEffort = $DesignEffort }
    } else {
        $kind = 'single'; $roundModel = $Model; $roundEffort = $Effort
    }
    $env:HONESTML_LOOP_KIND = $kind
    # -MaxTurns bounds runaway rounds (median ~110, observed peak 306 turns for a 73-test slice).
    # At 2.9-5 turns per test the ceiling covers roughly 160-275 tests - well past the ~150 planned
    # tests that LOOP.md 2a names as one of its two delivery-splitting thresholds. An aborted round
    # leaves its WIP in the tree for the resume below
    $claudeArgs = @('-p', '--dangerously-skip-permissions', '--model', $roundModel, '--effort', $roundEffort, '--max-turns', "$MaxTurns", '--output-format', 'stream-json', '--verbose')
    # A round that waited out a rate limit picks its own session back up instead of restarting over
    # its half-finished files: the context of what it was doing is worth more than a clean prompt.
    $launchArgs = $claudeArgs
    $prompt = [System.IO.File]::ReadAllText($loopPrompt, $utf8)
    if ($pendingResumeSid) {
        $launchArgs = $claudeArgs + @('--resume', $pendingResumeSid)
        $prompt = $rateLimitResumePrompt
        Write-Host "Resuming session $($pendingResumeSid.Substring(0,8)) after the rate-limit wait." -ForegroundColor Yellow
    }
    Write-Host "=== Round $iter ($(Get-Date -Format 'HH:mm:ss')) | $kind $roundModel/$roundEffort ===" -ForegroundColor Cyan
    # every relaunch of the same round number gets its own suffix, so the log of the attempt that
    # failed or was cut short by a limit survives next to the one that followed it
    $attempt = ''
    if ($transientRetries -gt 0) { $attempt += "-t$transientRetries" }
    if ($rateLimitAttempts -gt 0) { $attempt += "-a$rateLimitAttempts" }
    $outFile = Join-Path $logDir "$stamp-r$iter$attempt.out"
    $errFile = Join-Path $logDir "$stamp-r$iter$attempt.err"
    $preHead = git rev-parse HEAD
    $roundStart = Get-Date
    # how many round rows the ledger already had: the round is judged by the row IT added,
    # never by whatever sits last in the table. Captured after rotation, which removes rows.
    $priorRoundRows = @(Get-RoundTypes).Count
    $pendingResumeSid = ''

    # capture raw stream-json as UTF-8 (Tee-Object writes UTF-16 in PS 5.1); echo readable events live
    $sw = [System.IO.StreamWriter]::new($outFile, $false, $utf8); $sw.AutoFlush = $true
    $prevEAP = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        $prompt | & claude @launchArgs 2>$errFile | ForEach-Object {
            $sw.WriteLine([string]$_)
            $pretty = Format-StreamEvent $_
            if ($pretty) { Write-Host $pretty }
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEAP
        $sw.Close()
    }

    $outText = [System.IO.File]::ReadAllText($outFile, $utf8)
    $errText = [string](Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
    $postHead = git rev-parse HEAD
    Write-RoundMetrics -OutFile $outFile -Round $iter -Kind $kind -RoundModel $roundModel `
        -RoundEffort $roundEffort -ExitCode $code -HeadSha $postHead
    $tail = Get-StreamVerdictText $outText $errText

    if (Test-PolicyRefusal $tail) {
        Write-Host "Repeated Usage-Policy refusals - halting. WIP kept in the tree for inspection;" -ForegroundColor Red
        Write-Host "rework the round-contract framing before relaunching." -ForegroundColor Red
        Send-LoopNote "halted: round $iter hit repeated Usage-Policy refusals; WIP kept."
        break
    }

    if ($code -ne 0) {
        if (Test-RateLimited $tail) {
            # The round's WIP stays in the tree: a limit is a pause, not a verdict on the work. The
            # session id lets the wait end in a resume that already knows what the round was doing;
            # without one the next iteration simply starts the round afresh over the same files.
            $pendingResumeSid = Get-LastSessionId $outFile
            if (Test-Path $stopFile) { Write-Host "STOP file present - halting; WIP kept." -ForegroundColor Yellow; Send-LoopNote 'halted: STOP during rate-limit wait; WIP kept.'; break }
            $wait = Get-ResetWaitSeconds $tail $BufferMinutes $DefaultBackoffMinutes
            # A limit message whose reset time the driver could not read yields the flat default.
            # Repeating that same default can hammer a wall for hours, so blind waits double and
            # are bounded; a message that does name a reset time needs neither.
            if (Test-ResetTimeParsed $tail) {
                $backoffStreak = 0
            } else {
                $backoffStreak++
                if ($backoffStreak -gt $MaxBackoffRetries) {
                    Write-Host "Rate limit persists after $MaxBackoffRetries blind retries - halting. WIP kept." -ForegroundColor Red
                    Send-LoopNote "halted: rate limit persists after $MaxBackoffRetries blind retries; WIP kept."
                    break
                }
                $wait = [int]($wait * [math]::Pow(2, $backoffStreak - 1))
            }
            $waitH = [math]::Round($wait / 3600, 2)
            if ($wait -gt $MaxWaitHours * 3600) {
                Write-Host "Rate limit; next attempt ~$waitH h away (> -MaxWaitHours $MaxWaitHours). Exiting cleanly; WIP kept." -ForegroundColor Yellow
                Write-Host "Re-launch after reset, or schedule a relaunch (see ops/loop/README.md)." -ForegroundColor Yellow
                Send-LoopNote "halted: rate limit, next attempt ~$waitH h away (> -MaxWaitHours $MaxWaitHours); relaunch after reset."
                break
            }
            $resumeAt = (Get-Date).AddSeconds($wait).ToString('ddd HH:mm')
            Write-Host "Rate limit hit. Waiting ~$waitH h (resume ~$resumeAt), then continuing round $iter on its WIP..." -ForegroundColor Yellow
            Send-LoopNote "rate limit: waiting ~$waitH h (resume ~$resumeAt), then continuing round $iter."
            $rateLimitAttempts++
            $iter--  # same round number; the work it already did is still on disk
            Start-Sleep -Seconds $wait
            continue
        }
        if ((Test-TransientApiFailure $tail) -and $transientRetries -lt $MaxTransientRetries) {
            $transientRetries++
            Write-Host "Transport failure (exit $code); relaunching round $iter on its WIP ($transientRetries/$MaxTransientRetries)." -ForegroundColor Yellow
            $iter--  # same round number; a fresh process resumes from the files the round already wrote
            continue
        }
        Write-Host "claude exited with code $code (not a rate limit) - halting. WIP kept in the tree." -ForegroundColor Red
        Send-LoopNote "halted: round $iter, claude exited with code $code (not a rate limit); WIP kept."
        break
    }
    $transientRetries = 0
    $rateLimitAttempts = 0
    $backoffStreak = 0

    # stall guard: a completed round MUST leave a commit ("Коммит завершает раунд", LOOP.md §5).
    # exit 0 with HEAD unmoved almost always means the model ended its turn one step short — most
    # often while its own background DoD task was still running. Resume that same session ONCE with
    # a finish-the-round nudge; only a second empty result is a genuine no-progress stall. The
    # resume reuses the round's context, so it costs a fraction of redoing the round.
    if ($postHead -eq $preHead) {
        $sid = Get-LastSessionId $outFile
        if ($sid) {
            Write-Host "Round $iter left no commit - resuming session $($sid.Substring(0,8)) once to finish it..." -ForegroundColor Yellow
            $resumeFile = Join-Path $logDir "$stamp-r$iter-resume.out"
            $resumeCode = -1  # stays -1 if the pipeline throws before claude reports an exit code
            $sw2 = [System.IO.StreamWriter]::new($resumeFile, $false, $utf8); $sw2.AutoFlush = $true
            $prevEAP = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            try {
                $resumePrompt | & claude @claudeArgs '--resume' $sid 2>>$errFile | ForEach-Object {
                    $sw2.WriteLine([string]$_)
                    $pretty = Format-StreamEvent $_
                    if ($pretty) { Write-Host $pretty }
                }
                $resumeCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $prevEAP
                $sw2.Close()
            }
            $postHead = git rev-parse HEAD
            Write-RoundMetrics -OutFile $resumeFile -Round $iter -Kind "$kind-resume" -RoundModel $roundModel `
                -RoundEffort $roundEffort -ExitCode $resumeCode -HeadSha $postHead
        }
        if ($postHead -eq $preHead) {
            Write-Host "Round $iter exited green but produced NO commit (HEAD unmoved at $($preHead.Substring(0,8))) - no-progress stall, halting." -ForegroundColor Red
            Send-LoopNote "halted: round $iter exited green but produced no commit (no-progress stall)."
            break
        }
        Write-Host "Resume finished round $iter at $($postHead.Substring(0,8))." -ForegroundColor Green
    }

    # a round whose commit repeats the previous subject added no new work, only another ledger
    # row: one repetition is the whole signal, so the second identical round ends the loop
    $subject = [string](git log -1 --format=%s)
    if ($subject -eq $lastSubject) {
        Write-Host "Round $iter repeated the previous round's commit subject - no-progress loop, halting:" -ForegroundColor Red
        Write-Host "  $subject" -ForegroundColor Red
        Send-LoopNote "halted: round $iter repeated the previous commit subject (no-progress loop): $subject"
        break
    }
    $lastSubject = $subject

    # Round boundaries are the loop's only regular report: nobody watches the console for hours, and
    # a driver that speaks only when it halts leaves every finished round invisible until it stops.
    # The status is read from the row THIS round appended; a run under another contract, and a round
    # that appended no row, carry '?' rather than the verdict on someone else's work.
    $statuses = @()
    $types = @()
    if ($keepsLedger) { $statuses = @(Get-RoundStatuses); $types = @(Get-RoundTypes) }
    $recordedRow = $types.Count -gt $priorRoundRows
    $roundStatus = if ($recordedRow -and $statuses.Count -gt 0) { $statuses[-1] } else { '?' }
    $roundMinutes = [math]::Round(((Get-Date) - $roundStart).TotalMinutes)
    $roundNote = "round $iter ($kind, $roundStatus) done in $roundMinutes min: $subject"
    $volumeNote = Get-CodeVolumeNote $preHead
    if ($volumeNote) { $roundNote = "$roundNote`n$volumeNote" }
    Send-LoopNote $roundNote

    # Every verdict below is read from the rounds table of the LOOP ledger. A run driving another
    # contract has no row there to be judged by, and demanding one halts each of its rounds on a
    # rule that contract was never given. Such a round is judged by its own register and by the
    # commit checks above, which hold for every contract.
    if (-not $keepsLedger) { continue }

    # A round that committed without appending its row left the ledger describing someone else's
    # work; judging the table's last row would silently re-judge the previous round.
    if (-not $recordedRow) {
        Write-Host "Protocol violation: round $iter committed but recorded no ledger row - halting." -ForegroundColor Red
        Send-LoopNote "halted: round $iter committed without recording its ledger row."
        exit 1
    }
    $recordedType = $types[-1]
    # 'blocked-only' is a round TYPE (LOOP.md §5); the status column of such a row is 'blocked'.
    if ($recordedType -eq 'blocked-only') {
        Write-Host "Last round = blocked-only: nothing ready - halting." -ForegroundColor Yellow
        Send-LoopNote 'halted: blocked-only round, nothing ready.'
        break
    }
    # A declared obstacle earns one round. Reported again unchanged, it is not an obstacle the
    # loop can work around, and further rounds only spend budget restating it.
    if ($statuses[-1] -eq 'needs-fix') {
        $items = @(Get-RoundItems); $notes = @(Get-RoundNotes)
        $carryover = Get-ExternalBlockerCarryover $items[-1] $notes[-1]
        if ($carryover -and $carryover -eq $lastExternalBlocker) {
            Write-Host "Same external blocker declared twice in a row - halting: $carryover" -ForegroundColor Red
            Send-LoopNote "halted: external blocker declared twice unchanged ($carryover)."
            break
        }
        $lastExternalBlocker = $carryover
    } else {
        $lastExternalBlocker = ''
    }
    $terminal = Get-RoundTerminalStatuses $recordedType
    if ($terminal.Count -gt 0 -and $statuses[-1] -notin $terminal) {
        Write-Host "Protocol violation: the $recordedType round ended at '$($statuses[-1])', not $($terminal -join '/') - halting." -ForegroundColor Red
        Send-LoopNote "halted: protocol violation, the $recordedType round ended at '$($statuses[-1])'."
        exit 1
    }
}

Write-Host "Loop finished after $iter round(s). Ledger: docs/loop/state.md | blockers: docs/loop/blockers.md" -ForegroundColor Green
