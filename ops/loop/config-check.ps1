<#
.SYNOPSIS
  Validates the two configuration surfaces a round depends on: the Claude hook config and
  skill metadata.

.DESCRIPTION
  Half of the loop's enforcement lives in .CLAUDE/settings.json, which no test reads: a hook whose
  command points at a file that is not there fails open and the round loses its only barrier. The
  other half lives in skill frontmatter, where an over-long description is accepted locally and
  rejected by anything that packages or ships the tree.

  Checked here:
    * .CLAUDE/settings.json parses, and every hook command names a file that exists once
      ${CLAUDE_PROJECT_DIR} resolves to this checkout;
    * the Bash ceiling is at least the full DoD budget, so a gate run cannot be cut short;
    * every SKILL.md in both trees carries a frontmatter with exactly `name` and `description`,
      a name matching its directory, and a description within the 1024-character limit.

  Exit code 0 when everything holds, 1 on the first set of failures.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$settingsPath = Join-Path $root '.CLAUDE\settings.json'
$descriptionLimit = 1024
# the full gate set is one foreground call and the suite outlasts a short ceiling; gates.ps1
# asks for >= 45 min in its own header
$dodBudgetMs = 2700000
$allowedFields = @('name', 'description')

$failures = @()

if (-not (Test-Path -LiteralPath $settingsPath)) {
    $failures += "missing: $settingsPath"
} else {
    try {
        $settings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        $failures += "settings.json does not parse: $($_.Exception.Message)"
        $settings = $null
    }
    if ($settings) {
        foreach ($event in $settings.hooks.PSObject.Properties) {
            foreach ($group in $event.Value) {
                foreach ($hook in $group.hooks) {
                    $command = [string]$hook.command
                    foreach ($match in [regex]::Matches($command, '"([^"]+\.py)"')) {
                        $path = $match.Groups[1].Value.Replace('${CLAUDE_PROJECT_DIR}', $root)
                        if (-not (Test-Path -LiteralPath $path)) {
                            $failures += "$($event.Name) hook points at a missing file: $($match.Groups[1].Value)"
                        }
                    }
                }
            }
        }
        $ceiling = [int]($settings.env.BASH_MAX_TIMEOUT_MS)
        if ($ceiling -lt $dodBudgetMs) {
            $failures += "BASH_MAX_TIMEOUT_MS is $ceiling ms, below the $dodBudgetMs ms a full DoD needs"
        }
    }
}

foreach ($tree in @('.CLAUDE\skills', '.agents\skills')) {
    $treeRoot = Join-Path $root $tree
    if (-not (Test-Path -LiteralPath $treeRoot)) { continue }
    foreach ($skillFile in Get-ChildItem -LiteralPath $treeRoot -Filter 'SKILL.md' -Recurse -File) {
        $rel = $skillFile.FullName.Substring($root.Length + 1) -replace '\\', '/'
        $text = Get-Content -LiteralPath $skillFile.FullName -Raw -Encoding UTF8
        $matter = [regex]::Match($text, '\A---\r?\n(?<body>.*?)\r?\n---\r?\n', 'Singleline')
        if (-not $matter.Success) {
            $failures += "$rel has no frontmatter"
            continue
        }
        $body = $matter.Groups['body'].Value
        $fields = @([regex]::Matches($body, '(?m)^([A-Za-z][A-Za-z0-9_-]*):') |
            ForEach-Object { $_.Groups[1].Value })
        foreach ($field in $fields) {
            if ($allowedFields -notcontains $field) { $failures += "$rel carries an unsupported frontmatter field: $field" }
        }
        foreach ($required in $allowedFields) {
            if ($fields -notcontains $required) { $failures += "$rel has no $required" }
        }
        $name = [regex]::Match($body, '(?m)^name:\s*(.+?)\s*$')
        $expected = $skillFile.Directory.Name
        if ($name.Success -and $name.Groups[1].Value -ne $expected) {
            $failures += "$rel names itself '$($name.Groups[1].Value)' but lives in '$expected'"
        }
        $description = [regex]::Match($body, '(?ms)^description:\s*(.*?)(?=^[A-Za-z][A-Za-z0-9_-]*:|\z)')
        if ($description.Success) {
            $value = ($description.Groups[1].Value -replace '^[>|]-?\s*', '' -replace '\s+', ' ').Trim()
            if ($value.Length -gt $descriptionLimit) {
                $failures += "$rel description is $($value.Length) chars, over the $descriptionLimit limit"
            }
        }
    }
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Output "FAIL config: $failure" }
    exit 1
}
Write-Output "PASS config (hook commands resolve, frontmatter within limits)"
exit 0
