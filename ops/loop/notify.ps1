<#
.SYNOPSIS
  Best-effort one-way loop notification to Telegram; never fails the caller.

.DESCRIPTION
  Reads ops/loop/notify.json (git-ignored, so partial-round `git clean` spares it):
    { "telegram_bot_token": "<token from @BotFather>", "telegram_chat_id": "<chat id>" }
  A missing file or placeholder values disable delivery silently, so the drivers run
  unchanged on a machine without a configured bot. Any delivery failure is a console
  warning and nothing more: a notification must never break or stall the loop.

.PARAMETER Message
  Text sent as-is (plain text, no parse mode).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Message
)

try {
    $configPath = Join-Path $PSScriptRoot 'notify.json'
    if (-not (Test-Path -LiteralPath $configPath)) { return }
    $config = Get-Content -LiteralPath $configPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    $token = [string]$config.telegram_bot_token
    $chatId = [string]$config.telegram_chat_id
    if (-not $token -or -not $chatId -or
        $token -like 'REPLACE_WITH*' -or $chatId -like 'REPLACE_WITH*') { return }
    # Windows PowerShell 5.1 does not offer TLS 1.2 unless told to
    [System.Net.ServicePointManager]::SecurityProtocol =
        [System.Net.ServicePointManager]::SecurityProtocol -bor [System.Net.SecurityProtocolType]::Tls12
    $body = @{
        chat_id = $chatId
        text = $Message
        disable_web_page_preview = $true
    } | ConvertTo-Json -Compress
    $null = Invoke-RestMethod -Method Post -Uri "https://api.telegram.org/bot$token/sendMessage" `
        -ContentType 'application/json; charset=utf-8' `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
        -TimeoutSec 10
} catch {
    Write-Host "notify: $($_.Exception.Message)" -ForegroundColor DarkYellow
}
