# Starts the agent's web UI; first opens the SSH tunnel to the offer server
# when OFFERS_SSH_TARGET is set in .env.
#
#   powershell -ExecutionPolicy Bypass -File start.ps1
#
# .env:
#   OFFERS_SERVER_URL=http://127.0.0.1:8100   local end of the tunnel
#   OFFERS_SERVER_TOKEN=...                    from /etc/cvpilot-offer-server.env
#   OFFERS_SSH_TARGET=ubuntu@<vps-address>     empty = no tunnel
#
# The tunnel opens in its own window (type the VPS password there, if asked)
# and is closed when the agent stops.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$RemotePort = 8100  # the offer server listens on 127.0.0.1:8100 on the VPS
$WaitSeconds = 90   # time to type the password

function Read-DotEnv([string]$Path) {
    $values = @{}
    if (-not (Test-Path $Path)) { return $values }
    foreach ($line in Get-Content -Path $Path -Encoding UTF8) {
        $text = $line.Trim()
        if ($text -eq "" -or $text.StartsWith("#") -or -not $text.Contains("=")) { continue }
        $key, $value = $text -split "=", 2
        $values[$key.Trim().TrimStart([char]0xFEFF)] = $value.Trim().Trim('"').Trim("'")
    }
    return $values
}

function Test-Port([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $attempt = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        return $attempt.AsyncWaitHandle.WaitOne(500) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$envValues = Read-DotEnv (Join-Path $PSScriptRoot ".env")
$target = $envValues["OFFERS_SSH_TARGET"]
$url = $envValues["OFFERS_SERVER_URL"]
$tunnel = $null

try {
    if ($target) {
        if (-not $url) { throw "OFFERS_SSH_TARGET jest ustawiony, ale brakuje OFFERS_SERVER_URL w .env." }
        $localPort = ([System.Uri]$url).Port

        if (Test-Port $localPort) {
            Write-Host "Port $localPort juz odpowiada (tunel otwarty wczesniej), nie otwieram nowego."
        } else {
            Write-Host "Otwieram tunel do $target (port $localPort). Jesli pojawi sie pytanie o haslo, wpisz je w nowym oknie."
            $tunnel = Start-Process -FilePath "ssh" -PassThru -ArgumentList @(
                "-N",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=30",
                "-L", "${localPort}:127.0.0.1:$RemotePort",
                $target
            )
            $deadline = (Get-Date).AddSeconds($WaitSeconds)
            while (-not (Test-Port $localPort)) {
                if ($tunnel.HasExited) { throw "Tunel sie zamknal (zle haslo, adres albo port $localPort zajety)." }
                if ((Get-Date) -gt $deadline) { throw "Tunel nie odpowiada po $WaitSeconds s." }
                Start-Sleep -Milliseconds 500
            }
            Write-Host "Tunel dziala."
        }

        try {
            $health = Invoke-RestMethod -Uri ($url.TrimEnd("/") + "/health") -TimeoutSec 10
            Write-Host "Serwer ofert: $($health.offers) ofert, indeks z $($health.built_at)."
        } catch {
            Write-Warning "Serwer ofert nie odpowiada na /health: $($_.Exception.Message)"
        }
    }

    Write-Host "Agent: http://127.0.0.1:8000 (zatrzymanie: Ctrl+C)"
    uv run python -m agent_project.web
} finally {
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
        Write-Host "Tunel zamkniety."
    }
}
