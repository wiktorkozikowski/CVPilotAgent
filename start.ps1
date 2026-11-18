# Starts CVPilot locally: the web app with the login, the agent and the offer
# index in one process.
#
#   powershell -ExecutionPolicy Bypass -File start.ps1
#
# The first time, build the offer index and create the account:
#   uv run python -m offer_server.update --seed data/offers_sample_justjoin.json data/offers_sample_rocketjobs.json
#   uv run python -m agent_project.auth

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path "var/index/index.npz")) {
    Write-Host "Brak indeksu ofert, buduje go z plikow startowych..."
    uv run python -m offer_server.update --seed data/offers_sample_justjoin.json data/offers_sample_rocketjobs.json
}
if (-not (Test-Path "var/account.json")) {
    Write-Host "Brak konta. Zaloz je teraz (login i haslo, min. 10 znakow)."
    uv run python -m agent_project.auth
}

uv run python -m agent_project.web
