# Start alphaML 2.0 in Docker and open it in the web browser (Windows).
#   start-alphaml.cmd                     double-click, or run from a terminal
#   .\start-alphaml.ps1 -Port 8800        a specific port
# Without -Port, a running alphaML keeps its port; otherwise the first free port from 8765 is used.
# Stop it with:  docker compose down
param([int]$Port = 8765)
Set-Location $PSScriptRoot

# Checks use 127.0.0.1: Windows PowerShell tries IPv6 (::1) for "localhost" first and times out, while Docker
# publishes the port on IPv4 (browsers fall back by themselves).
function Test-AlphaML([int]$p) {
    try {
        $options = Invoke-RestMethod "http://127.0.0.1:$p/api/options" -TimeoutSec 2 -ErrorAction Stop
        return [bool]($options.models -contains 'AutoML')
    } catch {
        return $false
    }
}

function Test-PortUsed([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { return $client.ConnectAsync('127.0.0.1', $p).Wait(500) } catch { return $false } finally { $client.Dispose() }
}

# Native commands report failure through exit codes (Docker writes normal progress to stderr)
docker version --format '{{.Server.Version}}' > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Docker is not running. Start Docker Desktop and try again.' -ForegroundColor Red
    exit 1
}

$running = docker port alphaml 8000 2> $null | Select-Object -First 1
if (-not $PSBoundParameters.ContainsKey('Port') -and $running -match ':(\d+)$') {
    $Port = [int]$Matches[1]  # alphaML is already running: keep its port
} elseif (-not (Test-AlphaML $Port) -and (Test-PortUsed $Port)) {
    if ($PSBoundParameters.ContainsKey('Port')) {
        Write-Host "Port $Port is used by another program. Choose another one, e.g. start-alphaml.cmd 8800" -ForegroundColor Red
        exit 1
    }
    while (Test-PortUsed $Port) { $Port++ }
    Write-Host "Port 8765 is used by another program, using port $Port."
}

$env:ALPHAML_PORT = $Port
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host 'docker compose failed, see the messages above.' -ForegroundColor Red
    exit 1
}

$url = "http://localhost:$Port"
Write-Host "Waiting for alphaML 2.0 at $url ..."
for ($i = 0; $i -lt 120; $i++) {
    if (Test-AlphaML $Port) {
        Write-Host "alphaML 2.0 is running at $url  (stop it with: docker compose down)" -ForegroundColor Green
        $data = if ($env:ALPHAML_DATA_DIR) { $env:ALPHAML_DATA_DIR } else { Join-Path $PSScriptRoot 'tmp' }
        Write-Host "Data sets and results: $data"
        Start-Process $url
        exit 0
    }
    Start-Sleep -Seconds 1
}
Write-Host "alphaML did not answer at $url. Check: docker compose ps   and   docker compose logs alphaml" -ForegroundColor Red
exit 1
