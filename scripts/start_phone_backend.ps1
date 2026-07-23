[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$LanIp,

    [string]$PythonCommand = "python",

    [switch]$NoInstall,

    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$preflight = Join-Path $PSScriptRoot "check_demo_readiness.py"
$requirements = Join-Path $projectRoot "backend\requirements-phone.txt"
$runtimeRoot = Join-Path $projectRoot "cloud-staging\phone-backend-venv"
$runtimePython = Join-Path $runtimeRoot "Scripts\python.exe"
$checkpoint = Join-Path `
    $projectRoot `
    "ml\models\pad_hiba_convnext_tiny_source_balanced_final_seed42.pt"

try {
    $parsedAddress = [System.Net.IPAddress]::Parse($LanIp)
} catch {
    throw "LanIp must be an assigned RFC1918 IPv4 address."
}

if ($parsedAddress.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
    throw "LanIp must be an assigned RFC1918 IPv4 address."
}

$octets = $LanIp.Split(".") | ForEach-Object { [int]$_ }
$isPrivate = (
    $octets[0] -eq 10 -or
    ($octets[0] -eq 172 -and $octets[1] -ge 16 -and $octets[1] -le 31) -or
    ($octets[0] -eq 192 -and $octets[1] -eq 168)
)
if (-not $isPrivate) {
    throw "LanIp must be an assigned RFC1918 IPv4 address."
}

$assignedAddress = Get-NetIPAddress `
    -AddressFamily IPv4 `
    -IPAddress $LanIp `
    -ErrorAction SilentlyContinue
if (-not $assignedAddress) {
    throw "LanIp is not assigned to this computer."
}

$privateProfile = Get-NetConnectionProfile `
    -InterfaceIndex $assignedAddress.InterfaceIndex `
    -ErrorAction SilentlyContinue |
    Where-Object { $_.NetworkCategory -eq "Private" }
if (-not $privateProfile) {
    throw "The selected network must be marked Private before running the LAN demo."
}

$systemPython = Get-Command $PythonCommand -ErrorAction Stop
& $systemPython.Source -c `
    "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "PythonCommand must resolve to Python 3.13."
}

Push-Location $projectRoot
try {
    & $systemPython.Source $preflight --skip-runtime
    if ($LASTEXITCODE -ne 0) {
        throw "The configuration-only demo preflight failed."
    }

    if (-not (Test-Path -LiteralPath $runtimePython)) {
        if ($NoInstall) {
            throw "The pinned phone runtime is missing and NoInstall was requested."
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $runtimeRoot) -Force | Out-Null
        & $systemPython.Source -m venv $runtimeRoot
        if ($LASTEXITCODE -ne 0) {
            throw "The phone runtime could not be created."
        }
    }

    if (-not $NoInstall) {
        $requirementsLines = Get-Content -LiteralPath $requirements
        $torchPin = $requirementsLines | Where-Object { $_ -match "^torch==" }
        $torchvisionPin = $requirementsLines | Where-Object { $_ -match "^torchvision==" }
        if (@($torchPin).Count -ne 1 -or @($torchvisionPin).Count -ne 1) {
            throw "The backend runtime pins are missing or ambiguous."
        }

        & $runtimePython -m pip install `
            --disable-pip-version-check `
            --no-deps `
            --index-url https://download.pytorch.org/whl/cpu `
            $torchPin `
            $torchvisionPin
        if ($LASTEXITCODE -ne 0) {
            throw "The pinned CPU model runtime could not be installed."
        }
        & $runtimePython -m pip install `
            --disable-pip-version-check `
            -r $requirements
        if ($LASTEXITCODE -ne 0) {
            throw "The pinned API runtime could not be installed."
        }
    }

    & $runtimePython $preflight
    if ($LASTEXITCODE -ne 0) {
        throw "The exact-runtime demo preflight failed."
    }
    if ($ValidateOnly) {
        Write-Output "Validation completed; the model was not loaded."
        return
    }

    $listener = Get-NetTCPConnection `
        -State Listen `
        -LocalPort 8000 `
        -ErrorAction SilentlyContinue
    if ($listener) {
        throw "TCP port 8000 is already in use."
    }

    $environmentNames = @("MODEL_PATH", "MODEL_VERSION", "ALLOWED_HOSTS", "CORS_ORIGINS")
    $previousEnvironment = @{}
    foreach ($name in $environmentNames) {
        $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }

    try {
        $env:MODEL_PATH = $checkpoint
        $env:MODEL_VERSION = "pad-hiba-convnext-tiny-source-balanced-final-2026-07-22"
        $env:ALLOWED_HOSTS = "localhost,127.0.0.1,testserver,$LanIp"
        $env:CORS_ORIGINS = "http://localhost:8081,http://${LanIp}:8081"

        Write-Output "Starting the stateless experimental backend at http://${LanIp}:8000"
        Write-Output "Access logging is disabled. Press Ctrl+C to stop the backend."
        Write-Output "Use only synthetic, public, or otherwise non-sensitive test images."
        & $runtimePython -m uvicorn `
            backend.app.main:app `
            --host $LanIp `
            --port 8000 `
            --no-access-log `
            --no-server-header
        if ($LASTEXITCODE -ne 0) {
            throw "The phone backend exited with an error."
        }
    } finally {
        foreach ($name in $environmentNames) {
            [Environment]::SetEnvironmentVariable(
                $name,
                $previousEnvironment[$name],
                "Process"
            )
        }
    }
} finally {
    Pop-Location
}
