[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("US", "KR")]
    [string]$Market,

    [int]$Port
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "common/Load-DotEnv.ps1")
Import-MarketEnv -Market $Market -RepoRoot $repoRoot

$marketKey = $Market.ToLowerInvariant()
$outputDir = Join-Path $repoRoot "output/$marketKey"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

if (-not $PSBoundParameters.ContainsKey("Port")) {
    $configuredPort = [Environment]::GetEnvironmentVariable("STREAMLIT_PORT", "Process")
    if ($configuredPort) {
        $Port = [int]$configuredPort
    }
    else {
        $Port = if ($Market -eq "US") { 8501 } else { 8502 }
    }
}

$outLog = Join-Path $outputDir "streamlit_web.out.log"
$errLog = Join-Path $outputDir "streamlit_web.err.log"
$cmdText = "python -m streamlit run web/app.py --server.port $Port"
Add-Content -LiteralPath $outLog -Value ("`n=== [{0}] {1} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $cmdText)
Add-Content -LiteralPath $errLog -Value ("`n=== [{0}] {1} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $cmdText)

Push-Location $repoRoot
try {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python -m streamlit run web/app.py --server.port $Port 1>> $outLog 2>> $errLog
    }
    finally {
        $ErrorActionPreference = $prevEap
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
