[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("US", "KR")]
    [string]$Market,

    [string[]]$Tickers,
    [string]$StartDate,
    [string]$EndDate,

    [switch]$FullRefresh,
    [switch]$SkipQC,
    [switch]$SkipLabels,
    [switch]$SkipNews,
    [switch]$SkipStatusReport,
    [switch]$SkipDiscord
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# In PowerShell 7+, native command stderr can become ErrorRecords when
# ErrorActionPreference=Stop. We want to rely on process exit codes instead.
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "common/Load-DotEnv.ps1")
Import-MarketEnv -Market $Market -RepoRoot $repoRoot

$marketKey = $Market.ToLowerInvariant()
$outputDir = Join-Path $repoRoot "output/$marketKey"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Push-Location $repoRoot
try {
    function Invoke-Step {
        param(
            [Parameter(Mandatory = $true)][string]$StepName,
            [Parameter(Mandatory = $true)][string[]]$Command
        )

        $outLog = Join-Path $outputDir "$StepName.out.log"
        $errLog = Join-Path $outputDir "$StepName.err.log"

        Add-Content -LiteralPath $outLog -Value ("`n=== [{0}] {1} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), ($Command -join " "))
        Add-Content -LiteralPath $errLog -Value ("`n=== [{0}] {1} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), ($Command -join " "))

        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Command[0] @($Command[1..($Command.Length - 1)]) 1>> $outLog 2>> $errLog
        }
        finally {
            $ErrorActionPreference = $prevEap
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Step '$StepName' failed with exit code $LASTEXITCODE. See $outLog / $errLog"
        }
    }

    $ingestArgs = @("-m", "src.cli", "ingest", "--market", $marketKey)
    if ($Tickers) { $ingestArgs += @("--tickers") + $Tickers }
    if ($StartDate) { $ingestArgs += @("--start-date", $StartDate) }
    if ($EndDate) { $ingestArgs += @("--end-date", $EndDate) }
    if ($FullRefresh) { $ingestArgs += "--full-refresh" }
    if ($SkipQC) { $ingestArgs += "--skip-qc" }
    Invoke-Step -StepName "ingest" -Command (@("python") + $ingestArgs)

    $processArgs = @("-m", "src.cli", "process", "--market", $marketKey)
    if ($Tickers) { $processArgs += @("--tickers") + $Tickers }
    if ($StartDate) { $processArgs += @("--start-date", $StartDate) }
    if ($EndDate) { $processArgs += @("--end-date", $EndDate) }
    if ($SkipLabels) { $processArgs += "--skip-labels" }
    Invoke-Step -StepName "process" -Command (@("python") + $processArgs)

    if (-not $SkipNews) {
        $newsArgs = @("-m", "src.cli", "news-check", "--market", $marketKey)
        if ($Tickers) { $newsArgs += @("--tickers") + $Tickers }
        Invoke-Step -StepName "news" -Command (@("python") + $newsArgs)

        $newsDedupeArgs = @("-m", "src.cli", "news-dedupe", "--market", $marketKey)
        Invoke-Step -StepName "news_dedupe" -Command (@("python") + $newsDedupeArgs)
    }

    if (-not $SkipStatusReport) {
        $statusArgs = @("-m", "src.cli", "status-report", "--market", $marketKey)
        if ($SkipDiscord) { $statusArgs += "--skip-discord" }
        Invoke-Step -StepName "status_report" -Command (@("python") + $statusArgs)
    }

    Write-Host "[$Market] Pipeline completed. Logs: $outputDir"
}
finally {
    Pop-Location
}
